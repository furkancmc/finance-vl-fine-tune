import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
SEED = 42

NUM_RE = re.compile(r"^(\(\d{1,3}(\.\d{3})*(,\d+)?\)|-?\d{1,3}(\.\d{3})*(,\d+)?|-)$")
HEADER_RE = re.compile(
    r"\b(19|20)\d\d\b|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık|Dipnot|Dönem|TL\b|Türk Lirası",
    re.I)
SKIP_PREFIXES =("Dipnot", "Cari Dönem", "Önceki Dönem", "Bağımsız Denetimden")


def parse_number(token):
    # Turkce sayi bicimini int/float'a cevir
    t = token.strip()
    if t == "-":
        return 0
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    if t.startswith("-"):
        neg = True
        t = t[1:]
    t = t.replace(".", "")
    if "," in t:
        val = float(t.replace(",", "."))
    else:
        val = int(t)
    return -val if neg else val


def _lines_from_words(words):
    # kelimeleri y merkezine gore sirala, tolerans 3 pt ile satirlara grupla
    items = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        yc = (y0 + y1) / 2.0
        items.append((yc, x0, text))
    items.sort(key=lambda t: (t[0], t[1]))
    lines = []
    cur = []
    ref_y = None
    for yc, x0, text in items:
        if ref_y is None or abs(yc - ref_y) <= 3:
            cur.append((x0, text))
            if ref_y is None:
                ref_y = yc
        else:
            cur.sort(key=lambda t: t[0])
            lines.append(cur)
            cur = [(x0, text)]
            ref_y = yc
    if cur:
        cur.sort(key=lambda t: t[0])
        lines.append(cur)
    return lines


def _count_trailing_numbers(tokens):
    # sagdan baslayarak ardisik sayi tokenlarini say
    n = 0
    for i in range(len(tokens) - 1, -1, -1):
        if NUM_RE.match(tokens[i][1]):
            n += 1
        else:
            break
    return n


def extract_page_table(page):
    words = page.get_text("words")
    if not words:
        return None
    lines = _lines_from_words(words)

    line_info = []
    for tokens in lines:
        n = _count_trailing_numbers(tokens)
        line_info.append((tokens, n))

    # ncols = n >= 2 olan satirlarin n degerlerinin modu
    ns = [n for _, n in line_info if n >= 2]
    if not ns:
        return None
    ncols = Counter(ns).most_common(1)[0][0]

    # en az 8 satir n >= ncols olmali
    count = sum(1 for _, n in line_info if n >= ncols)
    if count < 8:
        return None

    rows = []
    pending = []
    for tokens, n in line_info:
        if n < ncols:
            # sayi icermeyen satirlar pending'e girer (en fazla 2 satir, 120 karakter)
            if n == 0:
                text = " ".join(t[1] for t in tokens)
                # sayfa basligi, tarih satiri ve cumle artiklari etikete yapismasin
                if HEADER_RE.search(text) or text.endswith(":") or len(text) > 80:
                    pending = []
                else:
                    pending = [text]
            continue

        label_tokens = tokens[:len(tokens) - n]
        value_tokens = tokens[len(tokens) - n:]

        label = " ".join(t[1] for t in label_tokens)
        if pending:
            label = " ".join(pending) + " " + label
            pending = []

        if n == ncols + 1:
            dipnot = value_tokens[0][1]
            vals = value_tokens[1:]
        else:
            dipnot = None
            vals = value_tokens[-ncols:]

        values = [parse_number(t[1]) for t in vals]
        rows.append({"kalem": label, "dipnot": dipnot, "degerler": values})

    # eleme
    filtered = []
    for r in rows:
        label = r["kalem"].strip()
        if len(label) < 3:
            continue
        if not label and all(isinstance(v, int) and 0 <= v <= 31 for v in r["degerler"]):
            continue
        if any(label.startswith(p) for p in SKIP_PREFIXES):
            continue
        label = unicodedata.normalize("NFC", label)
        label = re.sub(r"\s+", " ", label).strip()
        r["kalem"] = label
        filtered.append(r)

    if len(filtered) < 6 or len(filtered) > 120:
        return None
    return filtered


def scan():
    docs_dir = ROOT / "data" / "docs"
    pdfs = sorted(docs_dir.glob("*/attachment_*.pdf"))[:40]
    total_pages = 0
    table_pages = 0
    row_counts = []
    ncols_counts = []
    for pdf in pdfs:
        try:
            doc = pymupdf.open(pdf)
        except Exception:
            continue
        try:
            for page in doc:
                total_pages += 1
                rows = extract_page_table(page)
                if rows:
                    table_pages += 1
                    row_counts.append(len(rows))
                    ncols_counts.append(len(rows[0]["degerler"]))
        finally:
            doc.close()
    print(f"Taranan PDF: {len(pdfs)}")
    print(f"Toplam sayfa: {total_pages}")
    print(f"Tablo bulunan sayfa: {table_pages}")
    if row_counts:
        print(f"Ortalama satir: {sum(row_counts) / len(row_counts):.1f}")
        print(f"ncols dagilimi: {Counter(ncols_counts).most_common()}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="KAP PDF tablo cikarici")
    parser.add_argument("pdf", nargs="?", help="PDF dosyasi")
    parser.add_argument("page", nargs="?", type=int, help="Sayfa no (1 tabanli)")
    parser.add_argument("--scan", action="store_true", help="Ornek PDF'leri tara")
    args = parser.parse_args()

    if args.scan:
        scan()
        return

    if not args.pdf:
        parser.error("pdf dosyasi gerekli")

    doc = pymupdf.open(args.pdf)
    try:
        if args.page:
            if args.page < 1 or args.page > len(doc):
                print(f"Gecersiz sayfa: {args.page}")
                return
            page = doc[args.page - 1]
            rows = extract_page_table(page)
            if rows is None:
                print("Tablo bulunamadi")
            else:
                print(json.dumps(rows, ensure_ascii=False, indent=1))
        else:
            for i in range(len(doc)):
                rows = extract_page_table(doc[i])
                if rows:
                    print(f"Sayfa {i + 1}: {len(rows)} satir")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
