import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path
import multiprocessing as mp

import pymupdf
from PIL import Image

from table_extract import extract_page_table

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 42
DPI = 130
MAX_SIDE = 1400
MIN_CHARS = 50
MAX_CHARS = 5000
MAX_TABLE_SCAN = 120   # Finansal Rapor eklerinde taranacak ilk sayfa sayisi
MAX_TABLE_PAGES = 6    # belge basina secilecek tablo sayfasi


def normalize_text(s):
    # NFC + satir sonu bosluk temizligi + 3+ bos satiri 1'e indir
    s = unicodedata.normalize("NFC", s)
    # yerlesimden gelen bosluk dizilerini tek bosluga indir, satir basi/sonu bosluklarini at
    lines = [re.sub(r"[ \t ]{2,}", " ", ln).strip() for ln in s.split("\n")]
    out = []
    blank = 0
    for ln in lines:
        if ln == "":
            blank += 1
        else:
            if blank >= 3:
                out.append("")
            else:
                out.extend([""] * blank)
            blank = 0
            out.append(ln)
    if blank >= 3:
        out.append("")
    else:
        out.extend([""] * blank)
    return "\n".join(out).strip()


def render_page(page, out_path):
    # var olan dosyayi yeniden cizme
    if out_path.exists():
        try:
            with Image.open(out_path) as im:
                return im.size
        except Exception:
            pass
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return img.size


def make_record(doc, page_idx, fname, doc_id, meta, text, tbl):
    stem = Path(fname).stem
    pnum = page_idx + 1
    img_rel = f"pages/{doc_id}/{stem}_p{pnum}.png"
    img_path = DATA / img_rel
    w, h = render_page(doc[page_idx], img_path)
    return {
        "id": f"{doc_id}_{stem}_p{pnum}",
        "doc_id": doc_id,
        "file": fname,
        "page": pnum,
        "type": meta["type"],
        "company": meta["company"],
        "stock_code": meta["stockCode"],
        "image": img_rel,
        "text": text,
        "n_chars": len(text),
        "width": w,
        "height": h,
        "table": tbl,
    }


def process_doc(args):
    doc_id, meta, max_attach = args
    doc_dir = DATA / "docs" / doc_id
    rng = random.Random(f"{SEED}-{doc_id}")
    records = []

    if not doc_dir.exists():
        return doc_id, records

    # main.pdf: aday sayfalardan en fazla 3 tane
    main_path = doc_dir / "main.pdf"
    if main_path.exists():
        doc = pymupdf.open(str(main_path))
        candidates = []
        for i in range(len(doc)):
            text = normalize_text(doc[i].get_text("text", sort=True))
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                candidates.append((i, text))
        if len(candidates) > 3:
            chosen = sorted(rng.sample(candidates, 3), key=lambda x: x[0])
        else:
            chosen = candidates
        for i, text in chosen:
            records.append(make_record(doc, i, "main.pdf", doc_id, meta, text, None))
        doc.close()

    # ekler: her ekten en fazla 3, belge basina toplam max_attach
    attach_files = sorted(doc_dir.glob("attachment_*.pdf"))
    total_attach = 0
    selected_keys = set()
    for apath in attach_files:
        if total_attach >= max_attach:
            break
        doc = pymupdf.open(str(apath))
        candidates = []
        for i in range(len(doc)):
            text = normalize_text(doc[i].get_text("text", sort=True))
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                candidates.append((i, text))
        n_pick = min(3, len(candidates), max_attach - total_attach)
        if n_pick > 0:
            chosen = sorted(rng.sample(candidates, n_pick), key=lambda x: x[0])
        else:
            chosen = []
        for i, text in chosen:
            records.append(make_record(doc, i, apath.name, doc_id, meta, text, None))
            selected_keys.add((apath.name, i))
            total_attach += 1
        doc.close()

    # Finansal Rapor: eklerde tablo taramasi
    if meta["type"] == "Finansal Rapor":
        rec_index = {(r["file"], r["page"] - 1): r for r in records}
        # ilk MAX_TABLE_SCAN sayfadaki tum tablo adaylarini topla, sonra seed'li rastgele sec
        cands = []
        for apath in attach_files:
            doc = pymupdf.open(str(apath))
            for i in range(min(len(doc), MAX_TABLE_SCAN)):
                try:
                    tbl = extract_page_table(doc[i])
                except Exception:
                    tbl = None
                if tbl and len(json.dumps(tbl, ensure_ascii=False)) <= 4000:
                    cands.append((apath, i, tbl))
            doc.close()
        trng = random.Random(f"{SEED}-{doc_id}-tbl")
        for apath, i, tbl in sorted(trng.sample(cands, min(MAX_TABLE_PAGES, len(cands))), key=lambda c: (c[0].name, c[1])):
            key = (apath.name, i)
            if key in rec_index:
                # zaten secili sayfa tablo ciktiysa alani doldur
                rec_index[key]["table"] = tbl
                continue
            doc = pymupdf.open(str(apath))
            text = normalize_text(doc[i].get_text("text", sort=True))
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                rec = make_record(doc, i, apath.name, doc_id, meta, text, tbl)
                records.append(rec)
                rec_index[key] = rec
            doc.close()

    return doc_id, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-attach-pages", type=int, default=8)
    args = parser.parse_args()

    meta_rows = []
    with open(DATA / "metadata.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta_rows.append(row)

    if args.limit:
        meta_rows = meta_rows[:args.limit]

    tasks = [(row["disclosureIndex"], row, args.max_attach_pages) for row in meta_rows]

    all_records = []
    with mp.Pool(args.workers) as pool:
        for idx, (doc_id, records) in enumerate(pool.imap_unordered(process_doc, tasks)):
            all_records.append((doc_id, records))
            if (idx + 1) % 25 == 0:
                print(f"islenen belge: {idx+1}/{len(tasks)}", flush=True)

    order = {row["disclosureIndex"]: i for i, row in enumerate(meta_rows)}
    all_records.sort(key=lambda x: order.get(x[0], 0))

    out_path = DATA / "pages.jsonl"
    total_chars = 0
    table_pages = 0
    type_counts = {}
    total_pages = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc_id, records in all_records:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_pages += 1
                total_chars += rec["n_chars"]
                if rec["table"]:
                    table_pages += 1
                type_counts[rec["type"]] = type_counts.get(rec["type"], 0) + 1

    print(f"toplam sayfa: {total_pages}", flush=True)
    print("tur basina sayfa sayisi:", flush=True)
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}", flush=True)
    print(f"tablo sayfasi: {table_pages}", flush=True)
    print(f"toplam karakter: {total_chars}", flush=True)
    if total_pages:
        print(f"ortalama karakter/sayfa: {total_chars/total_pages:.1f}", flush=True)


if __name__ == "__main__":
    main()
