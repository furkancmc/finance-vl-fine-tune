import os
import csv
import pymupdf

DOCS_DIR = "data/docs"
OUT_CSV = "data/inventory.csv"
TURKISH_CHARS = set("çğıİöşüÇĞÖŞÜ")


def analyze_pdf(path):
    size_kb = os.path.getsize(path) / 1024.0
    try:
        doc = pymupdf.open(path)
    except Exception as e:
        return {
            "n_pages": 0,
            "n_pages_with_text": 0,
            "n_chars": 0,
            "n_replacement": 0,
            "replacement_ratio": 0.0,
            "n_turkish": 0,
            "size_kb": round(size_kb, 2),
            "error": str(e),
        }

    n_pages = doc.page_count
    n_pages_with_text = 0
    n_chars = 0
    n_replacement = 0
    n_turkish = 0
    error = ""
    try:
        for page in doc:
            text = page.get_text()
            if len(text) >= 20:
                n_pages_with_text += 1
            n_chars += len(text)
            n_replacement += text.count("\ufffd")
            for ch in text:
                if ch in TURKISH_CHARS:
                    n_turkish += 1
    except Exception as e:
        error = str(e)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return {
        "n_pages": n_pages,
        "n_pages_with_text": n_pages_with_text,
        "n_chars": n_chars,
        "n_replacement": n_replacement,
        "replacement_ratio": n_replacement / max(n_chars, 1),
        "n_turkish": n_turkish,
        "size_kb": round(size_kb, 2),
        "error": error,
    }


def main():
    files = []
    if os.path.isdir(DOCS_DIR):
        for doc_id in sorted(os.listdir(DOCS_DIR)):
            d = os.path.join(DOCS_DIR, doc_id)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".pdf"):
                    files.append((doc_id, f, os.path.join(d, f)))

    total = len(files)
    rows = []
    for i, (doc_id, fname, path) in enumerate(files, 1):
        res = analyze_pdf(path)
        row = {"doc_id": doc_id, "file": fname}
        row.update(res)
        rows.append(row)
        if i % 100 == 0:
            print(f"Processed {i}/{total}")

    out_dir = os.path.dirname(OUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "doc_id", "file", "n_pages", "n_pages_with_text", "n_chars",
        "n_replacement", "replacement_ratio", "n_turkish", "size_kb", "error",
    ]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def summarize(subset, label):
        n_pdf = len(subset)
        n_pages = sum(r["n_pages"] for r in subset)
        n_pages_with_text = sum(r["n_pages_with_text"] for r in subset)
        no_text_ratio = (n_pages - n_pages_with_text) / max(n_pages, 1)
        high_repl = sum(1 for r in subset if r["replacement_ratio"] > 0.01)
        print(
            f"[{label}] PDFs: {n_pdf}, pages: {n_pages}, "
            f"no-text-layer ratio: {no_text_ratio:.4f}, "
            f"replacement_ratio>0.01: {high_repl}"
        )

    print("=== SUMMARY ===")
    summarize(rows, "ALL")
    summarize([r for r in rows if r["file"] == "main.pdf"], "main.pdf")
    summarize([r for r in rows if r["file"] != "main.pdf"], "attachments")


if __name__ == "__main__":
    main()
