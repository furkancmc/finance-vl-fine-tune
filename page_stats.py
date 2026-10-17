"""data/pages.jsonl ve inventory.csv'den sayfa kapsama istatistigi."""
import collections
import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"

rows = [json.loads(l) for l in open(DATA / "pages.jsonl", encoding="utf-8")]
kind = collections.Counter("main" if r["file"] == "main.pdf" else "ek" for r in rows)
per_doc = collections.Counter(r["doc_id"] for r in rows)
inv = list(csv.DictReader(open(DATA / "inventory.csv", encoding="utf-8-sig")))
ek_total = sum(int(x["n_pages"]) for x in inv if x["file"] != "main.pdf")
main_total = sum(int(x["n_pages"]) for x in inv if x["file"] == "main.pdf")
long_ek = [x for x in inv if x["file"] != "main.pdf" and int(x["n_pages"]) >= 50]

print(f"toplam secilen sayfa: {len(rows)} ({len(per_doc)} belge, ort {len(rows)/len(per_doc):.1f}/belge, max {max(per_doc.values())})")
print(f"main: {kind['main']}/{main_total} sayfa, ek: {kind['ek']}/{ek_total} sayfa (%{100*kind['ek']/ek_total:.1f})")
print(f"tablo sayfasi: {sum(1 for r in rows if r['table'])}")
print(f"50+ sayfalik ek PDF: {len(long_ek)} adet, toplam {sum(int(x['n_pages']) for x in long_ek)} sayfa")
by_type = collections.Counter(r["type"] for r in rows)
print("tur basina:", dict(by_type.most_common()))
