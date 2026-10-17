import argparse
import json
import random
import zipfile
from pathlib import Path

from PIL import Image

from degrade import degrade

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 42

PROMPT_TRANSCRIBE = "Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz."
PROMPT_TABLE = "Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için `kalem`, `dipnot` ve `degerler` alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir)."


def load_pages():
    pages = []
    with open(DATA / "pages.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def split_companies(pages):
    # sirket bazinda bolme, sizinti yok
    companies = sorted({p["company"] for p in pages})
    rng = random.Random(SEED)
    rng.shuffle(companies)
    n = len(companies)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = set(companies[:n_train])
    val = set(companies[n_train:n_train + n_val])
    test = set(companies[n_train + n_val:])
    return train, val, test


LEVELS = "ABCD"  # A bozulmamis, B hafif, C orta, D agir
TRAIN_LEVEL_WEIGHTS = [0.25, 0.25, 0.25, 0.25]


def make_image(src_png, dst_jpg, level, rng):
    img = Image.open(src_png).convert("RGB")
    img, params = degrade(img, rng, level)
    if not dst_jpg.exists():
        img.save(dst_jpg, "JPEG", quality=92)
    return params


def build_example(page, task, answer, split, level, image_rel, degrade_params):
    prompt = PROMPT_TRANSCRIBE if task == "transcribe" else PROMPT_TABLE
    return {
        "id": page["id"],
        "task": task,
        "split": split,
        "doc_id": page["doc_id"],
        "type": page["type"],
        "level": level,
        "degraded": level != "A",
        "image": image_rel,
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "ref": answer,
        "degrade_params": degrade_params,
    }


def process_page(page, split, out_records, img_dir):
    pid = page["id"]
    rng = random.Random(f"{SEED}-{pid}")
    src = DATA / page["image"]

    if split == "train":
        level = rng.choices(LEVELS, TRAIN_LEVEL_WEIGHTS)[0]
    elif split == "val":
        level = "A"
    else:  # test_A .. test_D
        level = split[-1]

    img_name = f"{pid}.jpg" if level == "A" else f"{pid}_{level}.jpg"
    img_path = img_dir / img_name
    params = make_image(src, img_path, level, rng)

    text = page.get("text") or ""
    if text and len(text) <= 5000:
        out_records.append(build_example(
            page, "transcribe", text, split, level, f"images/{img_name}", params))

    table = page.get("table")
    if table:
        ans = json.dumps(table, ensure_ascii=False)
        if len(ans) <= 4000:
            out_records.append(build_example(
                page, "table", ans, split, level, f"images/{img_name}", params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="test icin ilk N sayfa")
    ap.add_argument("--zip", action="store_true", help="dataset.zip uret")
    args = ap.parse_args()

    pages = load_pages()
    print(f"Toplam sayfa: {len(pages)}", flush=True)

    train_c, val_c, test_c = split_companies(pages)
    print(f"Sirket: train={len(train_c)} val={len(val_c)} test={len(test_c)}", flush=True)

    test_pages = [p for p in pages if p["company"] in test_c]
    if args.limit:
        test_pages = test_pages[:args.limit]
    test_ids = {p["id"] for p in test_pages}

    out_dir = DATA / "dataset"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], **{f"test_{lv}": [] for lv in LEVELS}}

    count = 0
    for page in pages:
        comp = page["company"]
        if comp in train_c:
            process_page(page, "train", splits["train"], img_dir)
        elif comp in val_c:
            process_page(page, "val", splits["val"], img_dir)
        elif comp in test_c:
            if page["id"] not in test_ids:
                continue
            for lv in LEVELS:
                process_page(page, f"test_{lv}", splits[f"test_{lv}"], img_dir)
        count += 1
        if count % 200 == 0:
            print(f"  {count} sayfa islendi", flush=True)

    for name, records in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(records)} ornek -> {path}", flush=True)

    stats = {}
    for name, records in splits.items():
        n = len(records)
        n_deg = sum(1 for r in records if r["degraded"])
        stats[name] = {
            "n_examples": n,
            "n_transcribe": sum(1 for r in records if r["task"] == "transcribe"),
            "n_table": sum(1 for r in records if r["task"] == "table"),
            "n_degraded": n_deg,
            "levels": {lv: sum(1 for r in records if r["level"] == lv) for lv in LEVELS},
            "avg_answer_len": (sum(len(r["ref"]) for r in records) / n) if n else 0.0,
        }
    stats["companies"] = {"train": len(train_c), "val": len(val_c), "test": len(test_c)}

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    if args.zip:
        zip_path = DATA / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            # sadece guncel split dosyalari, stats ve onlarin kullandigi goruntuler
            files = [out_dir / f"{name}.jsonl" for name in splits] + [out_dir / "stats.json"]
            files += [out_dir / img for img in sorted({r["image"] for rs in splits.values() for r in rs})]
            for p in files:
                zf.write(p, p.relative_to(DATA))
        print(f"ZIP: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
