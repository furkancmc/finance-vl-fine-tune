import argparse
import json
import random
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from degrade import degrade
from metrics import normalize

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 42
MIN_REF_CHARS = 150  # bundan kisa metin = sadece SPK dogrulama damgasi, etiket degil

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


TRAIN_FRAC, VAL_FRAC = 0.7, 0.1  # kalan 0.2 test
TRAIN_N_TABLE = 500  # egitimde tablo ornegi (her tablo sayfasi bir transkripsiyon da uretir)
TRAIN_N_TEXT = 2000  # egitimde toplam transkripsiyon ornegi


def select_train_pages(pages, train_c):
    # tablo ve transkripsiyon sayilarini tam tutturur; bildirimler arasi dengeli (round-robin)
    rng = random.Random(SEED)

    def balanced(cands):
        cands = list(cands)
        rng.shuffle(cands)
        seen, ranked = {}, []
        for pg in cands:
            r = seen.get(pg["doc_id"], 0)
            seen[pg["doc_id"]] = r + 1
            ranked.append((r, rng.random(), pg))
        ranked.sort(key=lambda x: x[:2])
        return [pg for _, _, pg in ranked]

    tr = [p for p in pages if p["company"] in train_c]
    ok_text = lambda p: 0 < len(normalize(p.get("text") or "")) and len(p["text"]) <= 5000         and len(normalize(p["text"])) >= MIN_REF_CHARS
    tab = [p for p in tr if p.get("table") and ok_text(p) and len(json.dumps(p["table"], ensure_ascii=False)) <= 4000]
    chosen = balanced(tab)[:TRAIN_N_TABLE]
    ids = {p["id"] for p in chosen}
    rest = [p for p in tr if p["id"] not in ids and not p.get("table") and ok_text(p)]
    chosen += balanced(rest)[:TRAIN_N_TEXT - len(chosen)]
    return {p["id"] for p in chosen}


def split_companies(pages):
    # sirket bazinda bolme, sizinti yok
    companies = sorted({p["company"] for p in pages})
    rng = random.Random(SEED)
    rng.shuffle(companies)
    n = len(companies)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
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


def process_page(page, split, img_dir):
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

    out, dropped = [], []
    text = page.get("text") or ""
    if not text:
        dropped.append("transcribe:empty")
    elif len(text) > 5000:
        dropped.append("transcribe:too_long")
    elif len(normalize(text)) < MIN_REF_CHARS:
        dropped.append("transcribe:stamp_only")
    else:
        out.append(build_example(
            page, "transcribe", text, split, level, f"images/{img_name}", params))

    table = page.get("table")
    if table:
        ans = json.dumps(table, ensure_ascii=False)
        if len(ans) <= 4000:
            out.append(build_example(
                page, "table", ans, split, level, f"images/{img_name}", params))
        else:
            dropped.append("table:too_long")
    return split, out, dropped


def _job(args):
    page, split, img_dir = args
    return process_page(page, split, img_dir)


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

    train_ids = select_train_pages(pages, train_c)
    jobs = []
    for page in pages:
        comp = page["company"]
        if comp in train_c:
            if page["id"] in train_ids:
                jobs.append((page, "train", img_dir))
        elif comp in val_c:
            jobs.append((page, "val", img_dir))
        elif comp in test_c:
            if page["id"] not in test_ids:
                continue
            for lv in LEVELS:
                jobs.append((page, f"test_{lv}", img_dir))

    from collections import Counter
    drops = Counter()
    with Pool() as pool:
        for i, (split, recs, dropped) in enumerate(pool.imap(_job, jobs, chunksize=8), 1):
            splits[split].extend(recs)
            for d in dropped:
                drops[f"{split.split('_')[0]}/{d}"] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} is islendi", flush=True)
    print("Elenen ornekler:", dict(drops), flush=True)

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
