import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 42


def _shadow(img, rng, max_strength=0.25):
    # Kenardan merkeze dogru koyulasan gradyan
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)
    direction = rng.choice(["left", "right", "top", "bottom"])
    strength = rng.uniform(0.0, max_strength)
    if direction in ("left", "right"):
        ramp = np.linspace(1.0 - strength, 1.0, w, dtype=np.float32)
        if direction == "right":
            ramp = ramp[::-1]
        mask = ramp[None, :, None]
    else:
        ramp = np.linspace(1.0 - strength, 1.0, h, dtype=np.float32)
        if direction == "bottom":
            ramp = ramp[::-1]
        mask = ramp[:, None, None]
    arr = np.clip(arr * mask, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB"), direction, float(strength)


# Bozulma seviyeleri: A hic, B hafif, C orta, D agir (zar zor okunur).
# Her adim: (olasilik, aralik...). jpeg_q: (alt, ust) kalite, dusuk kalite = cok bozuk.
LEVELS = {
    "B": dict(gray=0.3, scale=(0.5, 0.85, 1.0), angle=(0.6, 0.8), blur=(0.5, 0.2, 0.6),
              enhance=(0.5, 0.9, 1.1, 0.95, 1.05), shadow=(0.1, 0.08), noise=(0.5, 1, 5), jpeg=(0.8, 75, 92)),
    "C": dict(gray=0.5, scale=(0.8, 0.6, 0.85), angle=(0.8, 2.5), blur=(0.7, 0.6, 1.4),
              enhance=(0.8, 0.75, 1.15, 0.88, 1.08), shadow=(0.3, 0.2), noise=(0.7, 5, 14), jpeg=(0.9, 45, 75)),
    "D": dict(gray=0.6, scale=(1.0, 0.4, 0.55), angle=(0.9, 4.0), blur=(0.9, 1.0, 1.8),
              enhance=(0.9, 0.55, 0.85, 0.75, 1.1), shadow=(0.6, 0.4), noise=(0.9, 14, 28), jpeg=(1.0, 12, 35)),
}


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade(img, rng, level="C"):
    # Dijital sayfa goruntusunu seviyeye gore taranmis gibi bozar (A = dokunma)
    params = {"level": level}
    img = img.convert("RGB")
    if level == "A":
        return img, params
    cfg = LEVELS[level]

    if rng.random() < cfg["gray"]:
        img = img.convert("L").convert("RGB")
        params["grayscale"] = True

    p, lo, hi = cfg["scale"]
    if rng.random() < p:
        s = rng.uniform(lo, hi)
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        img = small.resize((w, h), Image.LANCZOS)
        params["scale"] = round(s, 4)

    p, amax = cfg["angle"]
    if rng.random() < p:
        angle = rng.uniform(-amax, amax)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        params["angle"] = round(angle, 4)

    p, lo, hi = cfg["blur"]
    if rng.random() < p:
        r = rng.uniform(lo, hi)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
        params["blur"] = round(r, 4)

    p, clo, chi, blo, bhi = cfg["enhance"]
    if rng.random() < p:
        c, b = rng.uniform(clo, chi), rng.uniform(blo, bhi)
        img = ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(c)).enhance(b)
        params["contrast"], params["brightness"] = round(c, 4), round(b, 4)

    p, smax = cfg["shadow"]
    if rng.random() < p:
        img, d, s = _shadow(img, rng, smax)
        params["shadow"] = {"dir": d, "strength": round(s, 4)}

    p, lo, hi = cfg["noise"]
    if rng.random() < p:
        sigma = rng.uniform(lo, hi)
        np_rng = np.random.default_rng(rng.randrange(2 ** 32))
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + np_rng.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        params["noise"] = round(sigma, 4)

    p, qlo, qhi = cfg["jpeg"]
    if rng.random() < p:
        q = rng.randint(qlo, qhi)
        img = _jpeg(img, q)
        params["jpeg_q"] = q

    if len(params) == 1:  # hicbir adim secilmediyse en azindan jpeg uygula
        q = rng.randint(cfg["jpeg"][1], cfg["jpeg"][2])
        img = _jpeg(img, q)
        params["jpeg_q"] = q
    return img, params


def _collect_demo_items(limit=3):
    items = []
    pages_jsonl = DATA / "pages.jsonl"
    if pages_jsonl.exists():
        with open(pages_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_path = DATA / obj["image"]
                if img_path.exists():
                    items.append((obj["id"], img_path))
                if len(items) >= limit:
                    break
    if not items:
        pages_dir = DATA / "pages"
        if pages_dir.exists():
            for p in sorted(pages_dir.rglob("*.png")):
                items.append((p.stem, p))
                if len(items) >= limit:
                    break
    return items


def demo():
    out_dir = DATA / "degrade_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    items = _collect_demo_items(3)
    if not items:
        print("Demo icin sayfa bulunamadi.", flush=True)
        return

    for name, path in items:
        img = Image.open(path).convert("RGB")
        for level in "ABCD":
            deg, params = degrade(img, random.Random(f"{SEED}-{name}"), level)
            out = out_dir / f"{name}_{level}.png"
            deg.save(out)
            print(f"{name} {level}: {json.dumps(params, ensure_ascii=False)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Sayfa goruntusu bozma araci")
    ap.add_argument("--demo", action="store_true", help="Ilk 3 sayfayi bozup kaydet")
    args = ap.parse_args()
    if args.demo:
        demo()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
