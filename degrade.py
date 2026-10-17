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


def _shadow(img, rng):
    # Kenardan merkeze dogru koyulasan gradyan
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)
    direction = rng.choice(["left", "right", "top", "bottom"])
    strength = rng.uniform(0.0, 0.25)
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


def degrade(img, rng):
    # Dijital sayfa goruntusunu taranmis gibi bozar
    params = {}
    img = img.convert("RGB")
    applied = False

    # 1. gri tonlama
    if rng.random() < 0.5:
        img = img.convert("L").convert("RGB")
        params["grayscale"] = True
        applied = True

    # 2. dusuk cozunurluk
    if rng.random() < 0.7:
        s = rng.uniform(0.5, 1.0)
        w, h = img.size
        small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        img = small.resize((w, h), Image.LANCZOS)
        params["scale"] = round(s, 4)
        applied = True

    # 3. hafif egiklik
    if rng.random() < 0.8:
        angle = rng.uniform(-2.5, 2.5)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        params["angle"] = round(angle, 4)
        applied = True

    # 4. gaussian blur
    if rng.random() < 0.7:
        r = rng.uniform(0.3, 1.6)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
        params["blur"] = round(r, 4)
        applied = True

    # 5. kontrast ve parlaklik
    if rng.random() < 0.8:
        c = rng.uniform(0.7, 1.2)
        b = rng.uniform(0.85, 1.1)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Brightness(img).enhance(b)
        params["contrast"] = round(c, 4)
        params["brightness"] = round(b, 4)
        applied = True

    # 6. golge gradyani
    if rng.random() < 0.3:
        img, d, s = _shadow(img, rng)
        params["shadow"] = {"dir": d, "strength": round(s, 4)}
        applied = True

    # 7. gaussian gurultu
    if rng.random() < 0.7:
        sigma = rng.uniform(2, 18)
        np_rng = np.random.default_rng(rng.randrange(2 ** 32))
        arr = np.asarray(img).astype(np.float32)
        noise = np_rng.normal(0.0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        params["noise"] = round(sigma, 4)
        applied = True

    # 8. jpeg sikistirma
    if rng.random() < 0.9:
        q = rng.randint(35, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        params["jpeg_q"] = q
        applied = True

    # En az bir bozulma garantisi
    if not applied:
        q = rng.randint(35, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
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
        clean_path = out_dir / f"{name}_clean.png"
        img.save(clean_path)
        deg, params = degrade(img, rng)
        deg_path = out_dir / f"{name}_degraded.png"
        deg.save(deg_path)
        print(f"{name}: {json.dumps(params, ensure_ascii=False)}", flush=True)
        print(f"  clean    -> {clean_path}", flush=True)
        print(f"  degraded -> {deg_path}", flush=True)


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
