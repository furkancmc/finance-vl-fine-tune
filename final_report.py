"""Birlesik sonuc raporu: tum modeller, bootstrap %95 guven araliklari, ortak alt kume (Llama n=100/split), grafikler.

Kullanim: python final_report.py --eval-root <v2/eval> --out results/final
Girdi: <eval-root>/<model>/preds_<tag>_test_<A-D>.jsonl (eval_vllm.py / ocr_eval*.py ciktilari)
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from metrics import normalize, score_record

LEVELS = "ABCD"
FINETUNED = ["qwen3vl_8b", "qwen25vl_7b", "internvl35_8b", "gemma3_12b", "llama32_11b"]
OCR = ["paddleocr_vl", "hunyuanocr", "deepseekocr"]
MIN_REF = 150
N_BOOT = 1000


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def boot_ci(vals, stat, n=N_BOOT, seed=42):
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    k = len(vals)
    est = stat(vals)
    bs = sorted(stat([vals[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return est, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def mean(v):
    return sum(v) / len(v)


def collect(root, ids_filter=None):
    """-> {(model, tag, level): {"cer": [...], "number_f1": [...], "cell_exact": [...], "valid": [...], "row_count_match": [...]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for mdir in sorted(Path(root).iterdir()):
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("preds_*_test_?.jsonl")):
            stem = f.stem[len("preds_"):]
            tag, letter = stem.rsplit("_test_", 1)
            split = f"test_{letter}"
            for r in load_jsonl(f):
                if ids_filter is not None and (split, r["id"], r.get("task")) not in ids_filter:
                    continue
                task, ref, pred = r.get("task", "transcribe"), r.get("ref", "") or "", r.get("pred", "") or ""
                if task == "transcribe" and len(normalize(ref)) < MIN_REF:
                    continue
                s = score_record(task, ref, pred, clean=True) or {}
                d = out[(mdir.name, tag, split)]
                for k, v in s.items():
                    if v is not None:
                        d[(task, k)].append(v)
    return out


def fmt_ci(est, lo, hi, digits=3):
    return "-" if est is None else f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def build_tables(data, title):
    lines = [f"## {title}", ""]
    keys = sorted({(m, t) for (m, t, _) in data})
    order = {m: i for i, m in enumerate(FINETUNED + OCR)}
    keys.sort(key=lambda k: (order.get(k[0], 99), k[1]))

    def table(task, metric, stat, name, digits):
        ls = [f"### {name}", "", "| model / tag | n (A) | A | B | C | D |", "|---|---|---|---|---|---|"]
        for (m, t) in keys:
            row, n0 = [], None
            for lv in LEVELS:
                vals = data.get((m, t, f"test_{lv}"), {}).get((task, metric), [])
                if lv == "A":
                    n0 = len(vals)
                row.append(fmt_ci(*boot_ci(vals, stat), digits=digits))
            if n0:
                ls.append(f"| {m} / {t} | {n0} | " + " | ".join(row) + " |")
        return ls + [""]

    lines += table("transcribe", "cer", median, "Transkripsiyon medyan CER (dusuk iyi), %95 bootstrap GA", 3)
    lines += table("transcribe", "number_f1", mean, "Transkripsiyon sayi F1 (yuksek iyi)", 3)
    lines += table("table", "cell_exact", mean, "Tablo hucre dogrulugu (yuksek iyi)", 3)
    lines += table("table", "valid", mean, "Tablo gecerli JSON orani", 2)
    lines += table("table", "row_count_match", mean, "Tablo satir sayisi eslesmesi", 2)
    return lines


def charts(data, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("grafik atlandi:", e)
        return
    models = [m for m in FINETUNED if any((m, "lora", f"test_{lv}") in data for lv in LEVELS)]
    for task, metric, stat, ylabel, fname in [
        ("transcribe", "cer", median, "Medyan CER (dusuk iyi)", "cer_by_level.png"),
        ("table", "cell_exact", mean, "Tablo hucre dogrulugu (yuksek iyi)", "table_cell_exact.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        w, x0 = 0.8 / (2 * len(models)), 0
        for i, m in enumerate(models):
            for j, tag in enumerate(["base", "lora"]):
                ys = []
                for lv in LEVELS:
                    vals = data.get((m, tag, f"test_{lv}"), {}).get((task, metric), [])
                    ys.append(stat(vals) if vals else 0)
                xs = [k + (2 * i + j - len(models) + 0.5) * w for k in range(4)]
                ax.bar(xs, ys, w, label=f"{m} {tag}", alpha=0.45 if tag == "base" else 1.0, color=f"C{i}")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"Seviye {lv}" for lv in LEVELS])
        ax.set_ylabel(ylabel); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", default="results/final")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = ["# Birlesik sonuclar (v2 veri seti)", "",
          "Tek koşu, seed 42. Tum degerler temizlenmis gorunum (sondaki tekrar blogu kesilir, referansi <150 karakter olan damga-only sayfalar disarida). "
          "Koseli parantez: %95 bootstrap guven araligi (1000 yeniden ornekleme, sayfa duzeyinde).", ""]
    full = collect(args.eval_root)
    md += build_tables(full, "Tam test seti (ilk 4 model + OCR)")

    # Llama yalnizca her split'in ilk 100 ornegi: ortak alt kume
    ids = set()
    lp = Path(args.eval_root) / "llama32_11b"
    for f in lp.glob("preds_lora_test_?.jsonl"):
        split = f.stem.rsplit("_test_", 1)[1]
        ids |= {(f"test_{split}", r["id"], r.get("task")) for r in load_jsonl(f)}
    if ids:
        sub = collect(args.eval_root, ids_filter=ids)
        md += build_tables(sub, "Ortak alt kume (Llama 3.2 ile ayni sayfalar; split basina ilk 100 ornek)")

    (out / "FINAL_RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "final_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "model", "tag", "level", "task", "metric", "n", "estimate", "ci_lo", "ci_hi"])
        for name, data in [("full", full)] + ([("common", sub)] if ids else []):
            for (m, t, split), d in sorted(data.items()):
                for (task, metric), vals in sorted(d.items()):
                    if metric in ("cer", "number_f1", "cell_exact", "valid", "row_count_match"):
                        est, lo, hi = boot_ci(vals, median if metric == "cer" else mean)
                        w.writerow([name, m, t, split[-1], task, metric, len(vals), f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    charts(full, out)
    print("yazildi:", out)


if __name__ == "__main__":
    main()
