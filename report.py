import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from statistics import median

from metrics import normalize, score_record

METRIC_KEYS = {
    "transcribe": ["cer", "wer", "tr_char_acc", "number_f1"],
    "table": ["valid", "row_count_match", "label_cer", "value_exact", "cell_exact"],
}
MEDIAN_KEYS = ["cer", "wer"]
ALL_METRICS = METRIC_KEYS["transcribe"] + METRIC_KEYS["table"] + [f"median_{k}" for k in MEDIAN_KEYS]
LEVELS = ["A", "B", "C", "D"]


def parse_filename(name):
    # preds_<tag>_<split>.jsonl ; split test_A..test_D
    m = re.match(r"^preds_(.+)_(test_[A-D])\.jsonl$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(v, digits=4):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", action="append", required=True,
                    help="bir model icin eval/<model_key> klasoru (tekrarlanabilir)")
    ap.add_argument("--out", default=None, help="rapor klasoru")
    ap.add_argument("--clean", action="store_true",
                    help="transkripsiyonda cikti sonundaki tekrar blogunu kes (metrics.strip_repetition)")
    ap.add_argument("--min-ref-chars", type=int, default=0,
                    help="referansi bu kadar karakterden kisa transkripsiyon sayfalarini disla (or. 150: sadece damga iceren sayfalar)")
    args = ap.parse_args()

    eval_dirs = [Path(p) for p in args.eval_dir]
    out_dir = Path(args.out) if args.out else eval_dirs[0].parent / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (model, tag, task, split) -> list of score dicts
    acc = defaultdict(list)
    for ed in eval_dirs:
        model = ed.name
        for f in sorted(ed.glob("preds_*.jsonl")):
            parsed = parse_filename(f.name)
            if not parsed:
                continue
            tag, split = parsed
            for rec in load_jsonl(f):
                task = rec.get("task", "transcribe")
                ref = rec.get("ref", "") or ""
                pred = rec.get("pred", "") or ""
                if task == "transcribe" and len(normalize(ref)) < args.min_ref_chars:
                    continue
                scores = score_record(task, ref, pred, clean=args.clean) or {}
                acc[(model, tag, task, split)].append(scores)

    # ortalama
    agg = {}
    for key, lst in acc.items():
        model, tag, task, split = key
        n = len(lst)
        avg = {"n": n}
        for k in METRIC_KEYS.get(task, []):
            vals = [r.get(k) for r in lst if r.get(k) is not None]
            avg[k] = sum(vals) / len(vals) if vals else None
        if task == "transcribe":
            for k in MEDIAN_KEYS:
                vals = [r.get(k) for r in lst if r.get(k) is not None]
                avg[f"median_{k}"] = median(vals) if vals else None
        agg[key] = avg

    # CSV
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tag", "task", "level", "n"] + ALL_METRICS)
        for key in sorted(agg.keys()):
            model, tag, task, split = key
            level = split.split("_", 1)[1]
            d = agg[key]
            row = [model, tag, task, level, d["n"]]
            for k in ALL_METRICS:
                v = d.get(k)
                row.append("" if v is None else f"{v:.4f}")
            w.writerow(row)

    # Markdown
    def matrix(task, metric):
        mat = defaultdict(dict)
        for (model, tag, t, split), d in agg.items():
            if t != task:
                continue
            level = split.split("_", 1)[1]
            mat[(model, tag)][level] = d.get(metric)
        return mat

    def render(title, mat):
        lines = [f"### {title}", ""]
        lines.append("| model/tag | " + " | ".join(LEVELS) + " |")
        lines.append("|---|" + "|".join(["---"] * len(LEVELS)) + "|")
        for key in sorted(mat.keys()):
            model, tag = key
            row = [f"{model}/{tag}"]
            for lv in LEVELS:
                row.append(fmt(mat[key].get(lv)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        return "\n".join(lines)

    note = f"clean={args.clean}, min_ref_chars={args.min_ref_chars}"
    md = ["# Rapor", "", f"Ayarlar: {note}", ""]
    md.append(render("Transkripsiyon medyan CER", matrix("transcribe", "median_cer")))
    md.append(render("Transkripsiyon medyan WER", matrix("transcribe", "median_wer")))
    md.append(render("Transkripsiyon CER (ortalama)", matrix("transcribe", "cer")))
    md.append(render("Transkripsiyon tr_char_acc", matrix("transcribe", "tr_char_acc")))
    md.append(render("Tablo value_exact", matrix("table", "value_exact")))
    md.append(render("Tablo valid", matrix("table", "valid")))

    # kazanc: lora - base CER (negatif iyi)
    models = sorted({m for (m, _t, tk, _s) in agg if tk == "transcribe"})
    lines = ["### Kazanc (lora - base CER, negatif iyi)", ""]
    lines.append("| model | " + " | ".join(LEVELS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(LEVELS)) + "|")
    for model in models:
        row = [model]
        for lv in LEVELS:
            split = f"test_{lv}"
            lora = agg.get((model, "lora", "transcribe", split), {}).get("cer")
            base = agg.get((model, "base", "transcribe", split), {}).get("cer")
            if lora is None or base is None:
                row.append("-")
            else:
                row.append(f"{lora - base:+.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    md.append("\n".join(lines))

    md_text = "\n".join(md)
    (out_dir / "report.md").write_text(md_text, encoding="utf-8")
    print(md_text, flush=True)


if __name__ == "__main__":
    main()
