import argparse
import json
import re
import unicodedata
from collections import Counter

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
NUM_RE = re.compile(r"\d[\d.,]*\d|\d")


def normalize(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\r", "")
    lines = [ln.rstrip() for ln in s.split("\n")]
    out = []
    prev_blank = False
    for ln in lines:
        if ln == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        out.append(ln)
    s = "\n".join(out)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def cer(ref, hyp):
    ref = normalize(ref)
    hyp = normalize(hyp)
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    ref_w = normalize(ref).split()
    hyp_w = normalize(hyp).split()
    return Levenshtein.distance(ref_w, hyp_w) / max(len(ref_w), 1)


def turkish_char_acc(ref, hyp):
    ref = normalize(ref)
    hyp = normalize(hyp)
    total = sum(1 for c in ref if c in TR_CHARS)
    if total == 0:
        return None
    correct = 0
    for op in Levenshtein.opcodes(ref, hyp):
        if op.tag == "equal":
            seg = ref[op.src_start:op.src_end]
            correct += sum(1 for c in seg if c in TR_CHARS)
    return correct / total


def number_f1(ref, hyp):
    ref_nums = NUM_RE.findall(normalize(ref))
    hyp_nums = NUM_RE.findall(normalize(hyp))
    if not ref_nums and not hyp_nums:
        return 1.0
    rc = Counter(ref_nums)
    hc = Counter(hyp_nums)
    inter = sum((rc & hc).values())
    prec = inter / max(sum(hc.values()), 1)
    rec = inter / max(sum(rc.values()), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def parse_rows(text):
    if text is None:
        return None
    s = str(text)
    s = re.sub(r"```(?:json)?", "", s)
    s = s.replace("```", "")
    i = s.find("[")
    j = s.rfind("]")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        data = json.loads(s[i:j + 1])
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    for row in data:
        if not isinstance(row, dict):
            return None
        if "kalem" not in row or "degerler" not in row:
            return None
        if not isinstance(row["degerler"], list):
            return None
    return data


def table_scores(ref_rows, hyp_rows):
    if hyp_rows is None:
        return {
            "valid": 0,
            "row_count_match": 0,
            "label_cer": 1.0,
            "value_exact": 0.0,
            "cell_exact": 0.0,
        }

    valid = 1
    row_count_match = 1 if len(ref_rows) == len(hyp_rows) else 0

    used = set()
    matches = {}
    for i, rrow in enumerate(ref_rows):
        rlabel = normalize(str(rrow.get("kalem", "")))
        best_j = -1
        best_score = -1.0
        for j, hrow in enumerate(hyp_rows):
            if j in used:
                continue
            hlabel = normalize(str(hrow.get("kalem", "")))
            sc = fuzz.ratio(rlabel, hlabel)
            if sc > best_score:
                best_score = sc
                best_j = j
        if best_j >= 0 and best_score >= 60:
            used.add(best_j)
            matches[i] = best_j

    if ref_rows:
        cers = []
        for i, rrow in enumerate(ref_rows):
            if i in matches:
                hrow = hyp_rows[matches[i]]
                cers.append(cer(str(rrow.get("kalem", "")), str(hrow.get("kalem", ""))))
            else:
                cers.append(1.0)
        label_cer = sum(cers) / len(cers)
    else:
        label_cer = 0.0

    if ref_rows:
        cnt = 0
        for i, rrow in enumerate(ref_rows):
            if i in matches:
                hrow = hyp_rows[matches[i]]
                if list(rrow.get("degerler", [])) == list(hrow.get("degerler", [])):
                    cnt += 1
        value_exact = cnt / len(ref_rows)
    else:
        value_exact = 0.0

    total_cells = 0
    correct_cells = 0
    for i, rrow in enumerate(ref_rows):
        rvals = list(rrow.get("degerler", []))
        total_cells += len(rvals)
        if i in matches:
            hvals = list(hyp_rows[matches[i]].get("degerler", []))
            for k, rv in enumerate(rvals):
                if k < len(hvals) and hvals[k] == rv:
                    correct_cells += 1
    cell_exact = correct_cells / total_cells if total_cells > 0 else 1.0

    return {
        "valid": valid,
        "row_count_match": row_count_match,
        "label_cer": label_cer,
        "value_exact": value_exact,
        "cell_exact": cell_exact,
    }


def score_record(task, ref, pred):
    if task == "transcribe":
        return {
            "cer": cer(ref, pred),
            "wer": wer(ref, pred),
            "tr_char_acc": turkish_char_acc(ref, pred),
            "number_f1": number_f1(ref, pred),
        }
    if task == "table":
        if isinstance(ref, str):
            try:
                ref_rows = json.loads(ref)
            except Exception:
                ref_rows = []
        elif isinstance(ref, list):
            ref_rows = ref
        else:
            ref_rows = []
        hyp_rows = parse_rows(pred)
        return table_scores(ref_rows, hyp_rows)
    return {}


def summarize(records):
    groups = {}
    for rec in records:
        task = rec.get("task", "unknown")
        split = rec.get("split", "all")
        groups.setdefault((task, split), []).append(rec)

    result = {}
    for (task, split), recs in groups.items():
        scores = [score_record(task, r.get("ref", ""), r.get("pred", "")) for r in recs]
        keys = set()
        for s in scores:
            keys.update(s.keys())
        metrics = {}
        for k in sorted(keys):
            vals = [s[k] for s in scores if k in s and s[k] is not None]
            metrics[k] = (sum(vals) / len(vals)) if vals else None
        result[f"{task}|{split}"] = {"n": len(recs), "metrics": metrics}
    return result


def format_summary(summary):
    lines = []
    for key in sorted(summary.keys()):
        info = summary[key]
        lines.append(f"== {key} (n={info['n']}) ==")
        for k, v in info["metrics"].items():
            if v is None:
                lines.append(f"  {k}: -")
            else:
                lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)


def selftest():
    assert cer("kitap", "kitap") == 0
    assert abs(cer("çalışma", "calisma") - 3 / 7) < 1e-9
    assert turkish_char_acc("çalışma", "calisma") == 0.0
    assert turkish_char_acc("çalışma", "çalışma") == 1.0
    assert number_f1("1.234,56 ve 7", "7 ve 1.234,56") == 1.0
    assert parse_rows("bozuk json") is None
    assert parse_rows("```json\n[{\"kalem\":\"A\",\"dipnot\":null,\"degerler\":[1]}]\n```") is not None

    ref_rows = [{"kalem": "A", "dipnot": None, "degerler": [1, 2]}]
    hyp_rows = [{"kalem": "A", "dipnot": None, "degerler": [1, 2]}]
    s = table_scores(ref_rows, hyp_rows)
    assert s["valid"] == 1
    assert s["row_count_match"] == 1
    assert s["label_cer"] == 0
    assert s["value_exact"] == 1
    assert s["cell_exact"] == 1

    s2 = table_scores(ref_rows, None)
    assert s2["valid"] == 0
    assert s2["row_count_match"] == 0
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="?", help="preds.jsonl yolu")
    ap.add_argument("--out", help="ozet.json cikti yolu")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.preds:
        ap.error("preds.jsonl gerekli")

    records = []
    with open(args.preds, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    summary = summarize(records)
    print(format_summary(summary), flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
