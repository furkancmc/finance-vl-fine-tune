"""Gercek tahminlerden 'referans / baz / LoRA' karsilastirma ornekleri uretir -> results/EXAMPLES.md

Kullanim: python make_examples.py --eval-root <v2/eval> --out results/EXAMPLES.md [--model qwen3vl_8b]
Secim kurali deterministiktir (ilk uygun kayit); ornekler elle secilmez.
"""
import argparse
import json
from pathlib import Path

from metrics import cer, normalize, strip_repetition, score_record


def load(p):
    with open(p, encoding="utf-8") as f:
        return {(r["id"], r["task"]): r for ln in [None] for r in []} or {
            (json.loads(ln)["id"], json.loads(ln)["task"]): json.loads(ln) for ln in f if ln.strip()}


def clip(s, n=520):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def block(s):
    return "```text\n" + clip(s).replace("```", "'''") + "\n```"


def tbl_block(s, n=900):
    s = (s or "").strip()
    return "```json\n" + (s if len(s) <= n else s[:n] + " …") + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", default="results/EXAMPLES.md")
    ap.add_argument("--model", default="qwen3vl_8b")
    args = ap.parse_args()
    root = Path(args.eval_root) / args.model
    md = [f"# Gerçek çıktı örnekleri ({args.model})", "",
          "Aşağıdakiler test setinden, **seçim kuralı aşağıda yazılı** otomatik seçilmiş örneklerdir (elle ayıklanmadı). "
          "Her örnekte aynı sayfa için referans (PDF metin katmanı), baz model ve LoRA çıktısı yan yana verilir. Uzun çıktılar kısaltılmıştır.", ""]

    def pick(level, cond):
        base = load(root / f"preds_base_test_{level}.jsonl")
        lora = load(root / f"preds_lora_test_{level}.jsonl")
        for key in sorted(lora):
            if key[1] != "transcribe" or key not in base:
                continue
            ref = lora[key]["ref"]
            if len(normalize(ref)) < 150 or not (400 <= len(ref) <= 1100):
                continue
            cb = cer(ref, strip_repetition(base[key]["pred"]))
            cl = cer(ref, strip_repetition(lora[key]["pred"]))
            if cond(cb, cl):
                return key, ref, base[key]["pred"], lora[key]["pred"], cb, cl
        return None

    cases = [
        ("Örnek 1 — temiz sayfa (A)", "A", lambda b, l: l < 0.02 and b > 0.10,
         "ilk sayfa: ref 400–1100 karakter, LoRA CER < 0,02, baz CER > 0,10"),
        ("Örnek 2 — orta bozulma (C)", "C", lambda b, l: l < 0.03 and b > 0.10,
         "ilk sayfa: LoRA CER < 0,03, baz CER > 0,10"),
        ("Örnek 3 — ağır bozulma (D): LoRA iyileştirir ama kusursuz değil", "D", lambda b, l: 0.05 < l < 0.30 and b > l + 0.1,
         "ilk sayfa: 0,05 < LoRA CER < 0,30 ve baz LoRA'dan en az 0,1 kötü"),
        ("Örnek 4 — ağır bozulma (D): LoRA'nın da başarısız olduğu sayfa", "D", lambda b, l: l > 0.5,
         "ilk sayfa: LoRA CER > 0,5 (başarısızlık örneği)"),
    ]
    for title, level, cond, rule in cases:
        got = pick(level, cond)
        md += [f"## {title}", ""]
        if not got:
            md += ["_Koşulu sağlayan sayfa bulunamadı._", ""]
            continue
        key, ref, pb, pl, cb, cl = got
        md += [f"Sayfa: `{key[0]}` · seviye {level} · seçim kuralı: {rule}", "",
               f"**Referans** (PDF metin katmanı)", block(ref), "",
               f"**Baz model** — CER {cb:.3f}", block(pb), "",
               f"**LoRA** — CER {cl:.3f}", block(pl), ""]

    # tablo ornegi
    for level, title in [("A", "Örnek 5 — tablo çıkarımı (A)")]:
        base = load(root / f"preds_base_test_{level}.jsonl")
        lora = load(root / f"preds_lora_test_{level}.jsonl")
        done = False
        for key in sorted(lora):
            if key[1] != "table" or key not in base:
                continue
            ref = lora[key]["ref"]
            if len(ref) > 900:
                continue
            sl = score_record("table", ref, lora[key]["pred"])
            sb = score_record("table", ref, base[key]["pred"])
            if sl.get("cell_exact", 0) >= 0.9 and sb.get("cell_exact", 0) < sl["cell_exact"] - 0.2:
                md += [f"## {title}", "", f"Sayfa: `{key[0]}` · seçim kuralı: ilk tablo sayfası: referans ≤ 900 karakter, LoRA hücre doğruluğu ≥ 0,9, baz en az 0,2 geride", "",
                       "**Referans** (PDF metin katmanından çıkarılan satırlar)", tbl_block(ref), "",
                       f"**Baz model** — hücre doğruluğu {sb.get('cell_exact', 0):.2f}, geçerli JSON: {int(sb.get('valid', 0))}", tbl_block(base[key]["pred"]), "",
                       f"**LoRA** — hücre doğruluğu {sl.get('cell_exact', 0):.2f}", tbl_block(lora[key]["pred"]), ""]
                done = True
                break
        if not done:
            md += [f"## {title}", "", "_Koşulu sağlayan sayfa bulunamadı._", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("yazildi", out, len(md))


if __name__ == "__main__":
    main()
