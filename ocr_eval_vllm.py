"""Zero-shot OCR modellerini vLLM ile degerlendirir (HF generate bu modellerde batch'te hiz kazandirmiyor).

ocr_eval.py ile ayni cikti: <out>/preds_<model>_<split>.jsonl; vllm venv'inin python'uyla calistirilir.
"""
import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from ocr_eval import HF_IDS, PROMPTS, load_done_ids, load_jsonl, strip_markup


def build_llm(name, args):
    from vllm import LLM
    kw = dict(model=HF_IDS[name], trust_remote_code=True, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 1}, seed=42)
    if name == "deepseekocr":
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        kw.update(logits_processors=[NGramPerReqLogitsProcessor], enable_prefix_caching=False, mm_processor_cache_gb=0)
    return LLM(**kw)


def sampling(name, args):
    from vllm import SamplingParams
    if name == "deepseekocr":  # tekrar dongusunu kesen n-gram engelleyici (model karti onerisi)
        return SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, skip_special_tokens=False,
                              extra_args=dict(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}))
    return SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)


def run_batch(name, llm, sp, imgs):
    if name == "deepseekocr":
        inputs = [{"prompt": PROMPTS[name], "multi_modal_data": {"image": im}} for im in imgs]
        outs = llm.generate(inputs, sp)
    else:
        msgs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                              {"type": "text", "text": PROMPTS[name]}]}] for im in imgs]
        outs = llm.chat(msgs, sp)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_IDS))
    ap.add_argument("--data", default="/content/dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="test_A,test_B,test_C,test_D")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--max-pixels", type=int, default=1500000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = build_llm(args.model, args)
    sp = sampling(args.model, args)

    splits = args.splits.split(",")
    for split in splits:
        rows = [r for r in load_jsonl(os.path.join(args.data, f"{split}.jsonl")) if r.get("task") == "transcribe"]
        if args.limit:
            rows = rows[:args.limit]
        path = str(out_dir / f"preds_{args.model}_{split}.jsonl")
        done = load_done_ids(path)
        todo = [r for r in rows if r["id"] + "|transcribe" not in done]
        print(f"[{split}] {len(rows)} ornek, {len(done)} yapildi", flush=True)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as fout:
            for i in range(0, len(todo), args.chunk):
                chunk = todo[i:i + args.chunk]
                imgs = []
                for r in chunk:
                    img = Image.open(os.path.join(args.data, r["image"])).convert("RGB")
                    px = img.width * img.height
                    if px > args.max_pixels:
                        sc = (args.max_pixels / px) ** 0.5
                        img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
                    imgs.append(img)
                for r, (raw, n_tok) in zip(chunk, run_batch(args.model, llm, sp, imgs)):
                    rec = {"id": r["id"], "task": "transcribe", "split": r.get("split"), "level": r.get("level"),
                           "ref": r.get("ref", ""), "pred": strip_markup(raw), "pred_raw": raw, "n_tokens_out": n_tok}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  {min(i + args.chunk, len(todo))}/{len(todo)}  {time.time() - t0:.0f} sn", flush=True)

    from metrics import format_summary, summarize
    recs = []
    for split in splits:
        p = out_dir / f"preds_{args.model}_{split}.jsonl"
        if p.exists():
            recs += [{k: r.get(k) for k in ("id", "task", "split", "ref", "pred")} for r in load_jsonl(str(p))]
    if recs:
        print(format_summary(summarize(recs)), flush=True)


if __name__ == "__main__":
    main()
