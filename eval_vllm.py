# eval_vllm.py
# Bir modeli (baz ya da birlestirilmis LoRA) test setlerinde calistirip tahminleri yazar.
# Hizli motor vLLM, yedek motor HF generate. Metrikler ayri (metrics.py).

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

from model_zoo import get as get_model


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["id"])
            except Exception:
                continue
    return done


def make_processor(model_path, cfg, min_pixels, max_pixels):
    # vLLM'de processor sadece chat template icin; HF motorunda cozunurluk icin de kullanilir
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if cfg.get("dynamic_resolution"):
        ip = processor.image_processor
        ip.min_pixels, ip.max_pixels = min_pixels, max_pixels
        ip.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    return processor


def prepare_inputs(rows, data_dir, processor):
    """Her ornek icin (text, pil_image) uretir. Sadece user mesaji kullanilir."""
    out = []
    for r in rows:
        msgs = r["messages"][:1]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_path = os.path.join(data_dir, r["image"])
        pil = Image.open(img_path).convert("RGB")
        out.append((r, text, pil))
    return out


def load_vllm(args, cfg, model_path):
    # model bir kez yuklenir, tum split'lerde yeniden kullanilir
    from vllm import LLM

    mm_kwargs = {}
    if cfg.get("dynamic_resolution"):
        mm_kwargs = {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}
    return LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=mm_kwargs if mm_kwargs else None,
        trust_remote_code=True,
        seed=args.seed,
    )


def run_vllm(args, llm, rows, data_dir, processor, out_path, done_ids):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)

    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"[vllm] {len(todo)} ornek islenecek (toplam {len(rows)})", flush=True)
    if not todo:
        return

    prepared = prepare_inputs(todo, data_dir, processor)

    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    n_tok = 0
    CHUNK = 256
    for start in range(0, len(prepared), CHUNK):
        chunk = prepared[start:start + CHUNK]
        inputs = [{"prompt": text, "multi_modal_data": {"image": pil}} for (_, text, pil) in chunk]
        outputs = llm.generate(inputs, sp)
        for (r, _, _), o in zip(chunk, outputs):
            pred = o.outputs[0].text if o.outputs else ""
            n_out = len(o.outputs[0].token_ids) if o.outputs else 0
            rec = {
                "id": r["id"],
                "task": r.get("task"),
                "split": r.get("split"),
                "level": r.get("level"),
                "ref": r.get("ref", ""),
                "pred": pred,
                "n_tokens_out": n_out,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            n_tok += n_out
        dt = time.time() - t0
        print(f"[vllm] {n_done}/{len(prepared)}  {n_tok/max(dt,1e-6):.1f} tok/s  {n_done/max(dt,1e-6):.2f} orn/s", flush=True)
    fout.close()


def load_hf(model_path, processor):
    import torch
    from transformers import AutoModelForImageTextToText

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    processor.padding_side = "left"
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model


def run_hf(args, cfg, model, rows, data_dir, processor, out_path, done_ids):
    import torch

    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"[hf] {len(todo)} ornek islenecek (toplam {len(rows)})", flush=True)
    if not todo:
        return

    prepared = prepare_inputs(todo, data_dir, processor)

    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    n_tok = 0
    batch_size = args.hf_batch
    i = 0
    while i < len(prepared):
        chunk = prepared[i:i + batch_size]
        texts = [c[1] for c in chunk]
        images = [c[2] for c in chunk]
        if cfg.get("nested_images"):
            images = [[im] for im in images]
        try:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.inference_mode():
                gen = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            in_len = inputs["input_ids"].shape[1]
            for (r, _, _), g in zip(chunk, gen):
                out_ids = g[in_len:]
                pred = processor.tokenizer.decode(out_ids, skip_special_tokens=True)
                n_out = int(out_ids.shape[0])
                rec = {
                    "id": r["id"],
                    "task": r.get("task"),
                    "split": r.get("split"),
                    "level": r.get("level"),
                    "ref": r.get("ref", ""),
                    "pred": pred,
                    "n_tokens_out": n_out,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                n_tok += n_out
            i += batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"[hf] OOM, batch -> {batch_size}", flush=True)
                continue
            else:
                raise
        dt = time.time() - t0
        print(f"[hf] {n_done}/{len(prepared)}  {n_tok/max(dt,1e-6):.1f} tok/s  {n_done/max(dt,1e-6):.2f} orn/s", flush=True)
    fout.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="/content/dataset")
    ap.add_argument("--out", default="/content/drive/MyDrive/kap_lora/eval")
    ap.add_argument("--splits", default="test_A,test_B,test_C,test_D")
    ap.add_argument("--tasks", default="transcribe,table")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = kapali. Kacak uretim icin 1.03-1.1 denenebilir; sayi/nokta tekrarini da cezalandirir, once kucuk sette olc")
    ap.add_argument("--max-pixels", type=int, default=1500000)
    ap.add_argument("--min-pixels", type=int, default=200000)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--hf-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_model(args.model)
    model_path = args.model_path or cfg["hf_id"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tasks = set(t.strip() for t in args.tasks.split(",") if t.strip())

    processor = make_processor(model_path, cfg, args.min_pixels, args.max_pixels)

    # modeli bir kez yukle (split basina yeniden yuklemek GPU bellegini tasirir)
    engine = None
    if args.engine == "vllm":
        try:
            engine = load_vllm(args, cfg, model_path)
        except Exception as e:
            print(f"[vllm] yuklenemedi: {e}", flush=True)
            print("[vllm] Lutfen --engine hf ile tekrar deneyin.", flush=True)
            raise
    else:
        engine = load_hf(model_path, processor)

    for split in splits:
        path = os.path.join(args.data, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} yok", flush=True)
            continue
        rows = load_jsonl(path)
        if tasks:
            rows = [r for r in rows if r.get("task") in tasks]
        if args.limit and args.limit > 0:
            rows = rows[:args.limit]
        out_path = str(out_dir / f"preds_{args.tag}_{split}.jsonl")
        done_ids = load_done_ids(out_path)
        print(f"[{split}] {len(rows)} ornek, {len(done_ids)} zaten yapildi -> {out_path}", flush=True)

        if args.engine == "vllm":
            run_vllm(args, engine, rows, args.data, processor, out_path, done_ids)
        else:
            run_hf(args, cfg, engine, rows, args.data, processor, out_path, done_ids)

    # Ozet
    try:
        from metrics import summarize, format_summary
        records = []
        for split in splits:
            p = out_dir / f"preds_{args.tag}_{split}.jsonl"
            if not os.path.exists(p):
                continue
            for r in load_jsonl(str(p)):
                records.append({
                    "id": r["id"], "task": r.get("task"), "split": r.get("split"),
                    "ref": r.get("ref", ""), "pred": r.get("pred", ""),
                })
        if records:
            s = summarize(records)
            print(format_summary(s), flush=True)
    except Exception as e:
        print(f"[metrics] ozet alinamadi: {e}", flush=True)


if __name__ == "__main__":
    main()
