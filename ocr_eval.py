"""Zero-shot OCR/VL modellerini test setinde transkripsiyon icin degerlendirir (fine-tune yok).

Cikti eval_vllm.py ile ayni biçimde: <out>/preds_<tag>_<split>.jsonl (id, task, split, level, ref, pred, ...)
-> report.py ve metrics.py dogrudan okur. `pred` markup'tan arindirilmis, ham cikti `pred_raw`.
Her model ayri venv ister (transformers surumleri catisiyor), bu yuzden model ilk kullanimda yuklenir.
"""
import argparse
import html
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

PROMPTS = {
    "paddleocr_vl": "OCR:",
    "hunyuanocr": "请提取图片中的文字内容。",
    "deepseekocr": "<image>\nFree OCR. ",
}
HF_IDS = {
    "paddleocr_vl": "PaddlePaddle/PaddleOCR-VL",
    "hunyuanocr": "tencent/HunyuanOCR",
    "deepseekocr": "deepseek-ai/DeepSeek-OCR",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_done_ids(path):
    return {r["id"] + "|" + str(r.get("task")) for r in load_jsonl(path)} if os.path.exists(path) else set()


def strip_markup(s):
    # referans duz metin; OCR modelleri markdown/HTML uretebilir, bicim farki CER'e girmesin
    s = re.sub(r"</(tr|p|div|h\d|li)>|<br\s*/?>", "\n", s)
    s = re.sub(r"</t[dh]>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"^\s*\|?[\s:\-|]{3,}\|?\s*$", "", s, flags=re.M)  # markdown tablo ayraci
    s = s.replace("|", " ")
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"(\*\*|__|`)", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def load_model(name, device):
    import torch
    from transformers import AutoProcessor
    hf_id = HF_IDS[name]
    if name == "deepseekocr":  # transformers==4.46.3 + uzak kod; kendi infer() fonksiyonu var
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, use_safetensors=True,
                                              _attn_implementation="flash_attention_2")
        except Exception as e:  # flash-attn kurulu degilse
            print(f"flash_attention_2 olmadi ({e}); eager", flush=True)
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, use_safetensors=True,
                                              _attn_implementation="eager")
        return tok, model.eval().to(device).to(torch.bfloat16)
    if name == "hunyuanocr":
        from transformers import HunYuanVLForConditionalGeneration as Cls
        proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
        model = Cls.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    else:
        from transformers import AutoModelForImageTextToText  # transformers>=5 yerlesik destek, uzak kod uyumsuz
        proc = AutoProcessor.from_pretrained(hf_id)
        model = AutoModelForImageTextToText.from_pretrained(hf_id, torch_dtype=torch.bfloat16).to(device)
    return proc, model.eval()


def infer_batch(name, proc, model, imgs, max_new_tokens):
    """Doner: [(metin, uretilen_token)]. DeepSeek-OCR tek tek, digerleri toplu (sol padding)."""
    import tempfile
    import torch
    if name == "deepseekocr":
        outs = []
        for img in imgs:
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "p.jpg")
                img.save(f, quality=95)
                txt = model.infer(proc, prompt=PROMPTS[name], image_file=f, output_path=td, base_size=1024,
                                  image_size=640, crop_mode=True, save_results=False, eval_mode=True)
            outs.append((txt or "", len(proc.encode(txt or ""))))
        return outs
    tk = getattr(proc, "tokenizer", None)
    if tk is not None:
        tk.padding_side = "left"
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                          {"type": "text", "text": PROMPTS[name]}]}] for im in imgs]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    texts = proc.batch_decode(gen, skip_special_tokens=True)
    pad = tk.pad_token_id if tk is not None else None
    return [(t, int((g != pad).sum()) if pad is not None else int(g.shape[0])) for t, g in zip(texts, gen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_IDS))
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="results/ocr_preds")
    ap.add_argument("--splits", default="test_A,test_B,test_C,test_D")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--max-pixels", type=int, default=1500000, help="daha buyuk goruntuler oranli kucultulur (VRAM)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc, model = load_model(args.model, args.device)

    for split in args.splits.split(","):
        rows = [r for r in load_jsonl(os.path.join(args.data, f"{split}.jsonl")) if r.get("task") == "transcribe"]
        if args.limit:
            rows = rows[:args.limit]
        path = str(out_dir / f"preds_{args.model}_{split}.jsonl")
        done = load_done_ids(path)
        todo = [r for r in rows if r["id"] + "|transcribe" not in done]
        print(f"[{split}] {len(rows)} ornek, {len(done)} yapildi", flush=True)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as fout:
            bs = args.batch
            i = 0
            while i < len(todo):
                chunk = todo[i:i + bs]
                imgs = []
                for r in chunk:
                    img = Image.open(os.path.join(args.data, r["image"])).convert("RGB")
                    px = img.width * img.height
                    if px > args.max_pixels:
                        sc = (args.max_pixels / px) ** 0.5
                        img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
                    imgs.append(img)
                try:
                    res = infer_batch(args.model, proc, model, imgs, args.max_new_tokens)
                except Exception as e:  # OOM vb: yarim batch, tek ornekte de olmazsa hatayi yaz ve atla
                    import torch
                    torch.cuda.empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        print(f"  hata ({type(e).__name__}), batch -> {bs}", flush=True)
                        continue
                    print(f"  ATLANDI {chunk[0]['id']}: {e}", flush=True)
                    res = [("", 0)]
                for r, (raw, n_tok) in zip(chunk, res):
                    rec = {"id": r["id"], "task": "transcribe", "split": r.get("split"), "level": r.get("level"),
                           "ref": r.get("ref", ""), "pred": strip_markup(raw), "pred_raw": raw, "n_tokens_out": n_tok}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                i += len(chunk)
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.1f} sn/sayfa", flush=True)

    from metrics import summarize, format_summary
    recs = []
    for split in args.splits.split(","):
        p = out_dir / f"preds_{args.model}_{split}.jsonl"
        if p.exists():
            recs += [{k: r.get(k) for k in ("id", "task", "split", "ref", "pred")} for r in load_jsonl(str(p))]
    if recs:
        print(format_summary(summarize(recs)), flush=True)


if __name__ == "__main__":
    main()
