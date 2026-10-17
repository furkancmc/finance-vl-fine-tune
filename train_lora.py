import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from model_zoo import get as get_model


class JsonlVLDataset(Dataset):
    def __init__(self, path, tasks=None, limit=None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if tasks and obj.get("task") not in tasks:
                    continue
                self.rows.append(obj)
                if limit and len(self.rows) >= limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def load_processor(cfg, min_pixels, max_pixels):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg["hf_id"])
    if cfg.get("dynamic_resolution"):
        # transformers 4.x ve 5.x'te calisan yol: islemci ozniteliklerini dogrudan ayarla
        ip = processor.image_processor
        try:
            ip.min_pixels, ip.max_pixels = min_pixels, max_pixels
        except AttributeError:
            pass  # transformers 5.x: min/max_pixels salt okunur, size'dan turetilir
        ip.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    processor.tokenizer.padding_side = "right"
    return processor


def build_collator(processor, cfg, data_dir, max_pixels, min_pixels):
    data_dir = Path(data_dir)
    marker_ids = processor.tokenizer(cfg["response_marker"], add_special_tokens=False).input_ids
    nested = bool(cfg.get("nested_images", False))
    stats = {"unmaskable": 0}

    def collate(batch):
        texts = []
        imgs = []
        for ex in batch:
            img = Image.open(data_dir / ex["image"]).convert("RGB")
            text = processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            imgs.append(img)
        images_arg = [[im] for im in imgs] if nested else imgs
        inputs = processor(text=texts, images=images_arg, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        mlen = len(marker_ids)
        for i in range(labels.shape[0]):
            row = inputs["input_ids"][i].tolist()
            pos = -1
            if mlen > 0:
                for j in range(len(row) - mlen, -1, -1):
                    if row[j:j + mlen] == marker_ids:
                        pos = j
                        break
            if pos < 0:
                labels[i, :] = -100
                stats["unmaskable"] += 1
            else:
                labels[i, :pos + mlen] = -100  # isaretin kendisi de loss disi
        inputs["labels"] = labels
        return inputs

    collate.stats = stats
    return collate


def dry_run(args, cfg, tasks):
    processor = load_processor(cfg, args.min_pixels, args.max_pixels)
    data_dir = Path(args.data)
    ds = JsonlVLDataset(data_dir / "train.jsonl", tasks=tasks, limit=args.batch)
    collator = build_collator(processor, cfg, data_dir, args.max_pixels, args.min_pixels)
    n = min(args.batch, len(ds))
    if n == 0:
        print("no samples", flush=True)
        return
    batch = [ds[i] for i in range(n)]
    out = collator(batch)
    ids = out["input_ids"]
    labels = out["labels"]
    print(f"batch={ids.shape[0]} seq={ids.shape[1]}", flush=True)
    lens = out["attention_mask"].sum(dim=1).tolist()
    print(f"len min={min(lens)} max={max(lens)}", flush=True)
    total_lab = int((labels != -100).sum().item())
    total_tok = int((out["attention_mask"] == 1).sum().item())
    print(f"labeled={total_lab}/{total_tok} ratio={total_lab / max(1, total_tok):.3f}", flush=True)
    row = labels[0]
    idx = (row != -100).nonzero(as_tuple=True)[0]
    if len(idx) > 0:
        s = int(idx[0].item())
        e = int(idx[-1].item()) + 1
        txt = processor.tokenizer.decode(row[s:e], skip_special_tokens=False)
        print(f"first_labeled[:200]={txt[:200]!r}", flush=True)
    print(f"unmaskable={collator.stats['unmaskable']}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="/content/dataset")
    p.add_argument("--out", default=None)
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=1500000)
    p.add_argument("--min-pixels", type=int, default=200000)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--train-vision", action="store_true")
    p.add_argument("--optim", default="adamw_8bit")
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--attn", choices=["sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--max-eval", type=int, default=200)
    p.add_argument("--tasks", default="transcribe,table")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = get_model(args.model)
    if args.out is None:
        args.out = f"/content/drive/MyDrive/kap_lora/{args.model}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.dry_run:
        dry_run(args, cfg, tasks)
        return

    # unsloth transformers/peft'ten once import edilmeli
    if args.backend == "unsloth":
        from unsloth import FastVisionModel
    from transformers import Trainer, TrainingArguments

    processor = load_processor(cfg, args.min_pixels, args.max_pixels)

    if args.backend == "unsloth":
        model, tok = FastVisionModel.from_pretrained(
            cfg["hf_id"],
            load_in_4bit=args.load_in_4bit,
            use_gradient_checkpointing="unsloth" if not args.no_grad_ckpt else False,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=args.train_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
        )
        FastVisionModel.for_training(model)
        processor = tok
        processor.tokenizer.padding_side = "right"
    else:
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": args.attn,
            "device_map": {"": 0},
        }
        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForImageTextToText.from_pretrained(cfg["hf_id"], **kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        skip = ["visual", "vision", "multi_modal_projector", "merger"]
        leaf_ok = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        targets = set()
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in leaf_ok:
                    if not args.train_vision and any(s in name for s in skip):
                        continue
                    targets.add(name)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=sorted(targets),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if not args.no_grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        model.config.use_cache = False

    model.print_trainable_parameters()

    data_dir = Path(args.data)
    train_ds = JsonlVLDataset(data_dir / "train.jsonl", tasks=tasks, limit=args.limit)
    val_ds = JsonlVLDataset(data_dir / "val.jsonl", tasks=tasks, limit=args.max_eval)
    collator = build_collator(processor, cfg, data_dir, args.max_pixels, args.min_pixels)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} mem={props.total_memory / 1e9:.1f}GB", flush=True)
    eff = args.batch * args.grad_accum
    if args.max_steps > 0:
        steps = args.max_steps
    else:
        steps = int((len(train_ds) * args.epochs) // max(1, eff))
    print(f"train={len(train_ds)} val={len(val_ds)} eff_batch={eff} steps~{steps}", flush=True)

    # transformers 4.x: warmup_ratio, 5.x: warmup_steps (0-1 arasi kesir oran sayilir)
    import inspect
    warm = ({"warmup_ratio": 0.05} if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
            else {"warmup_steps": 0.05})
    targs = TrainingArguments(
        **warm,
        output_dir=str(out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        tf32=True if torch.cuda.is_available() else None,
        optim=args.optim,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    # sadece checkpoint varsa devam et; gercek hatalar yutulmaz
    has_ckpt = any(out.glob("checkpoint-*"))
    if args.resume and has_ckpt:
        print("checkpoint bulundu, kaldigi yerden devam ediliyor", flush=True)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    dur = time.time() - t0

    final = out / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))

    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    run_cfg = {
        "args": vars(args),
        "model_cfg": cfg,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "duration_sec": dur,
        "peak_gpu_bytes": peak,
    }
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)
    with open(out / "train_log.jsonl", "w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"done in {dur:.1f}s peak={peak / 1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
