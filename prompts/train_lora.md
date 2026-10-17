# train_lora.py

VL modelleri icin LoRA SFT. Tek script, iki arka uc: `unsloth` (hizli) ve `hf` (saf transformers + peft). Ortak kod: veri seti, collator, Trainer.

## Argumanlar (argparse)
`--model` (model_zoo anahtari, zorunlu), `--data` (/content/dataset), `--out` (/content/drive/MyDrive/kap_lora/<model>), 
`--backend {unsloth,hf}` (varsayilan unsloth), `--load-in-4bit` (QLoRA, bitsandbytes; varsayilan kapali, 80 GB'ta bf16 LoRA tercih),
`--epochs 3`, `--lr 2e-4`, `--batch 8` (cihaz basina), `--grad-accum 4`, `--max-steps -1`, `--limit N` (ilk N egitim ornegi, smoke test),
`--max-pixels 1500000`, `--min-pixels 200000`, `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.0`, `--train-vision` (gorsel encoder'a da LoRA; varsayilan kapali),
`--optim adamw_8bit` (bitsandbytes 8-bit AdamW; alternatif `adamw_torch_fused`), `--no-grad-ckpt` (gradient checkpointing kapat; varsayilan acik),
`--attn {sdpa,flash_attention_2}` (hf arka ucu icin, varsayilan sdpa), `--workers 8`, `--eval-steps 50`, `--save-steps 100`,
`--max-eval 200` (val'den ilk N ornek), `--tasks transcribe,table`, `--seed 42`, `--resume`, `--dry-run`.

## Veri seti ve collator (modulun ust duzeyinde, test edilebilir)
- `class JsonlVLDataset(torch.utils.data.Dataset)`: jsonl'i bellege oku (liste), `tasks` filtresi, `limit`. `__getitem__` ham dict doner.
- `build_collator(processor, cfg, data_dir, max_pixels, min_pixels)` -> callable(batch)->dict tensors. Her ornek icin:
  1. `PIL.Image.open(data_dir/image).convert("RGB")`.
  2. messages icindeki `{"type":"image"}` oge(ler)i aynen kalir (processor chat template'i goruntu yer tutucusunu uretir).
  3. Metin: `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` (cevap dahil tam konusma).
  4. Tum batch: `processor(text=texts, images=imgs, padding=True, return_tensors="pt")`; `imgs` = `[[img],...]` eger `cfg["nested_images"]`, degilse `[img, ...]`.
     Padding tarafi sag (`processor.tokenizer.padding_side="right"`).
  5. Maskeleme: `cfg["response_marker"]`'i `processor.tokenizer(marker, add_special_tokens=False).input_ids` ile tokenle; her satirda bu
     dizinin SON gecisini bul, ondan (dahil) onceki tum konumlari -100 yap. Padding konumlari -100. Marker bulunamazsa o ornegi tamamen
     -100 yap ve sayaca ekle (uyari bas). Bitis tokeni (ornek `<|im_end|>`) loss'ta KALSIN.
  6. `dynamic_resolution` olan ailelerde processor `min_pixels`/`max_pixels` ile acilir (asagida `load_processor`).
- `load_processor(cfg, min_pixels, max_pixels)`: `AutoProcessor.from_pretrained(cfg["hf_id"], **kwargs)`; dynamic_resolution ise
  `min_pixels`, `max_pixels` kwargs; `processor.tokenizer.padding_side = "right"`.

## --dry-run (CPU, model YUKLEMEZ)
Yalnizca processor + veri + collator: ilk `--batch` ornekten bir batch kur; sekil, token sayisi, etiketli (maskelenmemis) token orani, 
batch icindeki min/max uzunluk, ilk ornegin etiketli kismini decode edip ilk 200 karakterini yazdir, maskelenemeyen ornek sayisini yazdir.
Sonra cik. (`import unsloth` yapma.)

## Modeller
- `unsloth` arka ucu: `from unsloth import FastVisionModel` (transformers/peft'ten ONCE import edilmeli: dosya basinda degil, `main` icinde,
  ama `import transformers` / `peft` cagrilarindan once). 
  `model, tok = FastVisionModel.from_pretrained(cfg["hf_id"], load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth" (grad ckpt acik ise yoksa False))`;
  `model = FastVisionModel.get_peft_model(model, finetune_vision_layers=args.train_vision, finetune_language_layers=True,
   finetune_attention_modules=True, finetune_mlp_modules=True, r=..., lora_alpha=..., lora_dropout=..., bias="none", random_state=seed)`;
  `FastVisionModel.for_training(model)`. Donen `tok` processor'dur; collator'a onu ver (yine de padding_side=right ayarla).
- `hf` arka ucu: `AutoModelForImageTextToText.from_pretrained(cfg["hf_id"], torch_dtype=torch.bfloat16, attn_implementation=args.attn, device_map={"":0})`,
  4-bit ise `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` ve
  `prepare_model_for_kbit_training`. LoRA hedefleri: modeldeki `nn.Linear` yaprak modul adlarindan, adi
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` ile BITEN ve (train-vision yoksa) adinda `visual`, `vision`, `multi_modal_projector`, `merger` GECMEYENLER;
  tam adlari `LoraConfig(target_modules=liste)`. Gradient checkpointing acik ise `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`
  ve `model.enable_input_require_grads()`. `model.config.use_cache = False`.
- Her iki arka uc: `model.print_trainable_parameters()`.

## Egitim (HF Trainer, TRL YOK)
`TrainingArguments(output_dir=out, per_device_train_batch_size=batch, per_device_eval_batch_size=batch, gradient_accumulation_steps, num_train_epochs,
max_steps, learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01, bf16=True, tf32=True, optim=args.optim, logging_steps=5,
eval_strategy="steps", eval_steps, save_strategy="steps", save_steps, save_total_limit=3, dataloader_num_workers=workers, dataloader_pin_memory=True,
remove_unused_columns=False, report_to="none", seed, gradient_checkpointing=False (kendimiz yonetiyoruz), max_grad_norm=1.0)`.
`Trainer(model, args, train_dataset, eval_dataset=val (ilk --max-eval), data_collator=collator)`. `--resume` ise `trainer.train(resume_from_checkpoint=True)`
(checkpoint yoksa normal basla). Bitince `trainer.save_model(out/"final")` + `processor.save_pretrained(out/"final")`;
`out/run_config.json` (argumanlar, model cfg, ornek sayilari, egitim suresi, tepe GPU bellegi `torch.cuda.max_memory_allocated`),
`out/train_log.jsonl` (trainer.state.log_history).
- Basta GPU adi/bellegi, ornek sayilari, etkin batch (batch*accum), tahmini adim sayisini yazdir.
- `--limit` kullanildiginda eval'i kucult, 3 adimlik smoke test icin `--max-steps 3` ile calismali.
