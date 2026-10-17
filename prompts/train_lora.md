# train_lora.py  (Colab A100, tek GPU)

Qwen2.5-VL-7B-Instruct (HF: `Qwen/Qwen2.5-VL-7B-Instruct`) icin LoRA SFT scripti. Bu script Colab'da calisir (torch, transformers>=4.49,
peft, accelerate, qwen-vl-utils kurulu varsayilir). Bu kural dosyasindaki "sadece stdlib" kisiti BU script icin gecerli degil.
Trainer sinifi kullanma; sade ozel egitim dongusu yaz (daha az kirilgan).

## Arguman (argparse)
`--data` (varsayilan `/content/dataset`: icinde train.jsonl, val.jsonl, images/), `--out` (varsayilan `/content/drive/MyDrive/kap_lora/run1`),
`--model` (varsayilan Qwen/Qwen2.5-VL-7B-Instruct), `--epochs 2`, `--lr 1e-4`, `--batch 1`, `--grad-accum 16`, `--max-steps -1`,
`--limit N` (smoke test: ilk N ornek), `--max-pixels 1400*28*28`, `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.05`, `--seed 42`,
`--save-every 200`, `--eval-every 200`, `--tasks transcribe,table`.

## Veri
- JSONL satiri: `messages` (user: [{"type":"image"},{"type":"text","text":P}], assistant: [{"type":"text","text":A}]) ve `image`
  (`images/xxx.jpg`, `--data` altina gore). Egitimde user icerigindeki `{"type":"image"}` ogesini `{"type":"image","image":"file://<mutlak yol>"}`'e cevir.
- Tokenizasyon: `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` ile tam metin; goruntuler
  `qwen_vl_utils.process_vision_info`; `processor(text=[...], images=image_inputs, padding=True, return_tensors="pt")`.
  AutoProcessor'u `min_pixels=256*28*28, max_pixels=args.max_pixels` ile ac.
- ETIKET MASKELEME: sadece asistan cevabi (ve bitis tokeni `<|im_end|>`) loss'a girsin. Cevabin baslangicini bulmak icin ayni
  mesajlari `add_generation_prompt=True` ile ve asistan olmadan tokenlestir, prompt uzunlugunu al; labels[:prompt_len] = -100.
  Padding = -100. Batch=1 icin padding sorunu yok ama collator batch'e genel yazilsin.
- Train sirasi her epoch karisik (seed'li).

## Model ve egitim
- `Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"":0})`.
- `model.gradient_checkpointing_enable()`, `model.enable_input_require_grads()`, `use_cache=False`.
- PEFT LoRA: r, alpha, dropout; `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  (sadece dil modeli: gorsel encoder'a (`visual`) LoRA UYGULAMA; hedef modulleri `model.model.language_model` / ad filtresi ile sinirla,
  `visual` icerenleri atla). `model.print_trainable_parameters()`.
- AdamW (torch), lr cosine decay + %3 warmup, bf16, grad clip 1.0, grad accumulation.
- Her 20 adimda loss yazdir (ortalama). `--eval-every` adimda val.jsonl'den ilk 40 ornekle val loss hesapla ve yazdir.
- `--save-every` adimda ve sonda adaptoru `--out/checkpoint-<step>` ve `--out/final` olarak `model.save_pretrained` + `processor.save_pretrained`.
- `--out/train_log.jsonl`: adim, loss, lr, zaman.
- Calismaya devam: `--out` icinde `checkpoint-*` varsa en sonuncusundan LoRA agirliklarini yukleyip o adimdan devam etme (optimizer
  durumunu kaydetmek zorunda degilsin, sadece adaptor + adim sayaci json'u).
- `--limit` smoke test modunda 2-3 adimda bitip adaptoru kaydedebilmeli.
- Basta `torch.cuda` bilgisi ve ornek sayisi yazdir. Uzun ornek (token > 6000) atla ve say.
