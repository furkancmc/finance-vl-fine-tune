# eval_vllm.py

Bir modeli (baz ya da birlestirilmis LoRA) test setlerinde calistirip tahminleri yazar. Hizli motor vLLM, yedek motor HF `generate`.
Metrikler ayri (`metrics.py`, `report.py`). Bu script sadece tahmin uretir.

## Argumanlar
`--model` (model_zoo anahtari, processor/prompt icin), `--model-path` (verilmezse `hf_id`; birlestirilmis klasor yolu da olabilir),
`--tag` (ornek "base" ya da "lora"), `--data` (/content/dataset), `--out` (/content/drive/MyDrive/kap_lora/eval/<model>),
`--splits test_A,test_B,test_C,test_D`, `--tasks transcribe,table`, `--limit N`, `--engine {vllm,hf}` (varsayilan vllm),
`--max-new-tokens 3072`, `--max-pixels 1500000`, `--min-pixels 200000`, `--gpu-mem 0.90`, `--max-num-seqs 128`, `--max-model-len 8192`,
`--hf-batch 16` (hf motoru icin), `--seed 42`.

## Girdi hazirlama (iki motor icin ortak)
- Her ornekte SADECE user mesaji (goruntu + prompt) kullanilir: `messages[:1]`.
- Metin: `processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)`; processor: `AutoProcessor.from_pretrained(model_path, trust_remote_code=True,
  **({"min_pixels":..., "max_pixels":...} if cfg["dynamic_resolution"] else {}))`.
- Goruntu `PIL.Image.open(data/image).convert("RGB")`.

## vLLM motoru
- `LLM(model=model_path, dtype="bfloat16", max_model_len=..., gpu_memory_utilization=..., max_num_seqs=..., limit_mm_per_prompt={"image": 1},
  mm_processor_kwargs={"min_pixels":..., "max_pixels":...} (sadece dynamic_resolution ailesi), trust_remote_code=True, seed=seed)`.
- Girdiler: `{"prompt": text, "multi_modal_data": {"image": pil}}` listesi; `SamplingParams(temperature=0.0, max_tokens=max_new_tokens, repetition_penalty=1.0)`.
  `llm.generate(inputs, sp)`'e TUM split'in girdilerini tek seferde ver (vLLM kendi continuous batching'ini yapar). Buyuk split'te bellek icin 256'lik parcalara bol.
- Hata olursa (import/uyumsuzluk) acik mesaj bas ve `--engine hf` onerisi ver; otomatik gecme.

## HF motoru
- `AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=bf16, device_map={"":0}, attn_implementation="sdpa")`,
  processor `padding_side="left"`, `--hf-batch` boyutunda batch, `model.generate(**inputs, do_sample=False, max_new_tokens=...)`,
  yalniz uretilen tokenleri decode et. OOM'de batch'i yariya indirip tekrar dene.

## Cikti
- Her split icin `<out>/preds_<tag>_<split>.jsonl`, satir: `{"id","task","split","level","ref","pred","n_tokens_out"}`.
  Dosya varsa icindeki id'leri atla (devam edilebilir). Ilerleme ve saniye basina token/ornek yazdir.
- Sonda `metrics.summarize` ile hizli ozet bas (`from metrics import summarize, format_summary`; records `id, task, split, ref, pred`).
