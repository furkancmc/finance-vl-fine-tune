# evaluate_lora.py  (Colab A100)

Baz model ile LoRA'li modeli test setlerinde calistirip tahminleri `preds_*.jsonl` olarak yazar. Metrik hesabi `metrics.py` ile yapilir
(bu script sonunda `from metrics import summarize, format_summary` ile ozetler). torch/transformers/peft/qwen-vl-utils kurulu varsayilir;
"sadece stdlib" kisiti bu script icin gecerli degil.

## Arguman
`--data` (/content/dataset), `--adapter` (LoRA klasoru; verilmezse sadece baz model), `--model` (Qwen/Qwen2.5-VL-7B-Instruct),
`--splits test_clean,test_degraded`, `--out` (/content/drive/MyDrive/kap_lora/eval), `--tag` (ornek "base" ya da "lora"),
`--limit N`, `--max-new-tokens 3072`, `--batch 4`, `--max-pixels 1400*28*28`.

## Davranis
- Modeli bf16 yukle (`device_map={"":0}`); `--adapter` varsa `PeftModel.from_pretrained` ve `merge_and_unload()`.
- Her split icin `<data>/<split>.jsonl`; her ornekte sadece user mesajini (goruntu + prompt) kullan,
  `apply_chat_template(add_generation_prompt=True)`, `model.generate(do_sample=False, max_new_tokens=...)`, yalnizca uretilen tokenleri decode et.
  Batch uretimde `processor.tokenizer.padding_side = "left"`.
- Cikti: `<out>/preds_<tag>_<split>.jsonl`, satir: `{"id","task","split","ref","pred"}`. Mevcut dosya varsa icindeki id'leri atla (devam).
- Sonunda butun `preds_<tag>_*.jsonl` dosyalarini birlestirip `summarize` + `format_summary` ile yazdir ve `<out>/summary_<tag>.json` kaydet.
- OOM olursa batch'i yariya indirip o ornekleri tekrar dene.
