# merge_lora.py

LoRA adaptorunu baz modele birlestirip (bf16) vLLM'in dogrudan yukleyebilecegi tam bir model klasoru kaydeder.

## Argumanlar
`--model` (model_zoo anahtari), `--adapter` (adaptor klasoru, ornek .../final), `--out` (birlesik model klasoru).

## Davranis
- `AutoModelForImageTextToText.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map={"":0} ya da cpu; cuda yoksa cpu)`,
  `PeftModel.from_pretrained(model, adapter)`, `model = model.merge_and_unload()`.
- `model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")`, `AutoProcessor.from_pretrained(adapter ya da hf_id).save_pretrained(out)`.
  Processor adaptor klasorunde varsa onu, yoksa `hf_id`'yi kullan.
- Cikis klasoru doluysa ve icinde `config.json` varsa atla (`--force` ile yeniden yap).
- Sonunda klasor boyutunu ve dosya listesini yazdir.
