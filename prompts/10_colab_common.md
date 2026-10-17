# Colab scriptleri icin ortak kurallar (00_common.md'ye EK, onunla celisirse bu gecerli)

- Bu scriptler Colab A100 **80 GB** (Linux, CUDA) uzerinde calisir. Kutuphane kisiti burada GEVSEK: `torch`, `transformers>=4.57`, `peft`,
  `accelerate`, `bitsandbytes`, `unsloth`, `vllm`, `qwen-vl-utils`, `rapidfuzz`, `PIL`. Yerel Windows'ta GPU/unsloth/vllm YOKTUR;
  agir importlari (unsloth, vllm, bitsandbytes) kullanildiklari fonksiyonun/branch'in icinde yap, boylece `--dry-run` CPU'da calisir.
- Model anahtarlari `model_zoo.py`'den: `from model_zoo import get as get_model` ; `get_model(key)` -> dict
  (`hf_id, family, response_marker, nested_images, dynamic_resolution, ...`). Kendi model listesi uretme.
- Hiz hedefi: A100 80 GB'i doldur. bf16, TF32 acik (`torch.backends.cuda.matmul.allow_tf32 = True`), buyuk batch,
  `dataloader_num_workers=8`, `pin_memory`, `torch.compile` KULLANMA (kirilgan).
- Veri duzeni: `<data>/{train,val,test_A,test_B,test_C,test_D}.jsonl` ve `<data>/images/*.jpg`. Satir semasi:
  `{"id","task","split","level","image": "images/x.jpg" (data klasorune gore),
    "messages": [{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]},
                 {"role":"assistant","content":[{"type":"text","text":ANSWER}]}], "ref": ANSWER}`
- Yorumlar kisa, Turkce ve ASCII. Gereksiz soyutlama yok. Tek bir python kod blogu dondur, aciklama yazma.
