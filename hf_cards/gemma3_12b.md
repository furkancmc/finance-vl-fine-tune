---
language: tr
license: gemma
library_name: peft
base_model: google/gemma-3-12b-it
pipeline_tag: image-text-to-text
tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]
---

# Gemma-3-12B-KAP-TR-LoRA

**Gemma 3 12B-it** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).

> Bu bir araştırma/öğrenme projesidir; üretim kullanımı için uygun olduğu iddia edilmez. Sonuçlar yapay bozulmalı test sayfalarındandır (gerçek taranmış belge testi yok). Ayrıntılar: [GitHub deposu](https://github.com/furkancmc/finance-vl-fine-tune).

## Ne yapar?
Bir sayfa **görüntüsü** ve iki istemden biriyle çalışır:
1. *Transkripsiyon:* `Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz.` → sayfa metni
2. *Tablo:* `Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için kalem, dipnot ve degerler alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir).` → `{"kalem", "dipnot", "degerler"}` satırları (JSON)

## Sonuçlar

Test sayfaları eğitimde hiç görülmemiş **şirketlerden**dir (şirket bazlı bölme). Aynı sayfalar dört bozulma seviyesinde ölçüldü (A bozulmamış → D ağır; yapay bozulma, bkz. GitHub `docs/METHODOLOGY.md`). Köşeli parantez: %95 bootstrap güven aralığı.
n = 567 transkripsiyon, 133 tablo sayfası / seviye.

**Transkripsiyon, medyan CER** (karakter hata oranı, düşük iyi)

| Seviye | Taban model | + LoRA |
|---|---|---|
| A (temiz) | 0.421 [0.403, 0.440] | **0.147 [0.098, 0.178]** |
| B (hafif) | 0.402 [0.383, 0.424] | **0.145 [0.111, 0.176]** |
| C (orta) | 0.398 [0.384, 0.420] | **0.141 [0.116, 0.185]** |
| D (ağır) | 0.612 [0.586, 0.651] | **0.394 [0.359, 0.426]** |

**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı

| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |
|---|---|---|---|---|
| A (temiz) | 0.005 [0.000, 0.020] | **0.479 [0.428, 0.532]** | 0.01 [0.00, 0.04] | 0.96 [0.92, 0.99] |
| B (hafif) | 0.006 [0.000, 0.023] | **0.460 [0.409, 0.517]** | 0.01 [0.00, 0.03] | 0.95 [0.91, 0.98] |
| C (orta) | 0.003 [0.000, 0.013] | **0.393 [0.342, 0.449]** | 0.01 [0.00, 0.04] | 0.94 [0.90, 0.98] |
| D (ağır) | 0.003 [0.000, 0.013] | **0.130 [0.095, 0.172]** | 0.05 [0.01, 0.08] | 0.84 [0.77, 0.90] |

**Nasıl okunmalı:**
- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı "daha iyi okuma" değildir.
- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.
- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.
- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.

## Eğitim

| | |
|---|---|
| Taban model | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) |
| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |
| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |
| Süre / adım | 2 epoch, 158 adım, etkin batch 32 (2 × 16), lr 2e-4 cosine, A100 80 GB'ta 128 dk |
| Son kayıp | eğitim 0.3, doğrulama 0.3494 (modeller arası karşılaştırılamaz: tokenizer farklı) |
| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |

## Kullanım

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "google/gemma-3-12b-it"
ADAPTER = "furkancmc/Gemma-3-12B-KAP-TR-LoRA"
model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()
processor = AutoProcessor.from_pretrained(BASE)

PROMPT = "Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz."
image = Image.open("sayfa.png").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

Eğitimde görüntüler en çok 700 000 piksel olarak verildi; benzer çözünürlükte (≈ 130 DPI A4) sayfalarla en iyi sonuç beklenir.

## Sınırlamalar
- Yalnızca KAP bildirim sayfalarında (Ocak–Eylül 2026) eğitildi/ölçüldü; başka belge türlerine genellemesi ölçülmedi.
- Ağır bozulmada (D) hata hâlâ yüksektir; bazı sayfalarda çıktı tekrar döngüsüne girip token sınırına kadar uzayabilir.
- Gerçek tarama/fotoğraf testi yoktur; bozulmalar yapaydır.
- Tablo satır sayısını birebir yakalama zayıftır (bkz. GitHub sonuçları).

## Lisans
Bu adaptör **Gemma** modelinden türetilmiştir ve [Gemma Kullanım Şartları](https://ai.google.dev/gemma/terms) ile Gemma Yasaklı Kullanım Politikası'na tabidir. Dağıtırken bu şartları iletmeniz gerekir.

## Atıf
```
@misc{kap_tr_vl_lora_2026,
  title  = {KAP Turkish Document VL LoRA Adapters},
  author = {furkancmc},
  year   = {2026},
  url    = {https://github.com/furkancmc/finance-vl-fine-tune}
}
```
