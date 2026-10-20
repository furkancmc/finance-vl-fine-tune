---
language: tr
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-VL-8B-Instruct
pipeline_tag: image-text-to-text
tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]
---

# Qwen3-VL-8B-KAP-TR-LoRA

**Qwen3-VL-8B-Instruct** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).

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
| A (temiz) | 0.107 [0.091, 0.124] | **0.009 [0.007, 0.011]** |
| B (hafif) | 0.119 [0.094, 0.131] | **0.007 [0.006, 0.009]** |
| C (orta) | 0.116 [0.102, 0.128] | **0.009 [0.007, 0.012]** |
| D (ağır) | 0.289 [0.249, 0.352] | **0.092 [0.072, 0.111]** |

**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı

| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |
|---|---|---|---|---|
| A (temiz) | 0.452 [0.384, 0.512] | **0.715 [0.662, 0.770]** | 0.95 [0.92, 0.98] | 0.96 [0.92, 0.99] |
| B (hafif) | 0.333 [0.267, 0.400] | **0.695 [0.636, 0.755]** | 0.90 [0.85, 0.95] | 0.92 [0.88, 0.97] |
| C (orta) | 0.314 [0.250, 0.381] | **0.656 [0.595, 0.711]** | 0.92 [0.88, 0.97] | 0.92 [0.88, 0.96] |
| D (ağır) | 0.124 [0.080, 0.166] | **0.341 [0.278, 0.407]** | 0.68 [0.59, 0.75] | 0.80 [0.74, 0.87] |

**Nasıl okunmalı:**
- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı "daha iyi okuma" değildir.
- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.
- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.
- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.

## Eğitim

| | |
|---|---|
| Taban model | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |
| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |
| Süre / adım | 2 epoch, 158 adım, etkin batch 32 (4 × 8), lr 2e-4 cosine, A100 80 GB'ta 90 dk |
| Son kayıp | eğitim 0.054, doğrulama 0.0163 (modeller arası karşılaştırılamaz: tokenizer farklı) |
| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |

## Kullanım

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER = "furkancmc/Qwen3-VL-8B-KAP-TR-LoRA"
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
Taban model Apache-2.0 lisanslıdır; kullanmadan önce taban modelin sayfasındaki güncel lisansı doğrulayın.

## Atıf
```
@misc{kap_tr_vl_lora_2026,
  title  = {KAP Turkish Document VL LoRA Adapters},
  author = {furkancmc},
  year   = {2026},
  url    = {https://github.com/furkancmc/finance-vl-fine-tune}
}
```
