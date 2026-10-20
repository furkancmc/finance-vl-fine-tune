---
language: tr
license: apache-2.0
library_name: peft
base_model: OpenGVLab/InternVL3_5-8B-HF
pipeline_tag: image-text-to-text
tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]
---

# InternVL3.5-8B-KAP-TR-LoRA

**InternVL3.5-8B (HF)** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).

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
| A (temiz) | 0.245 [0.226, 0.262] | **0.015 [0.012, 0.017]** |
| B (hafif) | 0.246 [0.227, 0.265] | **0.016 [0.012, 0.021]** |
| C (orta) | 0.292 [0.262, 0.341] | **0.031 [0.027, 0.039]** |
| D (ağır) | 0.845 [0.826, 0.868] | **0.364 [0.308, 0.414]** |

**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı

| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |
|---|---|---|---|---|
| A (temiz) | 0.001 [0.000, 0.004] | **0.676 [0.626, 0.724]** | 0.03 [0.01, 0.06] | 0.97 [0.94, 0.99] |
| B (hafif) | 0.000 [0.000, 0.000] | **0.647 [0.596, 0.701]** | 0.01 [0.00, 0.04] | 0.95 [0.92, 0.98] |
| C (orta) | 0.000 [0.000, 0.000] | **0.576 [0.519, 0.632]** | 0.04 [0.01, 0.08] | 0.92 [0.88, 0.97] |
| D (ağır) | 0.001 [0.000, 0.003] | **0.170 [0.127, 0.219]** | 0.05 [0.01, 0.08] | 0.71 [0.63, 0.79] |

**Nasıl okunmalı:**
- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı "daha iyi okuma" değildir.
- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.
- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.
- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.

## Eğitim

| | |
|---|---|
| Taban model | [`OpenGVLab/InternVL3_5-8B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |
| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |
| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |
| Süre / adım | 2 epoch, 158 adım, etkin batch 32 (4 × 8), lr 2e-4 cosine, A100 80 GB'ta 151 dk |
| Son kayıp | eğitim 0.142, doğrulama 0.0509 (modeller arası karşılaştırılamaz: tokenizer farklı) |
| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |

## Kullanım

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "OpenGVLab/InternVL3_5-8B-HF"
ADAPTER = "furkancmc/InternVL3.5-8B-KAP-TR-LoRA"
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
