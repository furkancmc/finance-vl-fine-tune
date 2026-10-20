---
language: tr
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-VL-7B-Instruct
pipeline_tag: image-text-to-text
tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]
---

# Qwen2.5-VL-7B-KAP-TR-LoRA

**Qwen2.5-VL-7B-Instruct** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).

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
| A (temiz) | 0.105 [0.097, 0.115] | **0.009 [0.007, 0.011]** |
| B (hafif) | 0.115 [0.103, 0.128] | **0.010 [0.009, 0.012]** |
| C (orta) | 0.121 [0.112, 0.138] | **0.013 [0.011, 0.017]** |
| D (ağır) | 0.401 [0.347, 0.450] | **0.165 [0.131, 0.193]** |

**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı

| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |
|---|---|---|---|---|
| A (temiz) | 0.193 [0.138, 0.257] | **0.690 [0.631, 0.745]** | 0.56 [0.48, 0.64] | 0.92 [0.86, 0.96] |
| B (hafif) | 0.141 [0.092, 0.200] | **0.689 [0.630, 0.744]** | 0.48 [0.40, 0.56] | 0.93 [0.89, 0.97] |
| C (orta) | 0.155 [0.104, 0.215] | **0.636 [0.577, 0.695]** | 0.52 [0.44, 0.60] | 0.92 [0.88, 0.97] |
| D (ağır) | 0.059 [0.034, 0.093] | **0.287 [0.234, 0.345]** | 0.53 [0.44, 0.61] | 0.76 [0.68, 0.83] |

**Nasıl okunmalı:**
- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı "daha iyi okuma" değildir.
- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.
- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.
- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.

## Eğitim

| | |
|---|---|
| Taban model | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |
| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |
| Süre / adım | 2 epoch, 158 adım, etkin batch 32 (4 × 8), lr 2e-4 cosine, A100 80 GB'ta 152 dk |
| Son kayıp | eğitim 0.092, doğrulama 0.0201 (modeller arası karşılaştırılamaz: tokenizer farklı) |
| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |

## Kullanım

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER = "furkancmc/Qwen2.5-VL-7B-KAP-TR-LoRA"
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
