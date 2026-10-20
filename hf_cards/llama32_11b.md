---
language: tr
license: llama3.2
library_name: peft
base_model: meta-llama/Llama-3.2-11B-Vision-Instruct
pipeline_tag: image-text-to-text
tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]
---

# Llama-3.2-11B-Vision-KAP-TR-LoRA

**Llama 3.2 11B Vision-Instruct** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).

> Bu bir araştırma/öğrenme projesidir; üretim kullanımı için uygun olduğu iddia edilmez. Sonuçlar yapay bozulmalı test sayfalarındandır (gerçek taranmış belge testi yok). Ayrıntılar: [GitHub deposu](https://github.com/furkancmc/finance-vl-fine-tune).

## Ne yapar?
Bir sayfa **görüntüsü** ve iki istemden biriyle çalışır:
1. *Transkripsiyon:* `Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz.` → sayfa metni
2. *Tablo:* `Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için kalem, dipnot ve degerler alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir).` → `{"kalem", "dipnot", "degerler"}` satırları (JSON)

## Sonuçlar

Test sayfaları eğitimde hiç görülmemiş **şirketlerden**dir (şirket bazlı bölme). Aynı sayfalar dört bozulma seviyesinde ölçüldü (A bozulmamış → D ağır; yapay bozulma, bkz. GitHub `docs/METHODOLOGY.md`). Köşeli parantez: %95 bootstrap güven aralığı.
n = 67 transkripsiyon, 33 tablo sayfası / seviye (**yalnızca ilk 100 test örneği**, HF motoruyla; diğer modellerle kıyas için GitHub'daki ortak alt küme tablosuna bakın).

**Transkripsiyon, medyan CER** (karakter hata oranı, düşük iyi)

| Seviye | Taban model | + LoRA |
|---|---|---|
| A (temiz) | 0.449 [0.379, 0.557] | **0.032 [0.018, 0.052]** |
| B (hafif) | 0.417 [0.365, 0.602] | **0.033 [0.018, 0.052]** |
| C (orta) | 0.520 [0.452, 0.743] | **0.068 [0.030, 0.104]** |
| D (ağır) | 0.885 [0.848, 0.917] | **0.352 [0.244, 0.456]** |

**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı

| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |
|---|---|---|---|---|
| A (temiz) | 0.000 [0.000, 0.000] | **0.721 [0.605, 0.819]** | 0.00 [0.00, 0.00] | 0.97 [0.91, 1.00] |
| B (hafif) | 0.000 [0.000, 0.000] | **0.610 [0.484, 0.740]** | 0.00 [0.00, 0.00] | 0.85 [0.73, 0.97] |
| C (orta) | 0.000 [0.000, 0.000] | **0.482 [0.350, 0.624]** | 0.00 [0.00, 0.00] | 0.70 [0.55, 0.85] |
| D (ağır) | 0.000 [0.000, 0.000] | **0.082 [0.026, 0.156]** | 0.00 [0.00, 0.00] | 0.52 [0.36, 0.70] |

**Nasıl okunmalı:**
- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı "daha iyi okuma" değildir.
- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.
- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.
- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.

## Eğitim

| | |
|---|---|
| Taban model | [`meta-llama/Llama-3.2-11B-Vision-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |
| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |
| Süre / adım | 2 epoch, 158 adım, etkin batch 32 (2 × 16), lr 2e-4 cosine, A100 80 GB'ta 134 dk |
| Son kayıp | eğitim 0.145, doğrulama 0.0581 (modeller arası karşılaştırılamaz: tokenizer farklı) |
| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |

## Kullanım

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER = "furkancmc/Llama-3.2-11B-Vision-KAP-TR-LoRA"
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
**Built with Llama.** Bu adaptör Llama 3.2 modelinden türetilmiştir ve [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) ile Kabul Edilebilir Kullanım Politikası'na tabidir. Lisans, görüntü destekli Llama 3.2 modellerinin AB'de yerleşik kişi/şirketlerce kullanımını kısıtlar; kullanmadan önce lisansı okuyun.

## Atıf
```
@misc{kap_tr_vl_lora_2026,
  title  = {KAP Turkish Document VL LoRA Adapters},
  author = {furkancmc},
  year   = {2026},
  url    = {https://github.com/furkancmc/finance-vl-fine-tune}
}
```
