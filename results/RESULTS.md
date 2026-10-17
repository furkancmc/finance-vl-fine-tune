# Sonuçlar (test seti, 4 bozulma seviyesi)

Tarih: 2026-10-17. Tüm sayılar tek koşu (seed 42). Ham veri: `transcribe.csv`, `table.csv`.

## Kurulum

| | Qwen2.5-VL-7B | Qwen3-VL-8B |
|---|---|---|
| HF modeli | `Qwen/Qwen2.5-VL-7B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` |
| Yöntem | 16-bit LoRA (unsloth, bf16), r=32, alpha=64 | aynı |
| Eğitim verisi | train'den ilk 1000 örnek (karışık seviye, karışık görev) | aynı |
| Epoch / batch | 2 epoch, batch 4 x grad-accum 8 (etkin 32), 64 adım, lr 2e-4 | aynı |
| Görüntü | max 700k piksel, gradient checkpointing kapalı | aynı |
| Eğitim süresi | 61 dk (A100 80 GB) | 38 dk |
| Son eval loss / train loss | 0.0254 / 0.137 | 0.0204 / 0.093 |

Eval: vLLM 0.31 (ayrı venv), `temperature=0` (greedy), `repetition_penalty=1.0`, `max_new_tokens=3072`, görüntü max 700k piksel (eğitimle aynı).
Birleştirilmiş (merge) LoRA modeli vs. baz model, aynı sayfalar. Transkripsiyon n=299, tablo n=72 (her seviyede).

**Bu eğitimde veri ve epoch bilinçli olarak kısıldı (~1 saat hedefi); tam veri/3 epoch ile sonuçlar değişebilir.**

## Transkripsiyon: medyan CER (düşük iyi)

Ortalama CER yerine medyan kullanılıyor: sayfaların ~%7-8'inde model tekrar döngüsüne giriyor (aşağıya bakın), bu birkaç uç değer ortalamayı şişiriyor.

| Seviye | Qwen2.5 baz | Qwen2.5 LoRA | Qwen3 baz | Qwen3 LoRA |
|---|---|---|---|---|
| A | 0.110 | **0.015** | 0.108 | **0.018** |
| B | 0.129 | **0.017** | 0.126 | **0.013** |
| C | 0.124 | **0.020** | 0.128 | **0.016** |
| D | 0.413 | **0.217** | 0.335 | **0.142** |

## Transkripsiyon: medyan WER (düşük iyi)

| Seviye | Qwen2.5 baz | Qwen2.5 LoRA | Qwen3 baz | Qwen3 LoRA |
|---|---|---|---|---|
| A | 0.193 | 0.054 | 0.201 | 0.052 |
| B | 0.214 | 0.054 | 0.229 | 0.046 |
| C | 0.256 | 0.068 | 0.231 | 0.050 |
| D | 0.742 | 0.385 | 0.608 | 0.293 |

## Transkripsiyon: Türkçe karakter doğruluğu (ç ğ ı İ ö ş ü, yüksek iyi) ve sayı F1

| Seviye | Qwen2.5 baz | Qwen2.5 LoRA | Qwen3 baz | Qwen3 LoRA |
|---|---|---|---|---|
| A | 0.855 / 0.873 | 0.937 / 0.911 | 0.871 / 0.901 | 0.942 / 0.922 |
| B | 0.850 / 0.870 | 0.927 / 0.907 | 0.869 / 0.913 | 0.947 / 0.932 |
| C | 0.835 / 0.851 | 0.915 / 0.885 | 0.854 / 0.888 | 0.922 / 0.901 |
| D | 0.562 / 0.522 | 0.692 / 0.595 | 0.636 / 0.547 | 0.761 / 0.636 |

(her hücre: Türkçe karakter doğruluğu / sayı F1)

## Tekrar döngüsü: sayfa sayısı (CER > 1, n=299)

| Seviye | Qwen2.5 baz | Qwen2.5 LoRA | Qwen3 baz | Qwen3 LoRA |
|---|---|---|---|---|
| A | 21 | 22 | 23 | 23 |
| B | 23 | 22 | 26 | 21 |
| C | 23 | 26 | 27 | 21 |
| D | 91 | 71 | 105 | 58 |

CER > 1 olan sayfalarda çıktı referanstan uzun; örneklerde `.1.1.1.1…` gibi sonsuz tekrar görüldü. A-C seviyelerinde sayı baz ve LoRA'da benzer (~21-27), yani fine-tune bunu yaratmadı ama düzeltmedi.

## Tablo çıkarımı (n=72)

Hücre doğruluğu (`cell_exact`, yüksek iyi) / geçerli JSON oranı (`valid`):

| Seviye | Qwen2.5 baz | Qwen2.5 LoRA | Qwen3 baz | Qwen3 LoRA |
|---|---|---|---|---|
| A | 0.21 / 0.60 | 0.77 / 0.97 | 0.43 / 0.96 | 0.72 / 0.97 |
| B | 0.15 / 0.56 | 0.76 / 0.97 | 0.31 / 0.90 | 0.70 / 0.94 |
| C | 0.15 / 0.54 | 0.70 / 0.93 | 0.34 / 0.89 | 0.68 / 0.92 |
| D | 0.06 / 0.50 | 0.31 / 0.78 | 0.13 / 0.69 | 0.36 / 0.81 |

Satır sayısı eşleşmesi LoRA ile de düşük (0.25-0.39); diğer sütunlar `table.csv` içinde.

## Okuma

- LoRA, A-C seviyelerinde medyan CER'i yaklaşık 7-10 kat düşürdü (~0.11-0.13 → ~0.013-0.020); tabloda hücre doğruluğu 0.15-0.43 → ~0.70-0.77.
- D seviyesi (ağır bozulma) hâlâ zor: LoRA ile medyan CER 0.14-0.22. Qwen3-VL-8B burada belirgin daha iyi (0.142 vs 0.217).
- A seviyesinde iki model eşit düzeyde; farklar küçük, tek koşu olduğu için kesin sıralama iddia edilmiyor.
