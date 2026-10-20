# Proje durumu: tamamlandı

Tamamlanma: 2026-10-20. Ana sayfa: [`README.md`](README.md) · yöntem: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) · sonuçlar: [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

## Tamamlananlar
- **v2 veri seti:** şirket bazlı 261/50/80 bölme; eğitim 2500 (2000 metin + 500 tablo), doğrulama 500, test 700 × 4 bozulma seviyesi. Damga-only sayfalar filtrelendi.
- **5 model LoRA ile eğitildi ve değerlendirildi** (Colab A100, `colab/run_all.sh`): Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3.5-8B, Gemma 3 12B, Llama 3.2 11B Vision.
  Llama 3.2 yalnızca ilk 100 örnek/seviye ve HF motoruyla ölçüldü (vLLM bu modeli desteklemiyor).
- Bootstrap %95 güven aralıklı birleşik rapor, gerçek çıktı örnekleri, 5 model kartı ve Hugging Face adaptörleri (`furkancmc/*-KAP-TR-LoRA`).

## Kapsam dışı kalanlar ve olası sonraki çalışmalar
1. **Gerçek taranmış test seti** (elle doğru metinli 50–100 sayfa). En önemli eksik; yapay bozulmanın gerçeği temsil edip etmediğini ölçer.
2. **Etiket doğrulama:** rastgele 30–40 sayfanın referansını (özellikle tablo satırları) elle kontrol etmek.
3. **Birden fazla eğitim tohumu** (en az 3) ile eğitim rastgeleliğini ölçmek; birbirine yakın modelleri (Qwen3 ↔ Qwen2.5) ayırmak.
4. **OCR baz çizgileri** (PaddleOCR-VL, HunyuanOCR, DeepSeek-OCR): `ocr_eval_vllm.py` hazır; PaddleOCR-VL vLLM'de çalıştı (kısmi ön ölçüm: A seviyesinde medyan CER ≈ 0,05), diğerleri koşulmadı.
5. Daha uzun / geniş eğitim, `repetition_penalty` denemesi (ağır bozulmada kaçak üretim).
6. Gemma 3'ün zayıflığının nedeni (sabit 896×896 çözünürlük şüphesi) test edilmedi.
7. Yayın öncesi: KAP verisi yeniden dağıtım koşullarını doğrulamak; adaptör depolarını (şu an **private**) public yapmadan önce model kartlarını gözden geçirmek.

## Çalışma kuralları
- Türkçe, kısa ve net; doğrulanan/doğrulanmayan açık söylenir.
- Maliyetli işler (uzun eğitim) için onay; token'lar repoya/sohbete yazılmaz (Colab Secrets / ortam değişkeni).
- Colab'da çok uzun sessiz hücreler (≥ 30 dk) MCP tarafından iptal edilir; arka planda çalıştırıp kısa aralıklarla kontrol et.
