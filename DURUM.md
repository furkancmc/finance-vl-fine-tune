# Durum ve devam notları

Son güncelleme: 2026-10-17 akşam. Yarın buradan devam edilecek. Önce bu dosyayı, sonra `results/RESULTS.md` oku.

## Özet

Türkçe KAP PDF sayfalarında VL modelin LoRA ile ne kadar öğrendiğini ölçüyoruz. İki model eğitildi ve dört bozulma seviyesinde (A temiz … D ağır) test edildi:
Qwen2.5-VL-7B ve Qwen3-VL-8B. **LoRA belirgin işe yaradı** (A-C'de medyan CER ~0,11 → ~0,015; tablo hücre doğruluğu 0,15-0,43 → ~0,70-0,77).
Ağır bozulma (D) ve tablo satır sayısı hâlâ zayıf. Ayrıntı ve tablolar: `results/RESULTS.md`.

## Bugün ne yapıldı (sırayla)

1. Colab MCP ile A100 80 GB'a bağlanıldı; Drive, `dataset.zip`, token'lar (`GH_TOKEN`, `HF_TOKEN`) hazırlandı.
2. Smoke test (Qwen2.5-VL-7B, 3 adım) geçti. Bulunan hata düzeltildi: transformers 5.x'te `image_processor.min_pixels/max_pixels` salt okunur (`train_lora.py`, `4bfd222`).
3. Tam eğitim (3 epoch, tam veri) **~5 saat** sürecekti → iptal edildi. Ayarlar kısıldı: 1000 örnek x 2 epoch, 0,7M piksel, batch 4 x accum 8 (64 adım, ~1 saat).
4. Qwen2.5-VL-7B eğitildi (61 dk), merge edildi; Qwen3-VL-8B smoke + tam eğitim (38 dk).
5. vLLM ayrı venv'de kuruldu; dört eval (2 model x lora/base) x test_A..D çalıştırıldı.
6. Sonuçlar analiz edildi, `results/` altına konuldu; CER>1 sayfalarının analizi düzeltildi (aşağıda).
7. `metrics.strip_repetition`, `report.py --clean --min-ref-chars`, `eval_vllm.py --repetition-penalty` eklendi.

## Sonuç tek bakışta (temizlenmiş görünüm, medyan CER, n=282)

| Seviye | Qwen2.5 baz → LoRA | Qwen3 baz → LoRA |
|---|---|---|
| A | 0,102 → 0,015 | 0,098 → 0,015 |
| B | 0,117 → 0,015 | 0,118 → 0,012 |
| C | 0,116 → 0,019 | 0,119 → 0,014 |
| D | 0,375 → 0,180 | 0,263 → 0,122 |

Tek koşu, tek seed, verinin yarısından azı kullanıldı: küçük farklar (özellikle A seviyesinde iki model arası) kesin sıralama değildir.

## Önemli bulgular / dersler

- **Ham CER yanıltıcı.** Ham raporda transkripsiyon CER 1,3-2,7 çıkıyor. Nedeni uç sayfalar (CER > 1):
  - ~15 test sayfası **etiket kusuru**: PDF metin katmanında sadece SPK "Doğrulama Kodu" damgası var (<150 karakter), model sayfadaki asıl içeriği okuyor. Test setinde 17 sayfa. Her seviyede ve modelde aynı sayfalar.
  - Gerçek **kaçak üretim** (3072 token sınırına çarpıp sonu tekrar çöpü): A-B'de ~%3, D'de çok daha fazla.
  - Rapor için `report.py --clean --min-ref-chars 150` kullan. `--repetition-penalty` GPU'da **denenmedi**.
- **Eval loss kalite göstermez** (0,02-0,03 çıktı; cevap token'ları üzerinden, kopyalama görevinde zaten düşük). Kaliteyi CER/WER gösterir.
- **Hız:** adım süresi ~57 sn (Qwen2.5) / ~35 sn (Qwen3). Piksel düşürme ve `--no-grad-ckpt`'in hızı beklediğim kadar artırdığı **doğrulanmadı** (bellek ve hız neredeyse aynı kaldı); yavaşlığın asıl nedeni bilinmiyor (hesaplama verimi düşük, FA2 yok).
- **Eğitilen veri alt kümesi:** `--limit 1000` train.jsonl'in ilk 1000 satırı (dosya karışık sıralı, temsilî).

## Colab'da yaşadığımız sorunlar ve çözümleri

- `git clone` 403 "Write access to repository not granted": önce Colab Secret adı `GF_TOKEN` yazılmıştı (`GH_TOKEN` olmalı); sonra token'ın bu repoya erişimi yoktu (403). Repo erişimi olan yeni token ile çözüldü.
- vLLM motoru açılışta çöktü (`FileNotFoundError: 'ninja'`): venv'i activate etmeden mutlak yolla çağırınca `ninja` PATH'te yoktu. Çözüm: `export PATH=/content/vllm_env/bin:$PATH` ve `VLLM_USE_FLASHINFER_SAMPLER=0` (`colab/eval_all.sh` içinde).
- `pkill -f "python train_lora"` süreci bulmadı çünkü süreç adı `python3`. Kalıp `train_[l]ora.py` kullan.
- Dönen rapor adımı "TAMAM" yazsa bile eval'ler başarısızsa rapor boştur; eval log'larındaki BASARISIZ satırlarına bak.

## Yarın nasıl devam edilir

Colab runtime kapanmış olabilir. Kapandıysa `/content` altındaki şeyler gider (repo, `/content/dataset`, `/content/vllm_env`, HF önbelleği). **Drive'da kalanlar**: `kap_lora/dataset.zip`, adaptörler, merged modeller, tahminler, raporlar.

1. Claude Code'u yeniden başlat / `/mcp` ile `colab-mcp` bağla (`claude mcp add colab-mcp -- uvx --python 3.13 git+https://github.com/googlecolab/colab-mcp`). Colab sekmesini aç, **A100** runtime'a bağla.
2. Colab Secrets: `GH_TOKEN` (repoyu okuyabilen; classic `repo` kapsamı veya fine-grained + bu repo + Contents: Read), `HF_TOKEN`. "Notebook access" açık olsun.
3. Hücrelerle: Drive mount → `pip install unsloth bitsandbytes peft accelerate qwen-vl-utils rapidfuzz` → `git clone` (`/content/repo`) → `dataset.zip`'i `/content` altına aç.
4. Eğitim: `train_lora.py` komutu `colab/train_chain.sh` içindeki `TRAIN_ARGS` ile. Merge: `merge_lora.py`.
5. Eval: `bash colab/setup_vllm_env.sh` sonra `colab/eval_all.sh`. (`eval_all.sh` çalışan komutların derlenmiş hâlidir; bu hâliyle bir kez daha çalıştırılmadı.)
6. Kayıt: Drive'daki `eval/report_clean/report.csv` → `results/` altına aktar, `RESULTS.md`'yi güncelle, commit + push.

Drive düzeni (`MyDrive/kap_lora/`): `dataset.zip`, `qwen25vl_7b/` ve `qwen3vl_8b/` (`final/`, `checkpoint-*`, `merged/`, `train_log.jsonl`, `train_stdout.log`),
`eval/<model>/preds_<tag>_test_<A-D>.jsonl`, `eval/report/` (ham), `eval/report_clean/` (temiz), `pipeline.log`, `eval_pipeline.log`.
`qwen25vl_7b_iptal_*` klasörleri iptal edilen 5 saatlik koşuların kalıntısı, silinebilir.

## Sıradaki işler (karar bekleyenler işaretli)

1. **[karar]** Daha uzun eğitim: tam train seti (3253 örnek) ve 3 epoch ~5 saat (Qwen2.5 hızıyla); ara seçenekler var (örn. 2000 örnek x 2 epoch ~2 saat). D seviyesi ve tablo satır sayısı için en olası kazanç.
2. **[karar]** Damga-only sayfalarını eğitimden de çıkarmak: `make_dataset.py`'ye minimum referans uzunluğu filtresi (şu an yok; veri seti yeniden üretilmeli ve yeniden eğitilmeli).
3. `--repetition-penalty` (1,03-1,1) küçük bir sette denenmeli; sayı/nokta tekrarını da cezalandırabilir.
4. `notebooks/` güncel değil: eğitim ayarları bugünkü koşu ile uyuşmuyor, eval notebook'u vLLM'i ana ortama kuruyor (venv gerekli). `build_notebooks.py` güncellenmeli.
5. **[karar, lisans onayı sizden]** Diğer modeller: Gemma 3 12B, InternVL3.5-8B, Llama 3.2 11B Vision (HF lisansı kabul edilmeli); `model_zoo.py`'de tanımlı ama adaptörleri yok.
6. Hazır OCR modelleri zero-shot baz çizgileri: PaddleOCR-VL, DeepSeek-OCR, HunyuanOCR (çıktı temizleme gerekir).

## Çalışma kuralları (kullanıcı tercihleri)

- Turkçe, kısa ve net; neyin doğrulandığı / doğrulanmadığı açık söylenmeli.
- Her adım commit + push; commit mesajına `Co-Authored-By` satırı.
- Maliyetli iş (tam eğitim, uzun eval) için onay; bugün sadece "eğitimi bitince Qwen3'ü direkt başlat" onayı verildi.
- Token'lar sohbete / repoya yazılmaz; sadece Colab Secrets ve `EVREN_API_KEY` ortam değişkeni.
