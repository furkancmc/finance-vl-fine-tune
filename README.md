# KAP Türkçe VL Fine-Tune

Türkçe finansal belgeler (KAP, Kamuyu Aydınlatma Platformu) üzerinde bir vision-language modelin
**Türkçe öğrenme kapasitesini LoRA ile ölçmek** için veri hattı. Model sayfa görüntüsünü okur; çıktısı,
PDF'in kendi metin katmanından gelen referansla karşılaştırılır. Eğitim Colab A100'de yapılacak.

> Durum: veri hattı, eğitim ve değerlendirme tamam (Qwen2.5-VL-7B ve Qwen3-VL-8B, dört seviyede test edildi).
> Sonuçlar: [`results/RESULTS.md`](results/RESULTS.md). Devam notları ve sıradaki işler: [`DURUM.md`](DURUM.md).

## Fikir

- Elimizdeki 500 bildirim **dijital PDF**, yani metin katmanı var. Bu katman **cevap anahtarı** olur: etiket maliyeti sıfır, hatasız.
- Modele metin katmanını göstermiyoruz, sadece **sayfa görüntüsünü** veriyoruz. Gerçek hedef taranmış/fotoğraflanmış belge.
- Bu yüzden görüntüler taranmış belge gibi **dört seviyede bozuluyor**:

| Seviye | Anlamı | Örnek bozulma |
|---|---|---|
| A | Bozulmamış | yok |
| B | Hafif | hafif yumuşaklık, JPEG |
| C | Orta | gürültü, orta bulanıklık, düşük çözünürlük |
| D | Ağır (zar zor okunur) | çözünürlük %40-55, güçlü blur, JPEG kalite 12-35 |

## Görevler

1. **Transkripsiyon (ana görev):** sayfa görüntüsü → sayfadaki tüm Türkçe metin. Metrikler: CER, WER, Türkçe karakter doğruluğu (ç ğ ı İ ö ş ü), sayı F1.
2. **Tablo çıkarımı:** finansal tablo sayfası → `{"kalem", "dipnot", "degerler"}` satırları (JSON). Referans `table_extract.py` ile PDF kelime koordinatlarından üretilir.

Alan çıkarımı ve soru-cevap kapsam dışı (etiket maliyeti yüksek).

## Hat

```
kap_collect.py     KAP'tan PDF indir (500 bildirim)         -> data/docs/
inventory.py       PDF denetimi (sayfa, metin katmanı)       -> data/inventory.csv
make_pages.py      sayfa PNG + referans metin + tablo        -> data/pages/, data/pages.jsonl
make_dataset.py    şirket bazında böl, bozulma uygula        -> data/dataset/*.jsonl, images/
metrics.py         CER/WER/tablo metrikleri (selftest var)
degrade.py         A-D seviyeli tarama bozulması
table_extract.py   PDF sayfasından finansal tablo çıkarımı
llm.py             OpenAI uyumlu API ile kod üretimi (EVREN_API_KEY)
page_stats.py      sayfa kapsama istatistiği
prompts/           kod üretimi için spec dosyaları
PLAN.md            ilk plan ve kararlar (bazı bölümler eskidi, güncel durum DURUM.md)
DURUM.md           yapılanlar, bulgular, yarın nasıl devam edilir, sıradaki işler
results/           test sonuçları (RESULTS.md + csv)
colab/             Colab betikleri (vLLM venv kurulumu, eval zinciri, eğitim zinciri)
```

## Kullanım

```bash
pip install requests pymupdf pillow numpy rapidfuzz

python kap_collect.py          # indirir; yarıda kalırsa kaldığı yerden devam eder
python inventory.py
python make_pages.py           # --limit N ile deneme
python make_dataset.py --zip   # data/dataset.zip (Colab'a yüklenecek)
python metrics.py --selftest
```

Konsolda Türkçe çıktı için `python -X utf8` kullanın (Windows).

## Colab (A100 80 GB)

Model anahtarları `model_zoo.py`'de. Hazır olanlar: `qwen25vl_7b`, `qwen3vl_8b` (diğerleri sonra).

```
train_lora.py       LoRA SFT: --backend unsloth|hf, bf16 (isteğe bağlı --load-in-4bit), 8-bit AdamW (bitsandbytes), --dry-run
merge_lora.py       adaptörü baz modele birleştirir (vLLM için tam model klasörü)
eval_vllm.py        vLLM ile hızlı toplu çıkarım (--engine hf yedek), test_A..D
report.py           baz vs LoRA karşılaştırma raporu (CER, tr_char_acc, tablo metrikleri)
build_notebooks.py  notebooks/train_<model>.ipynb ve eval_<model>.ipynb üretir
```

Akış: `train_<model>.ipynb` (smoke test → tam eğitim → merge) sonra eval. unsloth ile vLLM aynı ortama kurulmaz:
notebook'taki eval yolu yeni bir runtime ister; gerçek koşuda bunun yerine **aynı runtime'da ayrı venv** kullanıldı
(`colab/setup_vllm_env.sh`, sonra `colab/eval_all.sh`; venv'in `bin/` klasörü `PATH`'e eklenmeli). Notebook'lar henüz bu yola göre güncellenmedi.
Colab Secrets: `GH_TOKEN` (bu repoyu okuyabilen token, aksi hâlde clone 403 verir), `HF_TOKEN`. Drive'a `kap_lora/dataset.zip` yükle.

Gerçek koşuda kullanılan ayarlar (1 saatlik hedef için kısılmış): 1000 örnek, 2 epoch, batch 4 x grad-accum 8, `--max-pixels 700000`,
`--no-grad-ckpt`. Tam veri / 3 epoch Qwen2.5-VL-7B'de ~5 saat sürer. Komutlar: `colab/train_chain.sh`.

Yerel doğrulama (GPU olmadan): `python train_lora.py --model qwen25vl_7b --data data/dataset --dry-run`.

## Veri seti

- Bölme **şirket bazında** (%80/10/10), aynı şirket iki tarafta yer almaz.
- `train`: dört seviye eşit ağırlıkla karışık. `val`: seviye A.
- `test_A`, `test_B`, `test_C`, `test_D`: aynı test sayfaları, dört seviyede. Bozulmaya dayanıklılık seviye seviye ölçülür.
- Her kayıt: `id, task, split, level, image, messages (user: görüntü + prompt, assistant: cevap), ref`.
- Seed'ler sabit (42), her şey tekrar üretilebilir.

Mevcut boyutlar (391 şirket: 312 train / 39 val / 40 test; 3.169 sayfa, 825 tablo sayfası):

| Set | Örnek | Transkripsiyon | Tablo |
|---|---|---|---|
| train | 3.253 (A 784, B 814, C 813, D 842) | 2.565 | 688 |
| val (A) | 370 | 305 | 65 |
| test_A / B / C / D | her biri 371 | 299 | 72 |

`data/dataset.zip` ≈ 840 MB (Colab'a Drive üzerinden yüklenecek).

## Sonuçlar (özet)

LoRA her iki modelde de büyük fark yarattı. Temizlenmiş görünümde (sondaki tekrar bloğu kesilir, referansı çok kısa 17 damga-only sayfa hariç; n=282)
transkripsiyon medyan CER, baz → LoRA:

| Seviye | Qwen2.5-VL-7B | Qwen3-VL-8B |
|---|---|---|
| A temiz | 0,102 → 0,015 | 0,098 → 0,015 |
| B hafif | 0,117 → 0,015 | 0,118 → 0,012 |
| C orta | 0,116 → 0,019 | 0,119 → 0,014 |
| D ağır | 0,375 → 0,180 | 0,263 → 0,122 |

Tablo çıkarımında hücre doğruluğu A'da 0,21 → 0,77 (Qwen2.5) ve 0,43 → 0,72 (Qwen3). Ağır bozulma (D) ve tablo satır sayısı hâlâ zayıf.
Tek koşu, kısılmış eğitim; ayrıntılar ve uyarılar `results/RESULTS.md` içinde.

## Notlar

- Ham CER/WER yanıltıcı olabilir: referans PDF'in metin katmanı bazı sayfalarda sadece SPK "Doğrulama Kodu" damgasıdır (test setinde 17 sayfa),
  ve bazı çıktılar 3072 token sınırına kadar tekrar eder. Bu yüzden rapor için `python report.py --eval-dir ... --clean --min-ref-chars 150`.

- `data/` (PDF'ler, görüntüler, veri seti) git'e girmez; yalnızca kod ve spec'ler.
- API anahtarı sadece `EVREN_API_KEY` ortam değişkeninde, kodda yok.
- Denetim: 985 PDF, 16.567 sayfa; sayfaların %4,1'inde metin katmanı yok (transkripsiyondan elenir).
