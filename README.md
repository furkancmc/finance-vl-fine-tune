# KAP Türkçe VL Fine-Tune

Türkçe finansal belgelerde (KAP bildirimleri) bir görüntü-dil modelinin (VL) LoRA ile ne kadar iyileştiğini ölçen deneysel bir proje.
Model bir PDF sayfasının **görüntüsünü** okur, çıktısı PDF'in kendi metin katmanından gelen referansla karşılaştırılır.

![durum](https://img.shields.io/badge/durum-devam%20ediyor%20(WIP)-orange)
![asama](https://img.shields.io/badge/a%C5%9Fama-ilk%20sonu%C3%A7lar-yellow)

> ## 🚧 Devam eden çalışma (work in progress)
> Bu repo bitmiş bir proje değil; çalışma sürerken açık tutuluyor. Kod, sonuçlar ve bu README sık değişebilir.
>
> | Aşama | Durum |
> |---|---|
> | Veri hattı (indirme, sayfa üretimi, bozulma, bölme) | ✅ tamam |
> | LoRA eğitimi: Qwen2.5-VL-7B, Qwen3-VL-8B | ✅ ilk koşu (kısaltılmış) |
> | Dört seviyede değerlendirme (baz vs LoRA) | ✅ ilk sonuçlar |
> | Tam veri / 3 epoch eğitim | ⏳ yapılmadı |
> | Ek modeller ve OCR baz çizgileri | ⏳ planlı |
> | Notebook'ların güncellenmesi | ⏳ planlı |
>
> Eğitim bilerek kısa tutuldu (verinin yaklaşık üçte biri, 2 epoch, tek koşu). Aşağıdaki sayılar **ilk sonuçlardır, kesin değildir.**
> Ayrıntılı sonuçlar: [`results/RESULTS.md`](results/RESULTS.md). Yapılacaklar: [`DURUM.md`](DURUM.md).

## Sonuç özeti: LoRA ne kazandırdı?

Aynı test sayfaları, aynı ayarlar; tek fark LoRA adaptörü. Her hücre **baz model → LoRA** şeklindedir.
Transkripsiyon sayıları "temizlenmiş" görünümden (n=282, açıklaması aşağıda). Tablo sayıları n=72.

### Transkripsiyon: medyan CER (düşük iyi)

| Model | Seviye | Baz | LoRA | Azalma | Kat |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | A temiz | 0,102 | **0,015** | %85 | 6,8x |
| | B hafif | 0,117 | **0,015** | %87 | 7,8x |
| | C orta | 0,116 | **0,019** | %84 | 6,1x |
| | D ağır | 0,375 | **0,180** | %52 | 2,1x |
| Qwen3-VL-8B | A temiz | 0,098 | **0,015** | %85 | 6,5x |
| | B hafif | 0,118 | **0,012** | %90 | 9,8x |
| | C orta | 0,119 | **0,014** | %88 | 8,5x |
| | D ağır | 0,263 | **0,122** | %54 | 2,2x |

Temiz ve hafif/orta bozulmada hata yaklaşık **6-10 kat** düşüyor. Ağır bozulmada (D) kazanç yarıya iniyor, kalan hata hâlâ büyük.

### Transkripsiyon: ortalama CER ve medyan WER

Ortalama CER az sayıdaki kötü sayfadan etkilenir, bu yüzden kazanç medyandan daha küçük görünür.

| Model | Seviye | Ort. CER baz → LoRA | Azalma | Medyan WER baz → LoRA | Azalma |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | A | 0,184 → **0,062** | %66 | 0,193 → **0,054** | %72 |
| | B | 0,195 → **0,066** | %66 | 0,214 → **0,054** | %75 |
| | C | 0,212 → **0,079** | %63 | 0,256 → **0,068** | %73 |
| | D | 0,531 → **0,360** | %32 | 0,742 → **0,385** | %48 |
| Qwen3-VL-8B | A | 0,161 → **0,055** | %66 | 0,201 → **0,052** | %74 |
| | B | 0,165 → **0,050** | %70 | 0,229 → **0,046** | %80 |
| | C | 0,182 → **0,059** | %68 | 0,231 → **0,050** | %78 |
| | D | 0,450 → **0,277** | %38 | 0,608 → **0,293** | %52 |

### Türkçe karakter doğruluğu ve sayı F1 (yüksek iyi)

Türkçe karakterler: ç ğ ı İ ö ş ü. Ham çıktı üzerinden, n=299.

| Model | Seviye | Türkçe karakter baz → LoRA | Fark | Sayı F1 baz → LoRA | Fark |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B | A | 0,855 → **0,937** | +0,082 | 0,873 → **0,911** | +0,038 |
| | B | 0,850 → **0,927** | +0,077 | 0,870 → **0,907** | +0,037 |
| | C | 0,835 → **0,915** | +0,080 | 0,851 → **0,885** | +0,034 |
| | D | 0,562 → **0,692** | +0,130 | 0,522 → **0,595** | +0,073 |
| Qwen3-VL-8B | A | 0,871 → **0,942** | +0,071 | 0,901 → **0,922** | +0,021 |
| | B | 0,869 → **0,947** | +0,078 | 0,913 → **0,932** | +0,019 |
| | C | 0,854 → **0,922** | +0,068 | 0,888 → **0,901** | +0,013 |
| | D | 0,636 → **0,761** | +0,125 | 0,547 → **0,636** | +0,089 |

Türkçe karakterlerde kazanç tutarlı (A-C'de yaklaşık +7-8 puan, D'de +13 puan). Sayılarda kazanç küçük (+1-4 puan, D'de +7-9): baz modeller sayıları zaten nispeten iyi okuyor.

### Tablo çıkarımı (hücre doğruluğu ve geçerli JSON oranı)

Görev: finansal tablo sayfası → `{"kalem", "dipnot", "degerler"}` satırları (JSON).

| Model | Seviye | Hücre doğruluğu baz → LoRA | Kat | Geçerli JSON baz → LoRA |
|---|---|---|---|---|
| Qwen2.5-VL-7B | A | 0,21 → **0,77** | 3,7x | 0,60 → **0,97** |
| | B | 0,15 → **0,76** | 5,1x | 0,56 → **0,97** |
| | C | 0,15 → **0,70** | 4,7x | 0,54 → **0,93** |
| | D | 0,06 → **0,31** | 5,2x | 0,50 → **0,78** |
| Qwen3-VL-8B | A | 0,43 → **0,72** | 1,7x | 0,96 → **0,97** |
| | B | 0,31 → **0,70** | 2,3x | 0,90 → **0,94** |
| | C | 0,34 → **0,68** | 2,0x | 0,89 → **0,92** |
| | D | 0,13 → **0,36** | 2,8x | 0,69 → **0,81** |

### Bu sonuçlardan çıkanlar

- **En büyük kazanç temiz ve hafif bozulmalı sayfalarda.** Medyan CER ~0,11 civarından ~0,015'e iniyor; yani sayfadaki karakterlerin yaklaşık %98,5'i doğru okunuyor.
- **Ağır bozulmada (D) kazanç var ama yetersiz.** Hata yarıya iniyor, yine de medyan CER 0,12-0,18. Qwen3-VL-8B burada Qwen2.5-VL-7B'den belirgin daha iyi (0,122 vs 0,180).
- **Tabloda baz model zayıf, LoRA farkı büyük.** Qwen2.5-VL-7B baz hâlinde geçerli JSON üretemiyor (A'da %60); bu yüzden onun tablo kazancının bir kısmı "çıktı formatını öğrenme" etkisi, Türkçe okuma yeteneğinden ayrı. Qwen3-VL-8B baz hâlinde formata zaten uyuyor (%96), bu yüzden onun kazancı (0,43 → 0,72) formatı ayıklanmış daha temiz bir ölçüm.
- **Tablo satır sayısı eşleşmesi LoRA ile de düşük** (0,25-0,39). Hücre doğruluğu iyileşti ama satırların tamamını yakalamak hâlâ çözülmedi.
- **A seviyesinde iki modelin farkı küçük.** Tek koşu olduğu için bu fark sıralama iddiası için yeterli değil.

### Sayılar nasıl okunmalı

- Ham CER/WER yanıltıcı. Referans PDF'in metin katmanı bazı sayfalarda sadece SPK "Doğrulama Kodu" damgasından ibaret (test setinde 17 sayfa); model sayfadaki asıl içeriği okuyunca CER 1'in üstüne çıkıyor. Ayrıca bazı çıktılar 3072 token sınırına kadar tekrar edip çöpe dönüyor.
- Bu yüzden transkripsiyon tablolarında **temizlenmiş görünüm** kullanıldı: çıktının sonundaki tekrar bloğu kesilir, damga-only 17 sayfa dışlanır (n=299 → 282). Baz ve LoRA aynı temizlemeden geçer. Ham sayılar `results/RESULTS.md` içinde duruyor.
- Komut: `python report.py --eval-dir ... --clean --min-ref-chars 150`
- Tek koşu, tek seed. Güven aralığı hesaplanmadı.

## Fikir

- 500 KAP bildirimi **dijital PDF**, yani metin katmanı var. Bu katman cevap anahtarı olarak kullanılıyor: elle etiketleme yok.
- Modele metin katmanı gösterilmez, yalnızca **sayfa görüntüsü** verilir. Asıl hedef taranmış veya fotoğraflanmış belge olduğu için görüntüler dört seviyede bozulur:

| Seviye | Anlamı | Örnek bozulma |
|---|---|---|
| A | Bozulmamış | yok |
| B | Hafif | hafif yumuşaklık, JPEG |
| C | Orta | gürültü, orta bulanıklık, düşük çözünürlük |
| D | Ağır (zor okunur) | çözünürlük %40-55, güçlü blur, JPEG kalite 12-35 |

## Görevler

1. **Transkripsiyon (ana görev):** sayfa görüntüsü → sayfadaki tüm Türkçe metin. Metrikler: CER, WER, Türkçe karakter doğruluğu, sayı F1.
2. **Tablo çıkarımı:** finansal tablo sayfası → `{"kalem", "dipnot", "degerler"}` satırları (JSON). Referans, `table_extract.py` ile PDF kelime koordinatlarından üretilir.

Alan çıkarımı ve soru-cevap kapsam dışı (etiket maliyeti yüksek).

## Veri hattı

```
kap_collect.py     KAP'tan PDF indirir (500 bildirim)         -> data/docs/
inventory.py       PDF denetimi (sayfa, metin katmanı)         -> data/inventory.csv
make_pages.py      sayfa PNG + referans metin + tablo          -> data/pages/, data/pages.jsonl
make_dataset.py    şirket bazında böler, bozulma uygular       -> data/dataset/*.jsonl, images/
metrics.py         CER/WER/tablo metrikleri (selftest var)
degrade.py         A-D seviyeli tarama bozulması
table_extract.py   PDF sayfasından finansal tablo çıkarımı
page_stats.py      sayfa kapsama istatistiği
llm.py             OpenAI uyumlu API ile kod üretimi (EVREN_API_KEY)
prompts/           kod üretimi için spec dosyaları
colab/             Colab betikleri (vLLM venv kurulumu, eval zinciri, eğitim zinciri)
results/           test sonuçları (RESULTS.md + csv)
examples/          örnek sayfa ve tablo çıktısı
PLAN.md            ilk plan (bazı bölümler eskidi)
DURUM.md           yapılanlar, bulgular, sıradaki işler
```

Kurulum ve çalıştırma:

```bash
pip install requests pymupdf pillow numpy rapidfuzz

python kap_collect.py          # indirir; yarıda kalırsa kaldığı yerden devam eder
python inventory.py
python make_pages.py           # --limit N ile deneme
python make_dataset.py --zip   # data/dataset.zip (Colab'a yüklenir)
python metrics.py --selftest
```

Windows konsolunda Türkçe çıktı için `python -X utf8` kullanın.

## Veri seti

- Bölme **şirket bazında** (%80/10/10): aynı şirket iki tarafta yer almaz.
- `train`: dört seviye eşit ağırlıkla karışık. `val`: seviye A.
- `test_A`, `test_B`, `test_C`, `test_D`: aynı test sayfaları, dört seviyede. Dayanıklılık seviye seviye ölçülür.
- Her kayıt: `id, task, split, level, image, messages, ref`.
- Seed sabit (42), her şey yeniden üretilebilir.

391 şirket (312 train / 39 val / 40 test), 3.169 sayfa, 825 tablo sayfası:

| Set | Örnek | Transkripsiyon | Tablo |
|---|---|---|---|
| train | 3.253 (A 784, B 814, C 813, D 842) | 2.565 | 688 |
| val (A) | 370 | 305 | 65 |
| test_A / B / C / D | her biri 371 | 299 | 72 |

Denetimde 985 PDF, 16.567 sayfa tarandı; sayfaların %4,1'inde metin katmanı yok (transkripsiyondan elenir).
`data/` klasörü (PDF'ler, görüntüler, veri seti) git'e girmez. `data/dataset.zip` yaklaşık 840 MB.

## Eğitim ve değerlendirme (Colab, A100 80 GB)

```
train_lora.py       LoRA SFT: --backend unsloth|hf, bf16, 8-bit AdamW, --dry-run
merge_lora.py       adaptörü baz modele birleştirir (vLLM için tam model klasörü)
eval_vllm.py        vLLM ile toplu çıkarım (--engine hf yedek), test_A..D
report.py           baz vs LoRA karşılaştırma raporu
model_zoo.py        model tanımları (hazır: qwen25vl_7b, qwen3vl_8b)
build_notebooks.py  notebook üretici (güncel değil, bkz. DURUM.md)
```

Gerçek koşuda kullanılan ayarlar:

| | |
|---|---|
| Yöntem | 16-bit LoRA (unsloth, bf16), r=32, alpha=64, lr 2e-4 |
| Veri | train'in ilk 1000 örneği (karışık seviye, karışık görev), toplamın yaklaşık %31'i |
| Epoch / batch | 2 epoch, batch 4 x grad-accum 8 (etkin 32), 64 adım |
| Görüntü | en fazla 700k piksel |
| Süre | Qwen2.5-VL-7B 61 dk, Qwen3-VL-8B 38 dk |
| Değerlendirme | vLLM 0.31, greedy (temperature 0), en fazla 3072 token |

Tam veri ve 3 epoch Qwen2.5-VL-7B'de yaklaşık 5 saat sürüyor; bu daha uzun koşu henüz yapılmadı.
Eğitim komutları: `colab/train_chain.sh`. Eval: `colab/setup_vllm_env.sh`, sonra `colab/eval_all.sh`.
unsloth ve vLLM aynı ortama kurulmadığı için eval, aynı Colab runtime'ında ayrı bir venv'de çalışıyor.

Colab Secrets: `GH_TOKEN` (repoyu okuyabilen token), `HF_TOKEN`. Drive'a `kap_lora/dataset.zip` yüklenir.
GPU olmadan yerel doğrulama: `python train_lora.py --model qwen25vl_7b --data data/dataset --dry-run`.

## Sırada ne var

- Daha uzun eğitim (tam train seti, 3 epoch): D seviyesi ve tablo satır sayısı için en olası kazanç.
- Referansı çok kısa (damga-only) sayfaları eğitim verisinden de çıkarmak.
- Tekrar döngüsünü azaltmak için `--repetition-penalty` denemesi (kodda var, GPU'da henüz denenmedi).
- Notebook'ların gerçek koşuya göre güncellenmesi.
- Başka modeller (Gemma 3, InternVL3.5, Llama 3.2 Vision) ve hazır OCR modelleriyle zero-shot karşılaştırma.

## Güvenlik notu

API anahtarı ve token'lar koda veya repoya yazılmaz. `llm.py` anahtarı `EVREN_API_KEY` ortam değişkeninden okur; Colab tarafında token'lar Colab Secrets'tan gelir.
