# KAP Türkçe VL Fine-Tune

Türkçe finansal belgeler (KAP, Kamuyu Aydınlatma Platformu) üzerinde bir vision-language modelin
**Türkçe öğrenme kapasitesini LoRA ile ölçmek** için veri hattı. Model sayfa görüntüsünü okur; çıktısı,
PDF'in kendi metin katmanından gelen referansla karşılaştırılır. Eğitim Colab A100'de yapılacak.

> Durum: veri hattı hazır. Eğitim/değerlendirme notebook'ları, fine-tune edilecek model seçilince eklenecek.

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
PLAN.md            ayrıntılı plan ve kararlar
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

## Veri seti

- Bölme **şirket bazında** (%80/10/10), aynı şirket iki tarafta yer almaz.
- `train`: dört seviye eşit ağırlıkla karışık. `val`: seviye A.
- `test_A`, `test_B`, `test_C`, `test_D`: aynı test sayfaları, dört seviyede. Bozulmaya dayanıklılık seviye seviye ölçülür.
- Her kayıt: `id, task, split, level, image, messages (user: görüntü + prompt, assistant: cevap), ref`.
- Seed'ler sabit (42), her şey tekrar üretilebilir.

## Notlar

- `data/` (PDF'ler, görüntüler, veri seti) git'e girmez; yalnızca kod ve spec'ler.
- API anahtarı sadece `EVREN_API_KEY` ortam değişkeninde, kodda yok.
- Denetim: 985 PDF, 16.567 sayfa; sayfaların %4,1'inde metin katmanı yok (transkripsiyondan elenir).
