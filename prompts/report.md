# report.py

Birden fazla `preds_<tag>_<split>.jsonl` dosyasindan karsilastirma raporu uretir. Colab ve yerel calisir (sadece stdlib + rapidfuzz, `metrics.py`).

## Argumanlar
`--eval-dir` (tek ya da birden fazla; her biri bir model icin `.../eval/<model_key>` klasoru, ornek `--eval-dir a --eval-dir b`),
`--out` (rapor klasoru, varsayilan ilk eval-dir'in ust klasoru / "report").

## Davranis
- Her eval-dir icin klasor adini `model` say; `preds_*.jsonl` dosyalarini oku (dosya adi: `preds_<tag>_<split>.jsonl`; tag ve split dosya adindan ayristirilir,
  split `test_A..test_D`).
- Her kayit icin `metrics.score_record(task, ref, pred)` hesapla (bos pred'i "" say), (model, tag, task, split) gruplarina gore ortalama al
  (None'lari atla), ornek sayisi ile birlikte.
- Cikti `report.csv` (sutunlar: model, tag, task, level (split'ten A-D), n, ve metrikler: cer, wer, tr_char_acc, number_f1; table icin valid, row_count_match,
  label_cer, value_exact, cell_exact) ve `report.md`:
  * transkripsiyon icin: satirlar = model/tag, sutunlar = seviye A..D, hucre = CER (ve altta tr_char_acc tablosu),
  * tablo icin: satirlar = model/tag, sutunlar A..D, hucre = value_exact ve valid orani,
  * "kazanc" tablosu: her model icin tag=lora vs tag=base CER farki (negatif iyi) seviye bazinda.
- stdout'a report.md'yi yazdir. Eksik kombinasyonlar "-" olsun.
