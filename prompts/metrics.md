# metrics.py

Transkripsiyon ve tablo cikarimi icin degerlendirme metrikleri. `rapidfuzz` kullan (rapidfuzz.distance.Levenshtein,
rapidfuzz.process). Colab'da da calisacak, yalniz stdlib + rapidfuzz.

## Fonksiyonlar (hepsi saf fonksiyon)
- `normalize(s)`: NFC, `\r` sil, satir sonu bosluklarini temizle, ardisik bosluklari/bos satirlari sadelestir, strip.
- `cer(ref, hyp)`: Levenshtein.distance(ref, hyp) / max(len(ref), 1). Girdileri normalize et.
- `wer(ref, hyp)`: kelime listeleri (split) uzerinde Levenshtein.distance (listeler icin de calisir) / max(len(ref_words),1).
- `turkish_char_acc(ref, hyp)`: referanstaki ozel Turkce karakterlerin (`çğıöşüÇĞİÖŞÜ`) kacinin hyp'te dogru uretildigi.
  `Levenshtein.opcodes(ref, hyp)` ile hizala; 'equal' bloklarindaki ref ozel karakterlerini say / ref'teki toplam ozel karakter.
  Ref'te hic ozel karakter yoksa `None` don.
- `number_f1(ref, hyp)`: regex `\d[\d.,]*\d|\d` ile sayi tokenlari; multiset (collections.Counter) kesisimi ile precision/recall/F1.
  Ikisinde de sayi yoksa 1.0.
- `parse_rows(text)`: modelin cikti metninden JSON liste cikar. ```json code fence'lerini temizle, ilk `[` ile son `]` arasini
  json.loads et; basarisizsa `None`. Her eleman dict olmali (kalem, dipnot, degerler), degilse None.
- `table_scores(ref_rows, hyp_rows)`: hyp None ise `{"valid":0, ...hepsi 0}`. Aksi halde:
  - `valid`: 1
  - `row_count_match`: len esit mi (0/1)
  - Satirlari `kalem` benzerligine gore bire bir eslestir (rapidfuzz.process.cdist ya da acgozlu: ref satirlari sirayla,
    kullanilmamis hyp satirlari arasinda en yuksek `fuzz.ratio`; esik 60).
  - `label_cer`: eslesen satirlarda kalem CER ortalamasi (eslesmeyen ref satiri CER=1 sayilir).
  - `value_exact`: ref satirlari icinde, eslesen hyp satirinin `degerler` listesi birebir ayni olanlarin orani.
  - `cell_exact`: tum ref hucreleri (degerler elemanlari) icinde dogru yerde dogru sayi olan hucre orani (eslesmeyen satirdakiler yanlis).
- `score_record(task, ref, pred)`: task "transcribe" ise `{"cer","wer","tr_char_acc","number_f1"}`;
  "table" ise ref'i json.loads ile liste yapip `parse_rows(pred)` ile `table_scores`.
- `summarize(records)`: records = [{"id","task","split","ref","pred"}], (task, split) gruplarina gore her metrigin
  ortalamasi (None'lari atla) ve ornek sayisi; dict dondur.
- `format_summary(summary)` : okunur metin tablosu string'i.

## CLI
- `python metrics.py preds.jsonl`: satirlari oku (alanlar: id, task, split, ref, pred; `split` yoksa "all"), summarize, yazdir,
  ayrica `--out ozet.json` secenegi.
- `python metrics.py --selftest`: assert tabanli kendi testleri:
  cer("kitap","kitap")==0, cer("çalışma","calisma") > 0 ve tam degeri elle dogrula (ç->c, ı->i, ş->s: 3/7),
  turkish_char_acc("çalışma","calisma")==0.0 (ç,ı,ş hepsi yanlis -> 0/3), turkish_char_acc("çalışma","çalışma")==1.0,
  number_f1("1.234,56 ve 7","7 ve 1.234,56")==1.0, parse_rows ile bozuk JSON -> None, table_scores mukemmel eslesmede hepsi 1/0 uygun.
  Sonunda "selftest ok" yazdir.
