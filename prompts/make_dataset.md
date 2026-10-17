# make_dataset.py

`data/pages.jsonl`'dan egitim/degerlendirme veri setini uretir (duzen icin 00_common.md'ye bak).

## Bolme (sirket bazinda, sizinti yok)
- Benzersiz `company` degerlerini `random.Random(SEED).shuffle` ile karistir; sirket sayisina gore ilk %80 train, sonraki %10 val,
  son %10 test. Bir sirketin tum sayfalari ayni bolumde.

## Gorev ve prompt'lar (sabit Turkce metin, sabitler olarak dosya basinda)
- `PROMPT_TRANSCRIBE` = "Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz."
- `PROMPT_TABLE` = "Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için `kalem`, `dipnot` ve `degerler` alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir)."
- Her sayfa icin bir "transcribe" ornegi (cevap = `text`). `table` alani dolu sayfalar icin ayrica bir "table" ornegi
  (cevap = `json.dumps(table, ensure_ascii=False)`; 4000 karakterden uzunsa o ornegi atla).
- Transcribe cevabi 5000 karakteri asmaz (make_pages zaten eliyor).

## Goruntu ve bozulma
- Kaynak: `data/<image>` (PNG). Cikti: `data/dataset/images/<id>[_deg].jpg`, JPEG kalite 92, RGB. Mevcut dosyayi yeniden yazma.
- `from degrade import degrade` (degrade(img, rng) -> (img, params)). Her ornek icin `rng = random.Random(f"{SEED}-{id}")`.
- train: her ornek icin %50 olasilikla bozulmus goruntu (`degraded: true`), yoksa temiz.
- val: temiz.
- test: iki dosya. `test_clean.jsonl` temiz, `test_degraded.jsonl` ayni orneklerin bozulmus hali (ayni `id` + "_deg" eki yok, `id` ayni,
  `split` alanlari "test_clean"/"test_degraded").
- Bozulma parametreleri her kayitta `degrade_params` alaninda (temizlerde null).

## Cikti ve ozet
- `data/dataset/{train,val,test_clean,test_degraded}.jsonl` (00_common.md'deki satir semasi, `split` alani dosya adiyla ayni).
- `data/dataset/stats.json` ve stdout: bolum basina ornek sayisi, gorev basina sayi, sirket sayisi, ortalama cevap uzunlugu (karakter),
  bozulmus oran.
- `--limit N` (test icin ilk N sayfa), `--zip`: `data/dataset.zip` uret (dataset klasoru, ZIP_STORED, jpg zaten sikismis).
- Ilerleme her 200 ornekte yazdir.
