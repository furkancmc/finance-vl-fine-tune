# table_extract.py

Dijital PDF sayfasindan (pymupdf kelime koordinatlari ile) finansal tabloyu (bilanco, gelir tablosu vb.) satir satir cikaran modul.
Mevcut `examples/extract_demo.py` mantiginin genellestirilmis hali: ncols ve x_split elle verilmez, otomatik bulunur.

## Arayuz
`extract_page_table(page) -> list[dict] | None`
- `page`: pymupdf.Page. Sayfa tablo degilse `None`; tablo ise satir listesi:
  `{"kalem": str, "dipnot": str | None, "degerler": [int|float, ...]}`.
- Ayrica `parse_number(token) -> int|float` (Turkce bicim: `1.234.567` -> 1234567, `1.234,56` -> 1234.56,
  `(1.234)` -> -1234, `-` -> 0, `-1.234` -> -1234).

## Algoritma
1. `page.get_text("words")` -> kelimeleri y merkezine gore sirala; ayni satir tolerans 3 pt (demo'daki `lines`).
2. Sayi tokeni regex: `^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?$|^-$` (demo'daki NUM).
3. Her satirda SAGDAN baslayarak ardisik sayi tokenlarini topla (soldan metin kalemi gelir). n = o sayi.
4. `ncols` = n >= 2 olan satirlarin n degerlerinin modu. En az 8 satir `n >= ncols` ise sayfa tablo sayilir, degilse `None`.
   n == ncols + 1 olan satirlarda soldaki ekstra token `dipnot`'tur (ornek "16"); degerler son `ncols` token.
5. Satir etiketi: o satirdaki sayi olmayan kelimeler soldan saga birlesir. Sayi icermeyen satirlar (alt baslik, cok satirli etiket
   devami) `pending` listesine girer ve sonraki sayili satirin basina eklenir; ancak pending en fazla 2 satir ve toplam 120 karakter
   olsun (daha uzunsa sadece son 2 satiri tut) ki sayfa basligi ilk satira yapismasin.
6. Elenecek satirlar: etiketi bos ya da 3 karakterden kisa olanlar; tum degerleri 0..31 araliginda kucuk tamsayi ve etiketi bos olanlar
   (ornek "1, 31" ay/gun artiklari); etiketi "Dipnot", "Cari Dönem", "Önceki Dönem", "Bağımsız Denetimden" ile baslayanlar.
7. Kalan satir sayisi 6'dan azsa veya 120'den fazlaysa `None`.
8. Etiketleri NFC normalize et, ardisik bosluklari tek bosluga indir.

## CLI
`python table_extract.py <pdf> [sayfa_no(1 tabanli)]`: sayfa verilmezse tum sayfalari tara, tablo bulunan sayfalari ve satir sayisini yazdir;
sayfa verilirse satirlari `json.dumps(..., ensure_ascii=False, indent=1)` ile yazdir. Konsol kodlamasi icin
`sys.stdout.reconfigure(encoding="utf-8")` kullan.
`python table_extract.py --scan`: `data/docs/*/attachment_*.pdf` icinden ilk 40 PDF'i tara, kac sayfada tablo bulundugunu ve
ortalama satir/ncols dagilimini ozetle.
