# make_pages.py

PDF'lerden sayfa goruntusu + referans metin uretir, `data/pages.jsonl` yazar.

## Sayfa secimi (SEED=42, belge basina `random.Random(f"{SEED}-{doc_id}")`)
- Aday sayfa: metin katmani olan sayfa; `page.get_text("text", sort=True)` normalize edildikten sonra 50 <= uzunluk <= 5000 karakter.
- `main.pdf`: aday sayfalarin hepsi, en fazla 3 tane (rastgele, sayfa sirasina gore sirali tut).
- Her `attachment_N.pdf`: en fazla 3 aday sayfa rastgele; belge basina eklerden toplam en fazla 8 sayfa (`--max-attach-pages`).
- Ek olarak `type == "Finansal Rapor"` belgelerinin eklerinde TUM sayfalari `table_extract.extract_page_table(page)` ile tara
  (`from table_extract import extract_page_table`); tablo bulunan sayfalardan belge basina en fazla 4 tanesini ekle (secili sayfalarla
  tekrar etmesin) ve bu sayfalarin kaydinda `"table"` alanina sonucu yaz. Tablo sayfasi olmayan kayitlarda `"table": null`.
  Secilen sayfalar tablo sayfasi cikarsa da (tablo bulunursa) `table` alanini doldur.
- Tarama kisa surmeli: tablo taramasi icin 40 sayfadan uzun ekleri atlama, sadece ilk 60 sayfaya bak.

## Goruntu
- `page.get_pixmap(dpi=130)` -> PIL; en uzun kenar 1400 pikseli asarsa LANCZOS ile 1400'e kucult. RGB PNG olarak
  `data/pages/<doc_id>/<stem>_p<N>.png` (klasoru olustur). Dosya varsa yeniden cizme (devam edilebilir).

## Referans metin
- `page.get_text("text", sort=True)`; NFC; her satirin sonundaki bosluklari sil; 3+ ardisik bos satiri 1 bos satira indir; strip.

## Calisma
- Belge listesi `data/metadata.csv`'den (utf-8-sig); `type`, `company`, `stockCode` alanlari kayda gecer (`stock_code` olarak).
- `multiprocessing.Pool(--workers, varsayilan 6)` ile belge basina isle (her isci kendi pymupdf belgesini acar); `imap_unordered`,
  her 25 belgede ilerleme yazdir. Sonuclari belge indeksine gore sirala ve `data/pages.jsonl`'a yaz (her calistirmada bastan yaz).
- `--limit N`: ilk N belgeyi isle (test icin). Windows spawn uyumlu (`if __name__ == "__main__":`).
- Sonda ozet: toplam sayfa, tur basina sayfa sayisi, tablo sayfasi sayisi, toplam karakter, ortalama karakter/sayfa.
