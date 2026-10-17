# KAP Finans VL Fine-Tune: Plan

Varsayilan hedef (teyit gerekli): KAP PDF sayfalarini (goruntu) okuyup Turkce finansal bilgiyi
yapisal JSON'a / soru-cevaba ceviren bir vision-language modeli fine-tune etmek.

## Roller
- Claude Code: yonetim. Dosya/komut/git/test calistirir, ciktiyi dogrular, her adimi commit'ler.
- `llm.py` (deepseek/deepseek-v4.1-flash): kod uretimi ve metin tabanli etiketleme isleri.
- Not: DeepSeek metin modeli, goruntu gormez. Etiketleri PDF metin katmanindan (pymupdf) uretir.

## Asamalar

### 0. Veri toplama (devam ediyor)
- `kap_collect.py` 500 bildirimi indiriyor (~336/500). Bitince tekrar calistirip hatalilari tamamla.
- Cikis: `data/docs/<id>/{main.pdf, attachment_N.pdf, detail.json.gz}`, `data/metadata.csv`.
- Kabul: metadata.csv'de `fail` sayisi ~0, tur kotalari dolu.

### 1. Veri denetimi
- Her PDF: sayfa sayisi, metin katmani var mi (taranmis mi), boyut.
- Tur bazli ozet tablosu -> `data/inventory.csv`. Bozuk/bos PDF'leri ele.

### 2. Sayfa -> goruntu + metin
- pymupdf ile sayfa PNG (150-200 dpi) ve kelime/kutu cikarimi.
- Cikis: `data/pages/<id>/p<N>.png`, `p<N>.json` (metin + koordinat).
- Mojibake: SORUN DEGIL. `inventory.py` ile dogrulandi, PDF metni dogru Unicode (U+0130, U+015E...);
  bozuk gorunum sadece Windows konsolu. Dosyalari her zaman utf-8 ile oku/yaz, konsola Turkce basarken
  `python -X utf8` kullan.
- Denetim (781 PDF, 14.838 sayfa): `�` yok; sayfalarin %4,2'si metin katmansiz (cogu ekler, %4,8).
  Bu sayfalar referans uretemez, transkripsiyon veri setinden elenecek.

### 3. Gorev tanimi (karar gerekli, asagida)
- Onerilen baslangic: (a) gelir tablosu/bilanco tablo cikarimi -> JSON,
  (b) bildirimden alan cikarimi (sirket, tarih, tutar, olay), (c) sayfa hakkinda soru-cevap.

### 3b. Tarama bozulmasi (KESIN, kullanici karari)
- Dijital sayfa goruntulerine gercek taranmis belge gibi bozulma uygulanir: bulaniklik, hafif egiklik
  (rotate +-2 derece), gurultu, JPEG sikistirma, kontrast/parlaklik degisimi, dusuk dpi.
- Egitimde karisik: temiz + bozulmus. Testte iki ayri set: `test_clean` ve `test_degraded`,
  boylece taranmis belgeye genelleme ayri olculur. Referans metin her zaman pymupdf'ten.
- Bozulma seed'li ve tekrar uretilebilir olmali (`degrade.py`), parametreler metadata'ya yazilir.

### 4. Etiket uretimi (ajan dongusu)
Her ornek icin dongu:
1. Planla: Claude Code gorevi ve girdiyi secer (sayfa metni + kutular).
2. Uret: `llm.py` ile aday etiket (JSON) uretilir; sema prompt'ta verilir.
3. Dogrula (kural tabanli): JSON sema, sayisal tutarlilik (toplamlar tutuyor mu),
   deger gercekten sayfa metninde geciyor mu.
4. Basarisizsa hata mesajiyla yeniden uret (en fazla 2-3 deneme), yoksa `rejected/` klasorune.
5. Kabul edilenler `data/labels/<task>.jsonl`; her 50 ornekte commit.
- Ikinci bir gecis: LLM-judge ile cakisan ornekleri ornekleme (10%) + elle kontrol.

### 5. Veri seti paketleme
- Train/val/test bolumu SIRKET bazinda (ayni sirket iki tarafta olmasin).
- Format: chat/messages JSONL, goruntu yolu + soru + cevap. Veri kart (README).

### 6. Fine-tune
- Donanim: RTX 4060 Laptop, 8 GB VRAM -> 3-4B VLM + QLoRA (4-bit) siniri.
  Adaylar: Qwen2.5-VL-3B, Qwen3-VL-4B. Daha buyugu icin bulut GPU gerekir.
- Unsloth/TRL ile QLoRA, kucuk deneme kosusu (50 ornek) -> tam kosu.
- Goruntu cozunurlugunu sinirla (max_pixels), yoksa OOM.

### 7. Degerlendirme
- Test setinde: alan bazli dogruluk, tablo hucre dogrulugu, JSON gecerlilik orani.
- Baz model vs fine-tune karsilastirmasi. Hata analizi -> veri iyilestirme dongusu.

## Calisma kurallari
- Her adim sonunda commit + push. Kucuk, anlamli commit mesajlari.
- `data/` git'e girmez; kod, sema, prompt, sonuc ozetleri girer.
- Her script tekrar calistirilabilir (var olani atla) olmali.
- Anahtarlar sadece env'de (`EVREN_API_KEY`).

## Acik kararlar (senden)
1. Nihai gorev ne? (tablo cikarimi / alan cikarimi / QA / hepsi)
2. Model: yerel 8 GB'ye sigan 3-4B mi, bulut GPU ile daha buyuk mu?
3. 500 bildirim yeterli mi, yoksa tekrar daha cok toplanacak mi?
4. Etiket kalitesi icin elle kontrol payi (onerim %5-10).
