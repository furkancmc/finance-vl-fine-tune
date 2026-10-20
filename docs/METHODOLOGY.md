# Yöntem: ne yaptık, nasıl yaptık

Bu belge projenin tüm hattını sırayla anlatır: veri nasıl toplandı, etiketler nereden geldi, görüntüler nasıl bozuldu,
modeller nasıl eğitildi ve nasıl ölçüldü. Sonuçlar için [`../results/FINAL_RESULTS.md`](../results/FINAL_RESULTS.md),
özet için [`../README.md`](../README.md).

## 1. Soru

Açık ağırlıklı görüntü-dil (VL) modelleri, Türkçe finansal belgelerde (KAP bildirimleri) sayfa görüntüsünden
metni ve tabloları ne kadar doğru okuyor? Küçük bir **LoRA** ince ayarı bunu ne kadar iyileştiriyor, ve bu iyileşme
görüntü kalitesi bozuldukça (tarama, fotoğraf benzeri bozulmalar) nasıl değişiyor?

Karşılaştırdığımız beş model, hepsi aynı veri, aynı protokol, aynı metriklerle:

| Model | Ailesi | Parametre |
|---|---|---|
| Qwen3-VL-8B-Instruct | Qwen | 8B |
| Qwen2.5-VL-7B-Instruct | Qwen | 7B |
| InternVL3.5-8B (HF) | InternVL | 8B |
| Gemma 3 12B-it | Gemma | 12B |
| Llama 3.2 11B Vision-Instruct | Llama | 11B |

## 2. Veri

### 2.1 Kaynak
KAP (Kamuyu Aydınlatma Platformu) bildirimleri, **Ocak–Eylül 2026**. Bildirim türü bazlı kotalarla örneklenmiş **500 bildirim**
(391 şirket, 17 bildirim türü; en büyük pay Finansal Rapor). Her bildirim için ana PDF ve en fazla 5 ek PDF indirildi
(`kap_collect.py`). Seçim `data/selection.json` içine yazılır, tekrar çalıştırınca aynı seçim kullanılır (seed 42).
Toplam **3169 sayfa**.

### 2.2 Sayfa görüntüsü ve etiket (referans)
- Her PDF sayfası PyMuPDF ile **130 DPI** görüntüye çevrilir, uzun kenar en fazla 1400 piksel (`make_pages.py`).
- **Referans metin PDF'in kendi metin katmanıdır** (dijital PDF, elle etiketleme yok). Modele metin katmanı gösterilmez;
  model yalnızca sayfa görüntüsünü görür. Metin uzunluğu 50–5000 karakter olan sayfalar alınır.
- **Tablo etiketleri:** Finansal Rapor eklerinin ilk 120 sayfası taranır, belge başına en fazla 6 tablo sayfası seçilir.
  Satırlar, PDF metin katmanındaki konum ve sayı kalıplarından (Türkçe sayı biçimi `1.234,56`, parantezli negatif `(123)`,
  `-` = 0) heuristik olarak çıkarılır ve `{"kalem", "dipnot", "degerler"}` biçiminde JSON listesine çevrilir
  (`table_extract.py`). Bu etiketler **otomatik üretildiği için gürültü içerebilir** (bkz. Sınırlamalar).

### 2.3 Görev tanımı
Aynı sayfa iki görevde kullanılabilir (tablo içeren sayfalar ikisinde de yer alır):

1. **Transkripsiyon:** `"Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz."` → sayfa metni.
2. **Tablo:** `"Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için kalem, dipnot ve degerler alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir)."` → JSON satır listesi.

### 2.4 Bozulma seviyeleri
Asıl hedef taranmış veya fotoğraflanmış belgedir, ama etiketli gerçek tarama yoktur. Bu yüzden dijital sayfalar
**yapay olarak** dört seviyede bozulur (`degrade.py`): rastgele gri tonlama, çözünürlük düşürüp geri büyütme, hafif döndürme,
Gauss bulanıklığı, kontrast/parlaklık kayması, kenar gölgesi, gürültü ve JPEG sıkıştırma. Her adım bir olasılıkla uygulanır;
sayfa başına tohum sabittir (aynı sayfa her çalıştırmada aynı bozulmayı alır).

| Seviye | Anlamı | Çözünürlük çarpanı | Döndürme (en çok) | Bulanıklık | Gürültü σ | JPEG kalitesi |
|---|---|---|---|---|---|---|
| A | bozulmamış | 1,0 | 0° | – | – | – |
| B | hafif | 0,85–1,0 (p=0,5) | 0,8° | 0,2–0,6 | 1–5 | 75–92 |
| C | orta | 0,6–0,85 (p=0,8) | 2,5° | 0,6–1,4 | 5–14 | 45–75 |
| D | ağır | 0,4–0,55 (her zaman) | 4° | 1,0–1,8 | 14–28 | 12–35 |

(Değerler `degrade.py` içindeki `LEVELS` sözlüğünden. Bozulma türleri olasılıkla uygulanır; ayrıntı koddadır.)

> **Önemli:** bu yapay bozulmadır, gerçek tarama değildir. Gerçek taramada sonuçlar farklı olabilir.

### 2.5 Veri seti (v2) ve bölme
`make_dataset.py`, **şirket bazlı** bölme yapar (bir şirketin sayfaları yalnızca tek bölümde olur, böylece şirket
adı/biçim sızıntısı olmaz): 391 şirket → **261 eğitim / 50 doğrulama / 80 test** (seed 42).

| Bölüm | Örnek | Transkripsiyon | Tablo | Not |
|---|---|---|---|---|
| Eğitim | 2500 | 2000 | 500 | Seviyeler A/B/C/D ≈ %25'er (sayfa başına rastgele seviye) |
| Doğrulama | 500 | 388 | 112 | Bozulmamış (A) |
| Test (her seviye) | 700 | 567 | 133 | **Aynı sayfalar** A, B, C, D seviyelerinde (dört kopya) |

Filtreler ve seçim kuralları:
- Referans metni **150 karakterden kısa** olan transkripsiyon sayfaları atılır. Bunlar PDF'te yalnızca SPK
  "Doğrulama Kodu" damgası taşıyan sayfalardır; model sayfadaki asıl içeriği okuduğunda hatalı "yanlış" sayılırdı
  (v1'de test CER'i bu yüzden şişmişti; bkz. `results/RESULTS.md`).
- Eğitimde tablo sayfaları (500) ve geri kalan sayfalar bildirimler arasında dengeli (round-robin) seçilir ki tek
  bir bildirim eğitimi domine etmesin. Tam sayılar (2000 / 500, val 500, test 700) sabitlenmiştir.

## 3. Eğitim

`train_lora.py` + [Unsloth](https://github.com/unslothai/unsloth) (`FastVisionModel`) + Hugging Face `Trainer`.
Hepsi için aynı ayarlar (Gemma ve Llama'da bellek için batch 2 × birikim 16):

| Ayar | Değer |
|---|---|
| Yöntem | LoRA, 16-bit (bf16), 4-bit niceleme yok |
| Rank / alpha / dropout | r=32, alpha=64, 0 |
| Eğitilen kısım | Dil modeli dikkat + MLP katmanları. **Görüntü kodlayıcı dondurulmuş** |
| Veri / epoch | 2500 örnek × 2 epoch = 156–158 adım |
| Etkin batch | 32 (4 × 8; Gemma/Llama 2 × 16) |
| Öğrenme hızı | 2e-4, cosine çizelge, %5 ısınma, ağırlık sönümü 0,01, gradyan kırpma 1,0, `adamw_8bit` |
| Görüntü çözünürlüğü | en çok 700 000 piksel (sabit çözünürlüklü Gemma/Llama kendi işlemcisine göre) |
| Kayıp | Yalnızca asistan cevabı (isteme ait token'lar maskelenir; `unmaskable=0` doğrulandı) |
| Tohum | 42 |

| Model | Süre | Son eğitim kaybı | Son doğrulama kaybı |
|---|---|---|---|
| Qwen3-VL-8B | 90 dk | 0,054 | 0,0163 |
| Qwen2.5-VL-7B | 152 dk | 0,092 | 0,0201 |
| InternVL3.5-8B | 151 dk | 0,142 | 0,0509 |
| Gemma 3 12B | 128 dk | 0,300 | 0,3494 |
| Llama 3.2 11B Vision | 134 dk | 0,145 | 0,0581 |

(A100 80 GB, tek koşu. **Kayıp değerleri modeller arasında karşılaştırılamaz**: tokenizer'lar farklı. Kalite için CER'e bakılır.)
Yazılım: torch 2.11, transformers 5.17, peft 0.21, trl 1.15, unsloth 2026.10.

## 4. Değerlendirme

Aynı test sayfaları, her model için **baz (ince ayarsız)** ve **LoRA** sürümüyle, dört seviyede (`eval_vllm.py`):
- Açgözlü çözümleme (`temperature=0`), `repetition_penalty=1.0`, en çok 3072 yeni token (Llama: 2048), görüntü en çok 700 000 piksel.
- Qwen3-VL, Qwen2.5-VL, InternVL3.5 ve Gemma 3: **vLLM 0.32**. Llama 3.2 Vision bu vLLM sürümünde desteklenmediği için
  **Hugging Face `generate`** ile ve yalnızca her bölümün **ilk 100 örneğinde** (4 seviye × 100) değerlendirildi.
- Tüm modeller aynı protokolle ve aynı promptlarla ölçüldü, hiçbirinde test setine bakılarak ayar yapılmadı.

### Metrikler (`metrics.py`)
**Transkripsiyon** (metin Unicode NFC normalizasyonundan geçer, çoklu boşluk ve satır birleştirilir):
- **CER** = karakter düzeyi Levenshtein mesafesi / referans uzunluğu. **Medyan** raporlanır (az sayıda uç sayfa ortalamayı şişirir).
- **WER** = aynı şey kelime düzeyinde.
- **Türkçe karakter doğruluğu:** referanstaki `ç ğ ı İ ö ş ü` karakterlerinden doğru hizalananların oranı.
- **Sayı F1:** referans ve çıktıdaki sayı dizilerinin çoklu küme kesişimi üzerinden F1 (tutarlar, tarihler).

**Tablo:**
- **Geçerli JSON:** çıktı beklenen şemada (`kalem`, `degerler` listesi) ayrıştırılabiliyor mu.
- **Hücre doğruluğu (`cell_exact`):** satırlar `kalem` metnine bulanık eşleştirmeyle (benzerlik ≥ 60) eşlenir; eşleşen satırdaki
  referans değerlerden birebir tutanların oranı. Geçersiz JSON = 0.
- **Satır sayısı eşleşmesi:** çıktıdaki satır sayısı referansla **birebir** eşit olan sayfaların oranı (katı: bir satır fazla/eksik = 0).

### Temiz görünüm
Tüm tablolar "temizlenmiş" görünümdür: (1) çıktının **sonundaki** tekrar bloğu kesilir (`strip_repetition`; token
sınırına kadar tekrar eden kaçak üretimler için), (2) referansı 150 karakterden kısa damga-only sayfalar dışlanır.
Baz ve LoRA aynı temizlemeden geçer.

### Güven aralıkları
`final_report.py`: sayfa düzeyinde **%95 bootstrap** (1000 yeniden örnekleme, tohum 42). Medyan CER için medyanın, diğer
metrikler için ortalamanın aralığı. Yalnızca **tek eğitim koşusu** var; aralıklar test-sayfası örneklemesinden gelen belirsizliği
gösterir, **eğitim tohumu belirsizliğini göstermez**.

## 5. Yeniden üretme

```bash
python kap_collect.py && python inventory.py      # veri indirme
python make_pages.py                               # sayfa görüntüsü + metin/tablo etiketi
python make_dataset.py --zip                       # bozulma + bolme -> data/dataset.zip
bash colab/setup_all.sh && bash colab/run_all.sh   # Colab A100: 5 model egitim + degerlendirme (gece zinciri)
python final_report.py --eval-root <drive>/v2/eval --out results/final
python make_examples.py --eval-root <drive>/v2/eval
```
Ham KAP verisi ve PDF'ler repoda yoktur (yeniden dağıtım koşulları ayrıca doğrulanmalıdır); `kap_collect.py` ile yeniden indirilir.

## 6. Sınırlamalar (açıkça)

1. **Yapay bozulma:** Taranmış belge sonuçları gerçek taramadan değil yapay bozulmadan gelir. Gerçek tarama seti yok.
2. **Etiket gürültüsü:** Referans PDF metin katmanıdır. Okuma sırası (çok sütun, tablo hücreleri) ve tablo satır ayrımı
   heuristiktir; **tablo "satır sayısı eşleşmesi" düşüklüğünün bir kısmı model değil etiket kaynaklı olabilir.** Etiketlerin
   küçük bir kısmı elle doğrulanmadı.
3. **Tek koşu, tek tohum.** Eğitim rastgeleliği (tohum) ölçülmedi. Güven aralıkları yalnızca test örneklemini yansıtır.
   Birbirine yakın modeller (özellikle Qwen3 ve Qwen2.5, A–C seviyeleri) arasında sıralama iddia edilmemelidir.
4. **Llama 3.2 eksik protokol:** Test setinin yalnızca ilk 100 örneği (67 transkripsiyon + 33 tablo / seviye) ve HF motoruyla;
   diğer modellerle doğrudan kıyas için `results/FINAL_RESULTS.md` içindeki **ortak alt küme** tablosuna bakın.
5. **Eğitim kısa:** 2 epoch, 2500 örnek. Daha uzun eğitim sonuçları değiştirebilir.
6. **Baz model hatası kısmen format hatasıdır.** Bazı modeller (InternVL3.5, Gemma 3, Llama 3.2) baz haliyle JSON
   şemasını üretemiyor (geçerli JSON ≈ %0–5); bu yüzden onlarda tablo kazancının büyük kısmı "formatı öğrenme"dir,
   okuma becerisi artışından ayrı düşünülmeli.
7. **OCR baz çizgileri tamamlanmadı.** PaddleOCR-VL, HunyuanOCR ve DeepSeek-OCR zero-shot karşılaştırması planlandı, tamamlanmadı.
8. **Veri ve model lisansları:** KAP içeriğinin yeniden dağıtımı için kullanım koşulları doğrulanmalıdır. Gemma ve Llama
   türevlerinin kendi lisans şartları vardır (bkz. model kartları).
