# Gerçek çıktı örnekleri (qwen3vl_8b)

Aşağıdakiler test setinden, **seçim kuralı aşağıda yazılı** otomatik seçilmiş örneklerdir (elle ayıklanmadı). Her örnekte aynı sayfa için referans (PDF metin katmanı), baz model ve LoRA çıktısı yan yana verilir. Uzun çıktılar kısaltılmıştır.

## Örnek 1 — temiz sayfa (A)

Sayfa: `1551616_attachment_1_p5` · seviye A · seçim kuralı: ilk sayfa: ref 400–1100 karakter, LoRA CER < 0,02, baz CER > 0,10

**Referans** (PDF metin katmanı)
```text
d. Hesap Dönemi İçinde Meydana Gelen Değişiklikler:

Yoktur.

B. İMTİYAZLI PAYLARA İLİŞKİN BİLGİLER

Yoktur.

C. ŞİRKETİN YÖNETİM ORGANI, ÜST DÜZEY YÖNETİCİ VE PERSONEL BİLGİLERİ

a. Şirketin Yönetim Organı:

9 kişiden oluşan Yönetim Kurulu’dur.

Yönetim Kurulu:

Adı Soyadı Görevi İmza Yetkisi Başlangıç – Bitiş Tarihi
Gamze Şebnem Muratoğlu Başkan 1. 19/03/2025- 19/03/2026
Dalya Kohen Başkan Yardımcısı 1. 19/03/2025- 19/03/2026
Ecem Nalbantgil Üye 1. 19/03/2025- 19/03/2026
Mehmet Ali Ersarı Üye 1. 19/03/2025- 19/03 …
```

**Baz model** — CER 0.326
```text
d. Hesap Dönemi İçinde Meydana Gelen Değişiklikler:
Yoktur.

B. İMTİYAZLI PAYLARA İLİŞKİN BİLGİLER
Yoktur.

C. ŞİRKETİN YÖNETİM ORGANI, ÜST DÜZEY YÖNETİCİ VE PERSONEL BİLGİLERİ
a. Şirketin Yönetim Organi:
9 kişiden oluşan Yönetim Kurulu'dur.

Yönetim Kurulu:

| Adı Soyadı         | Görevi           | İmza Yetkisi | Başlangıç – Bitiş Tarihi     |
|--------------------|------------------|--------------|------------------------------|
| Gamze Sebnem Muratoğlu | Başkan          | 1.           | 19/03/2025- 19/03/2026 …
```

**LoRA** — CER 0.007
```text
d. Hesap Dönemi İçinde Meydana Gelen Değişiklikler:

Yoktur.

B. İMTİYAZLI PAYLARA İLİŞKİN BİLGİLER

Yoktur.

C. ŞİRKETİN YÖNETİM ORGANI, ÜST DÜZEY YÖNETİCİ VE PERSONEL BİLGİLERİ

a. Şirketin Yönetim Organı:

9 kişiden oluşan Yönetim Kurulu’dur.

Yönetim Kurulu:

Adı Soyadı Görevi İmza Yetkisi Başlangıç – Bitiş Tarihi
Gamze Sebnem Muratoğlu Başkan 1. 19/03/2025- 19/03/2026
Dalya Kohen Başkan Yardımcısı 1. 19/03/2025- 19/03/2026
Ecem Nalbantgil Üye 1. 19/03/2025- 19/03/2026
Mehmet Ali Ersarı Üye 1. 19/03/2025- 19/03 …
```

## Örnek 2 — orta bozulma (C)

Sayfa: `1546346_main_p2` · seviye C · seçim kuralı: ilk sayfa: LoRA CER < 0,03, baz CER > 0,10

**Referans** (PDF metin katmanı)
```text
İzahname

İlgili Şirketler [NETCD]

İlgili Fonlar []

İzahname

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Bildirim İçeriği

Açıklamalar

Netcad Yazılım A.Ş.'nin Sermaye Piyasası Kurulu tarafından onaylanan İzahnamesinin 1. bölümü ekte yer almakta olup
aynı zamanda Şirket'in https://www.netcad.com/tr/halka-arz-dokumanlari sitesinden de erişilebilir.


Saygılarımızla.
```

**Baz model** — CER 0.150
```text
KAP
KARAYI ATIKLAMA PLATFORMU
PUBLIC DISCLOSURE PLATFORM

Izahname

İlgili Şirketler (NETCD)

İlgili Fonlar

Izahname

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Bildirim İçeriği

Açıklamalar

Netcad Yazılım A.Ş.'nin Sermaye Piyasası Kurulu tarafından onaylanan Izahnamesinin 1. bölümü ekte yer almakta olup aynı zamanda Şirket'in https://www.netcad.com/tr/halka-arz-dokumanlari sitesinden de erişilebilir.

Saygılarımla,
```

**LoRA** — CER 0.000
```text
İzahname

İlgili Şirketler [NETCD]

İlgili Fonlar []

İzahname

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Bildirim İçeriği

Açıklamalar

Netcad Yazılım A.Ş.'nin Sermaye Piyasası Kurulu tarafından onaylanan İzahnamesinin 1. bölümü ekte yer almakta olup
aynı zamanda Şirket'in https://www.netcad.com/tr/halka-arz-dokumanlari sitesinden de erişilebilir.

Saygılarımızla.
```

## Örnek 3 — ağır bozulma (D): LoRA iyileştirir ama kusursuz değil

Sayfa: `1543744_main_p2` · seviye D · seçim kuralı: ilk sayfa: 0,05 < LoRA CER < 0,30 ve baz LoRA'dan en az 0,1 kötü

**Referans** (PDF metin katmanı)
```text
Değerleme Raporu

İlgili Şirketler []

İlgili Fonlar []

Değerleme Raporu

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Bildirim İçeriği

Değerlemeyi Yapan Kuruluş İnvest Gayrimenkul Değerleme ve Danışmanlık A.Ş.

Avrupa Konutları Çamlıvadi 2025 Yıl Sonu Değerleme
Değerleme Raporunun İçeriği
Raporu

Rapor Tarihi 08.01.2026

Açıklamalar

Avrupa Konutları Çamlıvadi'ye ilişkin 2025 Yıl Sonu Değerleme Raporu ekte yer almakta …
```

**Baz model** — CER 0.338
```text
KAP
DEĞERLEME RAPORU

İlgili İşletmeler
İlgili Fonlar

Değerleme Raporu
Yapılan Açıklama Güvenilir mi?
Hayır (No)
Yapılan Açıklama Düzeltile mi?
Hayır (No)
Kuruya İlişkin Daha Önceden Yapılan Açıklamaların Tarihi

Raporun İçeriği
Değerlendirmeyi Yapan Kuruluş
İmport Gayrimenkul Değerleme ve Danışmanlık A.Ş.
Değerleme Raporunun İçeriği
Kuruya Konulmuş Çamurcadı 2025 Yılı Sonu Değerleme Raporu.
Rapor Tarihi
08.01.2024

Açıklamalar

Kuruya Konulmuş Çamurcadı ve Diğer 2025 Yılı Sonu Değerleme Raporu dikkatle okunmuştur …
```

**LoRA** — CER 0.177
```text
Değerleme Raporu

İlgili Şirketler

İlgili Fonlar

Değerleme Raporu

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Riskin İçeriği -

Değerlemeyi Yapan Kuruluş Imevet Gayrimenkul Değerleme ve Danışmanlık A.Ş.

Değerleme Raporunun İçeriği Konuya Konuflan Çamlıvacli 2025 Yıl Sonu Değerleme
Raporu

Rapor Tarihi 08.01.2026

Açıklamalar

Konuya Konuflan Çamlıvacli'ya İlişkin 2025 Yıl Sonu Değerleme Raporu ekte yer almaktadır.
```

## Örnek 4 — ağır bozulma (D): LoRA'nın da başarısız olduğu sayfa

Sayfa: `1571485_main_p2` · seviye D · seçim kuralı: ilk sayfa: LoRA CER > 0,5 (başarısızlık örneği)

**Referans** (PDF metin katmanı)
```text
Özel Durum Açıklaması (Genel)

İlgili Şirketler []

İlgili Fonlar []

Özel Durum Açıklaması (Genel)

Yapılan Açıklama Güncelleme mi? Hayır (No)

Yapılan Açıklama Düzeltme mi? Hayır (No)

Konuya İlişkin Daha Önce Yapılan Açıklamanın Tarihi -

Yapılan Açıklama Ertelenmiş Bir Açıklama mı? Hayır (No)

Bildirim İçeriği

Açıklamalar

Menkul Kıymet Yatırım Ortaklıklarına İlişkin Esaslar Tebliği (III-48.5) 36. Madde 2. fıkra gereğince Şirketimizin BİAŞ'da oluşan ağırlıklı ortalama fiyatın 11.03.2026 tarihinde pay

başına n …
```

**Baz model** — CER 0.891
```text
Özel Durum Açıklaması (Genel)

Açıklamalar
Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Özel Durum Açıklaması (Genel)

Açıklama Türü:
Öz …
```

**LoRA** — CER 0.794
```text
Özel Durum Açıklaması (Genel)

İlgili Şirketler []

İlgili Fonlar []

Özel Durum Açıklamalar (Genel)

Toplam Açıklama Eklenecek mi? Hayır (No)

Toplam Açıklama Eklenecek mi? Hayır (No)

Konuya İlişkin Daha Daha Daha Daha Daha Daha Daha
Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha Daha …
```

## Örnek 5 — tablo çıkarımı (A)

Sayfa: `1551587_attachment_1_p33` · seçim kuralı: ilk tablo sayfası: referans ≤ 900 karakter, LoRA hücre doğruluğu ≥ 0,9, baz en az 0,2 geride

**Referans** (PDF metin katmanından çıkarılan satırlar)
```json
[{"kalem": "Dönem başı", "dipnot": null, "degerler": [11805614, 6236459]}, {"kalem": "Dönem içinde ödenen", "dipnot": null, "degerler": [-1770797, -303524]}, {"kalem": "Dönem içindeki artış", "dipnot": null, "degerler": [12582597, 8819047]}, {"kalem": "Parasal Kayıp / Kazanç", "dipnot": null, "degerler": [-3970252, -2946368]}, {"kalem": "Dönem sonu", "dipnot": null, "degerler": [18647162, 11805614]}, {"kalem": "Dönem başı", "dipnot": null, "degerler": [63181919, 26598323]}, {"kalem": "Dönem içinde ödenen", "dipnot": null, "degerler": [-35897750, -36023787]}, {"kalem": "Dönem içindeki artış", "dipnot": null, "degerler": [53201875, 86937993]}, {"kalem": "Parasal Kayıp / Kazanç", "dipnot": null, "degerler": [-19917852, -14330610]}, {"kalem": "Dönem sonu", "dipnot": null, "degerler": [60568192, 63181919]}]
```

**Baz model** — hücre doğruluğu 0.00, geçerli JSON: 1
```json
[
    {
        "kalem": "Çalışanlara Sağlanan Faydalar (devamı) - Kullanılmamış izin karşılığı",
        "dipnot": "",
        "degerler": [
            {
                "yil": "2025",
                "deger": "11805614"
            },
            {
                "yil": "2024",
                "deger": "6236459"
            }
        ]
    },
    {
        "kalem": "Çalışanlara Sağlanan Faydalar (devamı) - Kullanılmamış izin karşılığı",
        "dipnot": "",
        "degerler": [
            {
                "yil": "2025",
                "deger": "-1770797"
            },
            {
                "yil": "2024",
                "deger": "-303524"
            }
        ]
    },
    {
        "kalem": "Çalışanlara Sağlanan Faydalar (devamı) - Kullanılmamış izin karşılığı",
        "dipnot": "",
        "degerler": [
            {
                "yil": "2025",
                "de …
```

**LoRA** — hücre doğruluğu 1.00
```json
[{"kalem": "Dönem başı", "dipnot": null, "degerler": [11805614, 6236459]}, {"kalem": "Dönem içinde ödenen", "dipnot": null, "degerler": [-1770797, -303524]}, {"kalem": "Dönem içindeki artış", "dipnot": null, "degerler": [12582597, 8819047]}, {"kalem": "Parasal Kayıp / Kazanç", "dipnot": null, "degerler": [-3970252, -2946368]}, {"kalem": "Dönem sonu", "dipnot": null, "degerler": [18647162, 11805614]}, {"kalem": "Dönem başı", "dipnot": null, "degerler": [63181919, 26598323]}, {"kalem": "Dönem içinde ödenen", "dipnot": null, "degerler": [-35897750, -36023787]}, {"kalem": "Dönem içindeki artış", "dipnot": null, "degerler": [53201875, 86937993]}, {"kalem": "Parasal Kayıp / Kazanç", "dipnot": null, "degerler": [-19917852, -14330610]}, {"kalem": "Dönem sonu", "dipnot": null, "degerler": [60568192, 63181919]}]
```
