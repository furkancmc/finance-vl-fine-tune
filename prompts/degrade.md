# degrade.py

Dijital sayfa goruntusunu taranmis/fotokopi belge gibi bozan kutuphane + kucuk demo CLI.

## Fonksiyon
`degrade(img: PIL.Image.Image, rng: random.Random) -> tuple[PIL.Image.Image, dict]`
- Girdi RGB. Cikti RGB, ayni boyut (rotate sonrasi kirpma yok, beyaz dolgu).
- Donen dict uygulanan parametreleri icerir (JSON'a yazilabilir, ornek: angle, blur, scale, noise, jpeg_q ...).
- Tum rastgelelik `rng`'den gelsin; numpy gurultusu icin `np.random.default_rng(rng.randrange(2**32))`.

## Adimlar (bu sirayla, her biri olasiliklidir)
1. Gri tonlama (p=0.5) -> RGB'ye geri don.
2. Dusuk cozunurluk: boyutu `s = uniform(0.5, 1.0)` ile kucult, sonra orijinal boyuta LANCZOS/BILINEAR ile buyut (p=0.7).
3. Hafif egiklik: aci `uniform(-2.5, 2.5)` derece, BICUBIC, fillcolor beyaz (p=0.8).
4. Gaussian blur yaricap `uniform(0.3, 1.6)` (p=0.7).
5. Kontrast `uniform(0.7, 1.2)`, parlaklik `uniform(0.85, 1.1)` (ImageEnhance, p=0.8).
6. Golge gradyani: rastgele yonde (sol/sag/ust/alt) kenardan merkeze dogru en cok %25 koyulasma (p=0.3).
7. Gaussian gurultu sigma `uniform(2, 18)` (p=0.7), degerleri 0-255'e kirp.
8. JPEG sikistirma gidis-donus, kalite `randint(35, 85)`, bellekte io.BytesIO ile (p=0.9).
En az bir bozulma uygulanmasini garanti et (hicbiri secilmediyse JPEG uygula).

## CLI
`python degrade.py --demo`: `data/pages.jsonl` varsa ilk 3 sayfayi, yoksa data/pages altindan ilk 3 png'yi al,
`data/degrade_demo/` altina `<ad>_clean.png` ve `<ad>_degraded.png` olarak kaydet, parametreleri yazdir.
