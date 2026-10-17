# Ortak kurallar (tum scriptler icin)

Proje: Turkce KAP (Kamuyu Aydinlatma Platformu) PDF'lerinden bir vision-language modeli (Qwen2.5-VL) LoRA ile
fine-tune etmek. Girdi: sayfa goruntusu. Hedef: sayfadaki Turkce metni (transkripsiyon) ve finansal tabloyu (JSON) uretmek.
Referans metin PDF'in kendi metin katmanindan gelir (pymupdf).

## Teknik kurallar
- Python 3.11+. Kutuphaneler: stdlib, `pymupdf` (import pymupdf), `PIL` (Pillow), `numpy`, `rapidfuzz`. Baska paket kullanma
  (Colab scriptleri haric, orada torch/transformers/peft kullanilir).
- Proje koku = scriptin bulundugu klasor: `ROOT = Path(__file__).parent`, veri `ROOT / "data"`.
- Tum dosya okuma/yazma `encoding="utf-8"`. JSONL satirlari `json.dumps(obj, ensure_ascii=False)`.
- Turkce metni `unicodedata.normalize("NFC", ...)` ile normalize et.
- Rastgelelik her zaman seed'li (SEED = 42), sonuc tekrar uretilebilir olmali.
- Scriptler yeniden calistirilabilir: var olan ciktiyi atla / uzerine guvenle yaz.
- `argparse` ile gerekli secenekler; `if __name__ == "__main__":` blogu. Ilerlemeyi `print(..., flush=True)` ile bas.
- Yorumlar kisa ve Turkce (ASCII, Turkce karakter kullanma kod yorumlarinda). Gereksiz soyutlama yok, sade kod.
- Sadece tek bir calisan python kod blogu dondur, aciklama yazma.

## Veri duzeni (mevcut)
- `data/metadata.csv` (utf-8-sig): sutunlar `disclosureIndex,title,type,company,stockCode,publishDate,attachmentCount,n_attachments,status`.
  `type` kota turu (ornek: "Finansal Rapor", "Faaliyet Raporu", "Özel Durum Açıklaması", "Kar Payı Dağıtım İşlemlerine İlişkin Bildirim" ...).
- `data/docs/<disclosureIndex>/main.pdf`, `attachment_1.pdf` ... `attachment_5.pdf` (bazilari yok olabilir).
- `data/inventory.csv`: PDF basina `doc_id,file,n_pages,n_pages_with_text,n_chars,...`.

## Uretilecek veri duzeni
- `data/pages/<doc_id>/<stem>_p<N>.png`: temiz sayfa goruntusu (N 1'den baslar, stem = "main" ya da "attachment_1").
- `data/pages.jsonl`: sayfa basina bir satir:
  `{"id": "<doc_id>_<stem>_p<N>", "doc_id": str, "file": "main.pdf", "page": N, "type": str, "company": str, "stock_code": str,
    "image": "pages/<doc_id>/<stem>_p<N>.png" (data/ altina gore), "text": str (referans metin), "n_chars": int,
    "width": int, "height": int, "table": null | [ {"kalem": str, "dipnot": str|null, "degerler": [sayi,...]} , ... ]}`
- `data/dataset/{train,val,test_clean,test_degraded}.jsonl` ve `data/dataset/images/*.jpg`:
  `{"id": str, "task": "transcribe"|"table", "split": str, "doc_id": str, "type": str, "degraded": bool,
    "image": "images/<ad>.jpg" (dataset/ klasorune gore),
    "messages": [{"role":"user","content":[{"type":"image"},{"type":"text","text": PROMPT}]},
                 {"role":"assistant","content":[{"type":"text","text": ANSWER}]}],
    "ref": ANSWER}`
