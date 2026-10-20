"""Her LoRA adaptoru icin Hugging Face model karti (README.md) uretir; sayilar results/final/final_summary.csv'den gelir.

Kullanim:
  python make_model_cards.py --summary results/final/final_summary.csv --out hf_cards            # sadece dosyalari uretir
  HF_WRITE_TOKEN=... python make_model_cards.py --summary ... --out hf_cards --upload              # Hub'a da yukler
"""
import argparse
import csv
import os
from pathlib import Path

LEVELS = "ABCD"
LEVEL_NAME = {"A": "A (temiz)", "B": "B (hafif)", "C": "C (orta)", "D": "D (ağır)"}

MODELS = {
    "qwen3vl_8b": dict(repo="furkancmc/Qwen3-VL-8B-KAP-TR-LoRA", base="Qwen/Qwen3-VL-8B-Instruct", license="apache-2.0",
                       name="Qwen3-VL-8B-Instruct", train_min=90, train_loss=0.054, eval_loss=0.0163, batch="4 × 8"),
    "qwen25vl_7b": dict(repo="furkancmc/Qwen2.5-VL-7B-KAP-TR-LoRA", base="Qwen/Qwen2.5-VL-7B-Instruct", license="apache-2.0",
                        name="Qwen2.5-VL-7B-Instruct", train_min=152, train_loss=0.092, eval_loss=0.0201, batch="4 × 8"),
    "internvl35_8b": dict(repo="furkancmc/InternVL3.5-8B-KAP-TR-LoRA", base="OpenGVLab/InternVL3_5-8B-HF", license="apache-2.0",
                          name="InternVL3.5-8B (HF)", train_min=151, train_loss=0.142, eval_loss=0.0509, batch="4 × 8"),
    "gemma3_12b": dict(repo="furkancmc/Gemma-3-12B-KAP-TR-LoRA", base="google/gemma-3-12b-it", license="gemma",
                       name="Gemma 3 12B-it", train_min=128, train_loss=0.300, eval_loss=0.3494, batch="2 × 16"),
    "llama32_11b": dict(repo="furkancmc/Llama-3.2-11B-Vision-KAP-TR-LoRA", base="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        license="llama3.2", name="Llama 3.2 11B Vision-Instruct", train_min=134, train_loss=0.145, eval_loss=0.0581, batch="2 × 16"),
}

LICENSE_NOTE = {
    "apache-2.0": "Taban model Apache-2.0 lisanslıdır; kullanmadan önce taban modelin sayfasındaki güncel lisansı doğrulayın.",
    "gemma": "Bu adaptör **Gemma** modelinden türetilmiştir ve [Gemma Kullanım Şartları](https://ai.google.dev/gemma/terms) ile "
             "Gemma Yasaklı Kullanım Politikası'na tabidir. Dağıtırken bu şartları iletmeniz gerekir.",
    "llama3.2": "**Built with Llama.** Bu adaptör Llama 3.2 modelinden türetilmiştir ve [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) "
                "ile Kabul Edilebilir Kullanım Politikası'na tabidir. Lisans, görüntü destekli Llama 3.2 modellerinin AB'de yerleşik kişi/şirketlerce kullanımını kısıtlar; "
                "kullanmadan önce lisansı okuyun.",
}


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["subset"], r["model"], r["tag"], r["level"], r["task"], r["metric"])] = r
    return rows


def cell(rows, subset, model, tag, level, task, metric, digits=3, ci=True):
    r = rows.get((subset, model, tag, level, task, metric))
    if not r:
        return "–"
    est = float(r["estimate"])
    if not ci:
        return f"{est:.{digits}f}"
    return f"{est:.{digits}f} [{float(r['ci_lo']):.{digits}f}, {float(r['ci_hi']):.{digits}f}]"


def card(key, rows):
    m = MODELS[key]
    subset = "common" if key == "llama32_11b" else "full"
    def n_of(task, level="A"):
        r = rows.get((subset, key, "lora", level, task, "cer" if task == "transcribe" else "cell_exact"))
        return r["n"] if r else "?"
    lines = [
        "---",
        "language: tr",
        f"license: {m['license']}",
        "library_name: peft",
        f"base_model: {m['base']}",
        "pipeline_tag: image-text-to-text",
        "tags: [lora, peft, vision-language, ocr, turkish, finance, kap, document-understanding, table-extraction]",
        "---",
        "",
        f"# {m['repo'].split('/')[1]}",
        "",
        f"**{m['name']}** için, **Türkçe KAP (Kamuyu Aydınlatma Platformu) belge sayfalarından** metin ve finansal tablo okuma amacıyla "
        "eğitilmiş bir **LoRA adaptörüdür** (taban ağırlıklar bu depoda yoktur, ayrıca indirilir).",
        "",
        "> Bu bir araştırma/öğrenme projesidir; üretim kullanımı için uygun olduğu iddia edilmez. Sonuçlar yapay bozulmalı test sayfalarındandır "
        "(gerçek taranmış belge testi yok). Ayrıntılar: [GitHub deposu](https://github.com/furkancmc/finance-vl-fine-tune).",
        "",
        "## Ne yapar?",
        "Bir sayfa **görüntüsü** ve iki istemden biriyle çalışır:",
        "1. *Transkripsiyon:* `Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz.` → sayfa metni",
        "2. *Tablo:* `Bu sayfadaki finansal tabloyu JSON listesi olarak çıkar. Her satır için kalem, dipnot ve degerler alanlarını yaz; sayıları rakam olarak yaz (binlik ayracı noktasını kaldır, parantezli değerler negatiftir).` → `{\"kalem\", \"dipnot\", \"degerler\"}` satırları (JSON)",
        "",
        "## Sonuçlar",
        "",
        f"Test sayfaları eğitimde hiç görülmemiş **şirketlerden**dir (şirket bazlı bölme). Aynı sayfalar dört bozulma seviyesinde ölçüldü "
        f"(A bozulmamış → D ağır; yapay bozulma, bkz. GitHub `docs/METHODOLOGY.md`). Köşeli parantez: %95 bootstrap güven aralığı.",
        f"n = {n_of('transcribe')} transkripsiyon, {n_of('table')} tablo sayfası / seviye"
        + (" (**yalnızca ilk 100 test örneği**, HF motoruyla; diğer modellerle kıyas için GitHub'daki ortak alt küme tablosuna bakın)" if key == "llama32_11b" else "") + ".",
        "",
        "**Transkripsiyon, medyan CER** (karakter hata oranı, düşük iyi)",
        "",
        "| Seviye | Taban model | + LoRA |",
        "|---|---|---|",
    ]
    for lv in LEVELS:
        lines.append(f"| {LEVEL_NAME[lv]} | {cell(rows, subset, key, 'base', lv, 'transcribe', 'cer')} | **{cell(rows, subset, key, 'lora', lv, 'transcribe', 'cer')}** |")
    lines += ["", "**Tablo çıkarımı, hücre doğruluğu** (yüksek iyi) ve geçerli JSON oranı", "",
              "| Seviye | Hücre doğruluğu: taban | + LoRA | Geçerli JSON: taban | + LoRA |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        lines.append(f"| {LEVEL_NAME[lv]} | {cell(rows, subset, key, 'base', lv, 'table', 'cell_exact')} | **{cell(rows, subset, key, 'lora', lv, 'table', 'cell_exact')}** | "
                     f"{cell(rows, subset, key, 'base', lv, 'table', 'valid', 2)} | {cell(rows, subset, key, 'lora', lv, 'table', 'valid', 2)} |")
    lines += [
        "",
        "**Nasıl okunmalı:**",
        "- Taban modelin hatasının bir kısmı **biçim** hatasıdır (ör. düz metin yerine markdown tablosu yazması, JSON şemasını üretememesi); "
        "LoRA'nın katkısının bir kısmı bu biçimi öğrenmektir, tamamı \"daha iyi okuma\" değildir.",
        "- KAP sayfalarında şablon (tekrarlayan) metin çoktur; yüksek doğruluğun bir kısmı bu şablonları öğrenmekten gelebilir.",
        "- Tek eğitim koşusu (tohum 42). Güven aralıkları test örneklemesini gösterir, eğitim rastgeleliğini göstermez.",
        "- Tablo etiketleri PDF metin katmanından **otomatik** çıkarıldı, gürültülü olabilir; satır sayısı eşleşmesi bu yüzden düşüktür.",
        "",
        "## Eğitim",
        "",
        "| | |",
        "|---|---|",
        f"| Taban model | [`{m['base']}`](https://huggingface.co/{m['base']}) |",
        "| Yöntem | LoRA (r=32, alpha=64, dropout 0), 16-bit bf16, yalnızca dil katmanları (görüntü kodlayıcı dondurulmuş) |",
        "| Veri | 2500 örnek: 2000 transkripsiyon + 500 tablo; sayfalar A/B/C/D bozulma seviyelerinde ≈ eşit dağılım; 261 şirket |",
        f"| Süre / adım | 2 epoch, 158 adım, etkin batch 32 ({m['batch']}), lr 2e-4 cosine, A100 80 GB'ta {m['train_min']} dk |",
        f"| Son kayıp | eğitim {m['train_loss']}, doğrulama {m['eval_loss']} (modeller arası karşılaştırılamaz: tokenizer farklı) |",
        "| Araçlar | Unsloth + Transformers 5.17 + PEFT 0.21 + TRL 1.15, torch 2.11 |",
        "",
        "## Kullanım",
        "",
        "```python",
        "import torch",
        "from PIL import Image",
        "from peft import PeftModel",
        "from transformers import AutoModelForImageTextToText, AutoProcessor",
        "",
        f"BASE = \"{m['base']}\"",
        f"ADAPTER = \"{m['repo']}\"",
        "model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=\"auto\")",
        "model = PeftModel.from_pretrained(model, ADAPTER).eval()",
        "processor = AutoProcessor.from_pretrained(BASE)",
        "",
        "PROMPT = \"Bu sayfadaki tüm metni okuma sırasına uygun şekilde, olduğu gibi yaz.\"",
        "image = Image.open(\"sayfa.png\").convert(\"RGB\")",
        "messages = [{\"role\": \"user\", \"content\": [{\"type\": \"image\"}, {\"type\": \"text\", \"text\": PROMPT}]}]",
        "text = processor.apply_chat_template(messages, add_generation_prompt=True)",
        "inputs = processor(text=text, images=[image], return_tensors=\"pt\").to(model.device)",
        "with torch.inference_mode():",
        "    out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)",
        "print(processor.batch_decode(out[:, inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)[0])",
        "```",
        "",
        "Eğitimde görüntüler en çok 700 000 piksel olarak verildi; benzer çözünürlükte (≈ 130 DPI A4) sayfalarla en iyi sonuç beklenir.",
        "",
        "## Sınırlamalar",
        "- Yalnızca KAP bildirim sayfalarında (Ocak–Eylül 2026) eğitildi/ölçüldü; başka belge türlerine genellemesi ölçülmedi.",
        "- Ağır bozulmada (D) hata hâlâ yüksektir; bazı sayfalarda çıktı tekrar döngüsüne girip token sınırına kadar uzayabilir.",
        "- Gerçek tarama/fotoğraf testi yoktur; bozulmalar yapaydır.",
        "- Tablo satır sayısını birebir yakalama zayıftır (bkz. GitHub sonuçları).",
        "",
        "## Lisans",
        LICENSE_NOTE[m["license"]],
        "",
        "## Atıf",
        "```",
        "@misc{kap_tr_vl_lora_2026,",
        "  title  = {KAP Turkish Document VL LoRA Adapters},",
        "  author = {furkancmc},",
        "  year   = {2026},",
        "  url    = {https://github.com/furkancmc/finance-vl-fine-tune}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="hf_cards")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--only", default="", help="virgullu model anahtarlari (bos = hepsi)")
    ap.add_argument("--public", action="store_true", help="yukledikten sonra depoyu public yap")
    args = ap.parse_args()
    rows = load(args.summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = None
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    only = [k for k in args.only.split(",") if k]
    for key, m in MODELS.items():
        if only and key not in only:
            continue
        text = card(key, rows)
        (out / f"{key}.md").write_text(text, encoding="utf-8")
        if api:
            api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo="README.md", repo_id=m["repo"], repo_type="model",
                            commit_message="Model card (sonuclar, egitim, kullanim)")
            print("yuklendi:", m["repo"], flush=True)
            if args.public:
                api.update_repo_settings(repo_id=m["repo"], private=False, repo_type="model")
                print("public:", m["repo"], flush=True)
    print("kartlar:", out)


if __name__ == "__main__":
    main()
