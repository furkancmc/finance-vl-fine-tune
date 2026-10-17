# merge_lora.py
# LoRA adaptorunu baz modele birlestirir, vLLM icin tam model klasoru kaydeder.
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def dir_size_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_zoo anahtari")
    ap.add_argument("--adapter", required=True, help="LoRA adaptor klasoru")
    ap.add_argument("--out", required=True, help="birlesik model cikis klasoru")
    ap.add_argument("--force", action="store_true", help="cikis varsa yeniden yap")
    ap.add_argument("--dry-run", action="store_true", help="sadece plan yazdir")
    args = ap.parse_args()

    out = Path(args.out)
    adapter = Path(args.adapter)

    if not adapter.exists():
        print(f"[hata] adaptor yok: {adapter}", flush=True)
        sys.exit(1)

    if (out / "config.json").exists() and not args.force:
        print(f"[atla] cikis zaten var: {out} (--force ile yeniden yap)", flush=True)
        return

    # model_zoo'dan hf_id al
    sys.path.insert(0, str(ROOT))
    from model_zoo import get as get_model
    info = get_model(args.model)
    hf_id = info["hf_id"]
    print(f"[bilgi] model={args.model} hf_id={hf_id}", flush=True)
    print(f"[bilgi] adapter={adapter}", flush=True)
    print(f"[bilgi] out={out}", flush=True)

    if args.dry_run:
        print("[dry-run] birlestirme yapilmadi.", flush=True)
        return

    # agir importlar burada
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        device_map = {"": 0}
        print("[bilgi] cuda kullanilacak", flush=True)
    else:
        device_map = "cpu"
        print("[uyari] cuda yok, cpu uzerinde birlestirilecek", flush=True)

    print("[bilgi] baz model yukleniyor...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    print("[bilgi] adaptor yukleniyor...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))

    print("[bilgi] merge_and_unload...", flush=True)
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    print(f"[bilgi] model kaydediliyor: {out}", flush=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")

    # processor: adaptor klasorunde varsa onu, yoksa hf_id
    proc_src = str(adapter) if (adapter / "preprocessor_config.json").exists() or (adapter / "processor_config.json").exists() else hf_id
    print(f"[bilgi] processor kaynagi: {proc_src}", flush=True)
    processor = AutoProcessor.from_pretrained(proc_src)
    processor.save_pretrained(str(out))

    # ozet
    size = dir_size_bytes(out)
    print(f"[bitti] klasor boyutu: {human(size)} ({size} bayt)", flush=True)
    print("[bitti] dosyalar:", flush=True)
    for f in sorted(out.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out)
            print(f"  {rel}  ({human(f.stat().st_size)})", flush=True)


if __name__ == "__main__":
    main()
