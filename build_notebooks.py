"""Colab notebook uretici: notebooks/train_<key>.ipynb ve notebooks/eval_<key>.ipynb."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf

from model_zoo import READY, get as get_model

ROOT = Path(__file__).parent
NB_DIR = ROOT / "notebooks"


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(text: str):
    return nbf.v4.new_code_cell(text)


def _new_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"gpuType": "A100"},
    }
    return nb


def build_train(key: str) -> nbf.NotebookNode:
    info = get_model(key)
    hf_id = info["hf_id"]

    cells = []

    cells.append(_md(
        f"# Egitim notebook'u: `{key}`\n\n"
        f"- Model: `{hf_id}` (family: `{info.get('family')}`)\n"
        "- Ortam: Colab A100 80 GB, bf16, TF32 acik.\n"
        "- Tahmini sure: ~3-6 saat (veri boyutuna ve epoch sayisina gore).\n"
        "- Ciktilar: Drive'da `kap_lora/" + key + "/` altina yazilir (dogrudan Drive'a).\n"
        "- Sonunda `merge_lora.py` ile adapter birlestirilir.\n"
    ))

    cells.append(_code(
        "# Ayarlar\n"
        f"MODEL_KEY = \"{key}\"\n"
        "BACKEND = \"unsloth\"\n"
        "LOAD_IN_4BIT = False\n"
        "BATCH = 8\n"
        "GRAD_ACCUM = 4\n"
        "EPOCHS = 3\n"
        "LR = 2e-4\n"
        "LORA_R = 32\n"
        "DRIVE_DIR = \"/content/drive/MyDrive/kap_lora\"\n"
        "DATASET_ZIP = f\"{DRIVE_DIR}/dataset.zip\"\n"
        "REPO_URL = \"github.com/furkancmc/finance-vl-fine-tune\"\n"
    ))

    cells.append(_code(
        "# Kurulum: Drive, paketler, HF token\n"
        "from google.colab import drive\n"
        "drive.mount(\"/content/drive\")\n"
        "!pip install -q unsloth bitsandbytes peft accelerate qwen-vl-utils rapidfuzz\n"
        "import os\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    tok = userdata.get(\"HF_TOKEN\")\n"
        "    if tok:\n"
        "        from huggingface_hub import login\n"
        "        login(tok)\n"
        "        print(\"HF login ok\")\n"
        "except Exception as e:\n"
        "    print(\"HF token yok/atlandi:\", e)\n"
    ))

    cells.append(_code(
        "# Repo cek: GH_TOKEN varsa private clone, yoksa Drive'dan kopyala\n"
        "import os, shutil, subprocess\n"
        "from pathlib import Path\n"
        "gh = None\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    gh = userdata.get(\"GH_TOKEN\")\n"
        "except Exception:\n"
        "    pass\n"
        "if os.path.exists(\"/content/repo\"):\n"
        "    shutil.rmtree(\"/content/repo\")\n"
        "if gh:\n"
        "    url = f\"https://{gh}@{REPO_URL}.git\"\n"
        "    subprocess.run([\"git\", \"clone\", url, \"/content/repo\"], check=True)\n"
        "else:\n"
        "    src = Path(DRIVE_DIR) / \"code\"\n"
        "    assert src.exists(), f\"GH_TOKEN yok ve {src} bulunamadi\"\n"
        "    shutil.copytree(src, \"/content/repo\")\n"
        "%cd /content/repo\n"
    ))

    cells.append(_code(
        "# Veri: dataset.zip ac ve train.jsonl satir sayisi\n"
        "import os, zipfile\n"
        "from pathlib import Path\n"
        "if not os.path.exists(\"/content/dataset\"):\n"
        "    assert os.path.exists(DATASET_ZIP), f\"zip yok: {DATASET_ZIP}\"\n"
        "    with zipfile.ZipFile(DATASET_ZIP) as z:\n"
        "        z.extractall(\"/content\")\n"
        "p = Path(\"/content/dataset/train.jsonl\")\n"
        "n = sum(1 for _ in p.open(\"r\", encoding=\"utf-8\"))\n"
        "print(\"train satir:\", n)\n"
    ))

    cells.append(_code(
        "# GPU kontrol\n"
        "!nvidia-smi\n"
        "import torch\n"
        "print(torch.cuda.get_device_name(0))\n"
        "print(\"bellek GB:\", torch.cuda.get_device_properties(0).total_memory / 1e9)\n"
        "import transformers\n"
        "from packaging.version import Version\n"
        "from model_zoo import get\n"
        "need = get(MODEL_KEY)[\"min_transformers\"]\n"
        "print(\"transformers\", transformers.__version__, \"gerekli >=\", need)\n"
        "assert Version(transformers.__version__) >= Version(need), \"transformers cok eski: !pip install -U transformers\"\n"
    ))

    cells.append(_md("## Smoke test (kisa, hizli dogrulama)"))

    cells.append(_code(
        "!python train_lora.py --model $MODEL_KEY --backend $BACKEND \\\n"
        "  --limit 32 --max-steps 3 --batch 4 --grad-accum 1 \\\n"
        "  --out /content/smoke --eval-steps 3 --save-steps 3 --max-eval 8\n"
    ))

    cells.append(_md("## Tam egitim (Drive'a yaz, --resume ile devam edilebilir)"))

    cells.append(_code(
        "FLAG_4BIT = \"--load-in-4bit\" if LOAD_IN_4BIT else \"\"\n"
        "!python train_lora.py --model $MODEL_KEY --backend $BACKEND \\\n"
        "  --data /content/dataset \\\n"
        "  --batch $BATCH --grad-accum $GRAD_ACCUM --epochs $EPOCHS --lr $LR \\\n"
        "  --lora-r $LORA_R $FLAG_4BIT \\\n"
        "  --out {DRIVE_DIR}/" + key + " --resume\n"
    ))

    cells.append(_md("## Kayip egrisi"))

    cells.append(_code(
        "import json\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "log = Path(f\"{DRIVE_DIR}/" + key + "/train_log.jsonl\")\n"
        "steps, losses, ev_steps, ev_losses = [], [], [], []\n"
        "for line in log.open(\"r\", encoding=\"utf-8\"):\n"
        "    r = json.loads(line)\n"
        "    if \"loss\" in r and r.get(\"step\") is not None:\n"
        "        steps.append(r[\"step\"]); losses.append(r[\"loss\"])\n"
        "    if \"eval_loss\" in r and r.get(\"step\") is not None:\n"
        "        ev_steps.append(r[\"step\"]); ev_losses.append(r[\"eval_loss\"])\n"
        "plt.figure(figsize=(8,4))\n"
        "plt.plot(steps, losses, label=\"train\")\n"
        "if ev_steps:\n"
        "    plt.plot(ev_steps, ev_losses, label=\"eval\")\n"
        "plt.xlabel(\"step\"); plt.ylabel(\"loss\"); plt.legend(); plt.grid(True)\n"
        "plt.show()\n"
    ))

    cells.append(_md("## LoRA birlestirme"))

    cells.append(_code(
        "!python merge_lora.py --model $MODEL_KEY \\\n"
        "  --adapter {DRIVE_DIR}/" + key + "/final \\\n"
        "  --out {DRIVE_DIR}/" + key + "/merged\n"
    ))

    cells.append(_md(
        "## Sonraki adim\n\n"
        "Degerlendirme icin **yeni bir runtime** acin ve `eval_" + key + ".ipynb` notebook'unu calistirin "
        "(vLLM unsloth ile ayni ortama kurulmaz).\n"
    ))

    return _new_nb(cells)


def build_eval(key: str) -> nbf.NotebookNode:
    info = get_model(key)
    hf_id = info["hf_id"]

    cells = []

    cells.append(_md(
        f"# Degerlendirme notebook'u: `{key}`\n\n"
        f"- Model: `{hf_id}`\n"
        "- **TEMIZ runtime** gerekir (unsloth ile ayni ortama KURULMAZ).\n"
        "- Motor: vLLM (sorun cikarsa `ENGINE=\"hf\"`).\n"
        "- Ciktilar: Drive'da `kap_lora/eval/" + key + "/` altina yazilir.\n"
    ))

    cells.append(_code(
        "# Ayarlar\n"
        f"MODEL_KEY = \"{key}\"\n"
        "DRIVE_DIR = \"/content/drive/MyDrive/kap_lora\"\n"
        "ENGINE = \"vllm\"\n"
        "MAX_NEW_TOKENS = 3072\n"
        "GPU_MEM = 0.90\n"
        "MAX_NUM_SEQS = 128\n"
        "DATASET_ZIP = f\"{DRIVE_DIR}/dataset.zip\"\n"
        "REPO_URL = \"github.com/furkancmc/finance-vl-fine-tune\"\n"
    ))

    cells.append(_code(
        "# Kurulum: Drive, paketler, HF token\n"
        "from google.colab import drive\n"
        "drive.mount(\"/content/drive\")\n"
        "!pip install -q vllm rapidfuzz qwen-vl-utils\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    tok = userdata.get(\"HF_TOKEN\")\n"
        "    if tok:\n"
        "        from huggingface_hub import login\n"
        "        login(tok)\n"
        "        print(\"HF login ok\")\n"
        "except Exception as e:\n"
        "    print(\"HF token yok/atlandi:\", e)\n"
    ))

    cells.append(_code(
        "# Repo cek\n"
        "import os, shutil, subprocess\n"
        "from pathlib import Path\n"
        "gh = None\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    gh = userdata.get(\"GH_TOKEN\")\n"
        "except Exception:\n"
        "    pass\n"
        "if os.path.exists(\"/content/repo\"):\n"
        "    shutil.rmtree(\"/content/repo\")\n"
        "if gh:\n"
        "    url = f\"https://{gh}@{REPO_URL}.git\"\n"
        "    subprocess.run([\"git\", \"clone\", url, \"/content/repo\"], check=True)\n"
        "else:\n"
        "    src = Path(DRIVE_DIR) / \"code\"\n"
        "    assert src.exists(), f\"GH_TOKEN yok ve {src} bulunamadi\"\n"
        "    shutil.copytree(src, \"/content/repo\")\n"
        "%cd /content/repo\n"
    ))

    cells.append(_code(
        "# Veri: dataset.zip ac\n"
        "import os, zipfile\n"
        "if not os.path.exists(\"/content/dataset\"):\n"
        "    assert os.path.exists(DATASET_ZIP), f\"zip yok: {DATASET_ZIP}\"\n"
        "    with zipfile.ZipFile(DATASET_ZIP) as z:\n"
        "        z.extractall(\"/content\")\n"
        "print(\"dataset hazir\")\n"
        "import transformers\n"
        "from packaging.version import Version\n"
        "from model_zoo import get\n"
        "need = get(MODEL_KEY)[\"min_transformers\"]\n"
        "print(\"transformers\", transformers.__version__, \"gerekli >=\", need)\n"
        "assert Version(transformers.__version__) >= Version(need), \"transformers cok eski: !pip install -U transformers\"\n"
    ))

    cells.append(_md("## Baz model degerlendirmesi"))

    cells.append(_code(
        "!python eval_vllm.py --model $MODEL_KEY --tag base \\\n"
        "  --data /content/dataset \\\n"
        "  --engine $ENGINE --max-new-tokens $MAX_NEW_TOKENS \\\n"
        "  --gpu-mem $GPU_MEM --max-num-seqs $MAX_NUM_SEQS \\\n"
        "  --out {DRIVE_DIR}/eval/" + key + "\n"
    ))

    cells.append(_md("## LoRA (birlestirilmis) degerlendirmesi"))

    cells.append(_code(
        "!python eval_vllm.py --model $MODEL_KEY --tag lora \\\n"
        "  --model-path {DRIVE_DIR}/" + key + "/merged \\\n"
        "  --data /content/dataset \\\n"
        "  --engine $ENGINE --max-new-tokens $MAX_NEW_TOKENS \\\n"
        "  --gpu-mem $GPU_MEM --max-num-seqs $MAX_NUM_SEQS \\\n"
        "  --out {DRIVE_DIR}/eval/" + key + "\n"
    ))

    cells.append(_md("## Rapor"))

    cells.append(_code(
        "!python report.py --eval-dir {DRIVE_DIR}/eval/" + key + "\n"
        "from IPython.display import Markdown, display\n"
        "from pathlib import Path\n"
        "p = Path(f\"{DRIVE_DIR}/eval/" + key + "/report.md\")\n"
        "if p.exists():\n"
        "    display(Markdown(p.read_text(encoding=\"utf-8\")))\n"
        "else:\n"
        "    print(\"report.md bulunamadi\")\n"
    ))

    cells.append(_md(
        "## Not\n\n"
        "vLLM sorun cikarirsa ayar hucresinde `ENGINE = \"hf\"` yapip yeniden calistirin.\n"
    ))

    return _new_nb(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(NB_DIR), help="notebook cikis klasoru")
    ap.add_argument("--keys", nargs="*", default=None, help="sadece bu anahtarlar")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(READY) if args.keys is None else [k for k in args.keys if k in READY]
    if not keys:
        print("Uretilecek model yok (READY bos veya filtre bos).", flush=True)
        return

    written = []
    for key in keys:
        tr = out / f"train_{key}.ipynb"
        ev = out / f"eval_{key}.ipynb"
        nbf.write(build_train(key), str(tr))
        nbf.write(build_eval(key), str(ev))
        written += [tr, ev]
        print(f"[ok] {tr.name}, {ev.name}", flush=True)

    print("\nUretilen dosyalar:", flush=True)
    for p in written:
        print(" -", p, flush=True)


if __name__ == "__main__":
    main()
