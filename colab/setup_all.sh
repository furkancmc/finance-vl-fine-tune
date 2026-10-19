#!/bin/bash
# Colab tek seferlik kurulum (tekrar calistirilabilir, olanlari atlar).
# On kosul: Drive mount edilmis, GH_TOKEN ve HF_TOKEN export edilmis, Drive'da kap_lora/dataset.zip (v2) var.
# Kullanim: bash colab/setup_all.sh   (repo yoksa clone eder)
D=/content/drive/MyDrive/kap_lora
LOG=$D/v2/setup.log
mkdir -p $D/v2
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }

[ -d /content/repo/.git ] || git clone -q https://x-access-token:${GH_TOKEN}@github.com/furkancmc/finance-vl-fine-tune /content/repo
cd /content/repo && git pull -q
log "repo hazir: $(git log --oneline -1)"

if [ ! -f /content/dataset/train.jsonl ]; then
  unzip -q -o $D/dataset.zip -d /content && log "dataset acildi" || { log "HATA: dataset.zip acilamadi"; exit 1; }
fi
log "dataset: $(wc -l < /content/dataset/train.jsonl) train, $(wc -l < /content/dataset/val.jsonl) val, $(wc -l < /content/dataset/test_A.jsonl) test_A"

python -c "import unsloth" 2>/dev/null || pip install -q unsloth bitsandbytes peft accelerate qwen-vl-utils rapidfuzz
log "egitim paketleri hazir"

[ -x /content/vllm_env/bin/python ] || bash /content/repo/colab/setup_vllm_env.sh >> $D/v2/setup_vllm.log 2>&1
log "vllm venv: $([ -x /content/vllm_env/bin/python ] && echo var || echo YOK)"

# OCR ortamlari: PaddleOCR-VL ve HunyuanOCR transformers>=5.13, DeepSeek-OCR transformers==4.46.3 ister (catisir -> ayri venv)
pip install -q uv
if [ ! -x /content/ocr_new/bin/python ]; then
  uv venv /content/ocr_new --python 3.12 --seed >> $D/v2/setup_ocr.log 2>&1
  uv pip install --python /content/ocr_new/bin/python torch torchvision "transformers>=5.13" accelerate pillow rapidfuzz sentencepiece einops >> $D/v2/setup_ocr.log 2>&1
fi
if [ ! -x /content/ocr_ds/bin/python ]; then
  uv venv /content/ocr_ds --python 3.12 --seed >> $D/v2/setup_ocr.log 2>&1
  uv pip install --python /content/ocr_ds/bin/python torch==2.6.0 torchvision "transformers==4.46.3" "tokenizers==0.20.3" accelerate pillow rapidfuzz einops addict easydict matplotlib >> $D/v2/setup_ocr.log 2>&1
  uv pip install --python /content/ocr_ds/bin/python flash-attn==2.7.3 --no-build-isolation >> $D/v2/setup_ocr.log 2>&1 || log "flash-attn kurulamadi (DeepSeek-OCR eager calisir)"
fi
log "ocr venvleri: new=$([ -x /content/ocr_new/bin/python ] && echo var || echo YOK) ds=$([ -x /content/ocr_ds/bin/python ] && echo var || echo YOK)"
log "KURULUM BITTI"
