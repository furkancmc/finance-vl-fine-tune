#!/bin/bash
# Colab: Qwen2.5-VL-7B egitimi bittikten sonra merge + Qwen3-VL-8B smoke + tam egitim.
# 2026-10-17'de bir kez kullanildi (tek seferlik zincir). Referans / yeniden kullanim icin repoda duruyor.
# Qwen2.5-VL-7B egitimi once elle, ayri baslatildi (asagidaki TRAIN_ARGS ile ayni ayarlar):
#   cd /content/repo && nohup python train_lora.py --model qwen25vl_7b --backend unsloth $TRAIN_ARGS --out $D/qwen25vl_7b --resume > $D/qwen25vl_7b/train_stdout.log 2>&1 &
# Dikkat: surec adini bulurken pgrep/pkill kalibi "python3 train_lora.py" ile eslesmeli ("python train_lora" eslesmez).
D=/content/drive/MyDrive/kap_lora
LOG=$D/pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
cd /content/repo

# ~1 saatlik hedef icin kisilmis ayarlar (tam veri/3 epoch ~5 saat surerdi):
TRAIN_ARGS="--data /content/dataset --limit 1000 --epochs 2 --batch 4 --grad-accum 8 --lr 2e-4 --lora-r 32 \
  --max-pixels 700000 --no-grad-ckpt --eval-steps 32 --save-steps 32 --max-eval 50"

log "Qwen2.5-VL-7B egitiminin bitmesi bekleniyor"
while pgrep -f "train_[l]ora.py --model qwen25vl_7b" > /dev/null; do sleep 30; done
[ -d $D/qwen25vl_7b/final ] || { log "HATA: qwen25vl_7b/final yok"; exit 1; }

log "merge qwen25vl_7b"
python merge_lora.py --model qwen25vl_7b --adapter $D/qwen25vl_7b/final --out $D/qwen25vl_7b/merged >> $D/qwen25vl_7b/merge.log 2>&1 \
  && log "merge TAMAM" || log "merge BASARISIZ"

log "Qwen3-VL-8B smoke test"
BACKEND=""
for B in unsloth hf; do
  if python train_lora.py --model qwen3vl_8b --backend $B --data /content/dataset --limit 32 --max-steps 3 \
       --batch 4 --grad-accum 1 --max-pixels 700000 --out /content/smoke_q3_$B --eval-steps 3 --save-steps 3 --max-eval 8 \
       > $D/qwen3vl_smoke_$B.log 2>&1; then BACKEND=$B; log "smoke TAMAM ($B)"; break
  else log "smoke BASARISIZ ($B)"; fi
done
[ -n "$BACKEND" ] || { log "Qwen3-VL icin calisan backend yok"; exit 1; }

mkdir -p $D/qwen3vl_8b
python train_lora.py --model qwen3vl_8b --backend $BACKEND $TRAIN_ARGS --out $D/qwen3vl_8b --resume > $D/qwen3vl_8b/train_stdout.log 2>&1
[ -d $D/qwen3vl_8b/final ] && log "Qwen3-VL-8B egitimi TAMAM" || log "Qwen3-VL-8B egitimi BASARISIZ"
