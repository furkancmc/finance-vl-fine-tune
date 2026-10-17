#!/bin/bash
# Colab: merge (varsa atlar) + her model icin lora ve base eval (vLLM venv) + rapor.
# On kosul: /content/repo (git clone), /content/dataset (dataset.zip acik), /content/vllm_env (setup_vllm_env.sh),
#           Drive baglı, adaptorler $D/<model>/final. Log: $D/eval_pipeline.log
# Kullanim: nohup bash colab/eval_all.sh > /content/eval_all_stdout.log 2>&1 &
D=/content/drive/MyDrive/kap_lora
LOG=$D/eval_pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
cd /content/repo
git pull -q

VPY=/content/vllm_env/bin/python
# ZORUNLU: venv'i activate etmeden mutlak yolla cagirdigimiz icin ninja PATH'te olmali; FlashInfer sampler JIT'i kapat.
export PATH=/content/vllm_env/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
# Egitimde 0.7M piksel kullanildi; adil karsilastirma icin eval de ayni.
COMMON="--data /content/dataset --engine vllm --max-new-tokens 3072 --max-pixels 700000 --gpu-mem 0.85 --max-num-seqs 128"

merge_if_needed() { # key
  if [ -d $D/$1/merged ]; then log "$1 merged zaten var"; return; fi
  python merge_lora.py --model $1 --adapter $D/$1/final --out $D/$1/merged >> $D/$1/merge.log 2>&1 \
    && log "$1 merge TAMAM" || log "$1 merge BASARISIZ ($1/merge.log)"
}

run_eval() { # key tag [model-path]
  KEY=$1; TAG=$2; MP=$3
  log "EVAL basladi: $KEY / $TAG"
  if [ -n "$MP" ]; then ARGS="--model-path $MP"; else ARGS=""; fi
  $VPY eval_vllm.py --model $KEY --tag $TAG $ARGS $COMMON --out $D/eval/$KEY > $D/eval/${KEY}_${TAG}.log 2>&1 \
    && log "EVAL TAMAM: $KEY / $TAG" || log "EVAL BASARISIZ: $KEY / $TAG (eval/${KEY}_${TAG}.log)"
}

mkdir -p $D/eval
for K in qwen25vl_7b qwen3vl_8b; do merge_if_needed $K; done
for K in qwen25vl_7b qwen3vl_8b; do run_eval $K lora $D/$K/merged; done
for K in qwen25vl_7b qwen3vl_8b; do run_eval $K base; done

# Not: eval tamamlanmadiysa report bos/yaniltici olur; log'taki BASARISIZ satirlarina bak.
python report.py --eval-dir $D/eval/qwen25vl_7b --eval-dir $D/eval/qwen3vl_8b --out $D/eval/report > $D/eval/report.log 2>&1
python report.py --eval-dir $D/eval/qwen25vl_7b --eval-dir $D/eval/qwen3vl_8b --out $D/eval/report_clean --clean --min-ref-chars 150 >> $D/eval/report.log 2>&1
log "EVAL PIPELINE BITTI (rapor: eval/report ve eval/report_clean)"
