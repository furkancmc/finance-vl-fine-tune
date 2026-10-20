#!/bin/bash
# Colab gece zinciri: her model icin smoke -> egitim -> merge -> eval(lora) -> eval(base); sonra OCR baz cizgileri; sonra rapor.
# Her adim korumali: bir model/adim basarisiz olursa log'a yazilir, zincir sonrakiyle devam eder.
# Yeniden calistirilabilir: biten adimlar (DONE isareti, final/, preds) atlanir.
# On kosul: bash colab/setup_all.sh bitmis olmali.
# Kullanim: nohup bash /content/repo/colab/run_all.sh > /content/run_all_stdout.log 2>&1 &
# Sonuclar Drive'da: kap_lora/v2/{<model>/, eval/<model>/, eval/report*, STATUS.txt, pipeline.log}
D=/content/drive/MyDrive/kap_lora
R=$D/v2
LOG=$R/pipeline.log
STATUS=$R/STATUS.txt
mkdir -p $R/eval
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
status() { echo "[$(date '+%F %T')] $*" >> $STATUS; log "$*"; }
cd /content/repo && git pull -q

# venv PATH'in SONUNA: vLLM'in ninja'si bulunsun ama `python` sistem python'u kalsin (unsloth/peft orada kurulu)
export PATH=$PATH:/content/vllm_env/bin
export VLLM_USE_FLASHINFER_SAMPLER=0
VPY=/content/vllm_env/bin/python
MODELS=${MODELS:-"qwen3vl_8b qwen25vl_7b internvl35_8b gemma3_12b llama32_11b"}
OCR_MODELS=${OCR_MODELS:-"paddleocr_vl hunyuanocr deepseekocr"}
OCR_LIMIT=${OCR_LIMIT:-150}   # OCR baz cizgisi: her seviyede ilk N transkripsiyon sayfasi (hiz icin)

# 2500 ornek x 2 epoch = 156 adim (etkin batch 32)
train_args() { # key
  case $1 in
    gemma3_12b|llama32_11b) echo "--batch 2 --grad-accum 16" ;;   # 11-12B: bellek icin kucuk batch + grad ckpt
    *) echo "--batch 4 --grad-accum 8 --no-grad-ckpt" ;;
  esac
}
COMMON_TRAIN="--data /content/dataset --epochs 2 --lr 2e-4 --lora-r 32 --max-pixels 700000 --eval-steps 39 --save-steps 39 --max-eval 100"
COMMON_EVAL="--data /content/dataset --max-new-tokens 3072 --max-pixels 700000 --gpu-mem 0.85 --max-num-seqs 128"

do_model() { # key
  K=$1
  [ -f $R/$K/DONE ] && { log "$K zaten tamam"; return; }
  mkdir -p $R/$K $R/eval/$K
  HFID=$(python -c "from model_zoo import get; print(get('$K')['hf_id'])")
  python -c "from huggingface_hub import hf_hub_download as h; h('$HFID','config.json')" >/dev/null 2>&1 \
    || { status "$K ATLANDI: HF erisimi yok (lisans kabul / HF_TOKEN?)"; return; }

  if [ ! -d $R/$K/final ]; then
    BACKEND=""
    for B in unsloth hf; do
      if python train_lora.py --model $K --backend $B --data /content/dataset --limit 32 --max-steps 3 --batch 2 --grad-accum 1 \
           --max-pixels 700000 --out /content/smoke_${K}_$B --eval-steps 3 --save-steps 3 --max-eval 8 > $R/$K/smoke_$B.log 2>&1; then
        BACKEND=$B; log "$K smoke TAMAM ($B)"; break
      else log "$K smoke BASARISIZ ($B)"; fi
    done
    [ -n "$BACKEND" ] || { status "$K ATLANDI: smoke test hicbir backend'de gecmedi ($R/$K/smoke_*.log)"; return; }
    python train_lora.py --model $K --backend $BACKEND $COMMON_TRAIN $(train_args $K) --out $R/$K --resume > $R/$K/train_stdout.log 2>&1
    [ -d $R/$K/final ] || { status "$K ATLANDI: egitim basarisiz ($R/$K/train_stdout.log)"; return; }
    log "$K egitim TAMAM"
  fi

  python merge_lora.py --model $K --adapter $R/$K/final --out $R/$K/merged >> $R/$K/merge.log 2>&1 \
    || { status "$K ATLANDI: merge basarisiz ($R/$K/merge.log)"; return; }

  for MODE in lora base; do
    MP=""; [ $MODE = lora ] && MP="--model-path $R/$K/merged"
    OK=0
    for ENG in vllm hf; do
      EPY=$VPY; [ $ENG = hf ] && EPY=python   # HF motoru sistem python'unda (vllm venv'inde accelerate yok)
      $EPY eval_vllm.py --model $K --tag $MODE $MP --engine $ENG $COMMON_EVAL --out $R/eval/$K > $R/eval/${K}_${MODE}_$ENG.log 2>&1 && { OK=1; break; }
      log "$K eval $MODE $ENG BASARISIZ, siradaki motor"
    done
    [ $OK = 1 ] && log "$K eval $MODE TAMAM" || status "$K eval $MODE BASARISIZ"
  done
  touch $R/$K/DONE
  status "$K TAMAM (egitim+eval)"
}

for K in $MODELS; do do_model $K; done

# OCR baz cizgileri (fine-tune yok), sadece transkripsiyon, her seviyede ilk $OCR_LIMIT sayfa
for O in $OCR_MODELS; do
  mkdir -p $R/eval/$O
  PY=/content/ocr_new/bin/python; [ $O = deepseekocr ] && PY=/content/ocr_ds/bin/python
  $PY ocr_eval.py --model $O --data /content/dataset --out $R/eval/$O --limit $OCR_LIMIT --max-pixels 1500000 --batch 8 > $R/eval/$O.log 2>&1 \
    && status "OCR $O TAMAM" || status "OCR $O BASARISIZ ($R/eval/$O.log)"
done

# Rapor: preds'i olan tum model klasorleri
EVALS=""; for d in $R/eval/*/; do ls $d/preds_*_test_A.jsonl >/dev/null 2>&1 && EVALS="$EVALS --eval-dir $d"; done
python report.py $EVALS --out $R/eval/report > $R/eval/report.log 2>&1
python report.py $EVALS --out $R/eval/report_clean --clean --min-ref-chars 150 >> $R/eval/report.log 2>&1
status "ZINCIR BITTI (rapor: v2/eval/report ve report_clean)"
