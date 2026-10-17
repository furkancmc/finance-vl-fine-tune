# build_notebooks.py

`nbformat` ile Colab notebook'larini `notebooks/` klasorune uretir. Her notebook ayni sablondan, farkli `MODEL_KEY` ile. Sadece
`model_zoo.READY` icindeki modeller icin uret (`from model_zoo import READY`). Dosya adlari: `notebooks/train_<key>.ipynb`, `notebooks/eval_<key>.ipynb`.
Notebook metin ve basliklari TURKCE. Hucreler kisa ve calistirilabilir olsun; kod hucrelerinde Turkce karakter KULLANMA (yorumlar ASCII).

## Egitim notebook'u (train_<key>)
1. Markdown baslik: ne yapar, A100 80 GB, ~sure tahmini, ciktilar nereye yazilir.
2. Ayar hucresi: `MODEL_KEY`, `BACKEND="unsloth"`, `LOAD_IN_4BIT=False`, `BATCH=8`, `GRAD_ACCUM=4`, `EPOCHS=3`, `LR=2e-4`, `LORA_R=32`, `DRIVE_DIR="/content/drive/MyDrive/kap_lora"`,
   `DATASET_ZIP=f"{DRIVE_DIR}/dataset.zip"`, `REPO_URL="github.com/furkancmc/finance-vl-fine-tune"`.
3. Kurulum hucresi: Drive baglama (`from google.colab import drive; drive.mount("/content/drive")`); `!pip install -q unsloth bitsandbytes peft accelerate qwen-vl-utils rapidfuzz`
   (transformers'i unsloth'un getirdigi surumde birak, `!pip install -q -U "transformers>=4.57"` ekle); HF token: `from google.colab import userdata` ile `HF_TOKEN` varsa `huggingface_hub.login`.
4. Kod: GitHub private repoyu `git clone https://<GH_TOKEN>@{REPO_URL}.git /content/repo` ile cek (`userdata.get("GH_TOKEN")`); token yoksa
   `{DRIVE_DIR}/code` klasorunden `/content/repo`'ya kopyala. `%cd /content/repo`.
5. Veri: `/content/dataset` yoksa `DATASET_ZIP`'i `/content` altina ac (zip icindeki kok `dataset/`), `train.jsonl` satir sayisini yazdir.
6. GPU kontrolu: `!nvidia-smi`, `torch.cuda.get_device_name`, bellek.
7. Smoke test hucresi: `!python train_lora.py --model $MODEL_KEY --backend $BACKEND --limit 32 --max-steps 3 --batch 4 --grad-accum 1 --out /content/smoke --eval-steps 3 --save-steps 3 --max-eval 8`.
8. Tam egitim hucresi: `train_lora.py` tam argumanlarla, ciktisi `{DRIVE_DIR}/<key>/` altina (Drive'a dogrudan yaz, oturum kopsa kaybolmasin), `--resume` ile.
9. Kayip egrisi hucresi: `train_log.jsonl`'dan matplotlib ile loss ve eval_loss grafigi.
10. Birlestirme hucresi: `!python merge_lora.py --model $MODEL_KEY --adapter {DRIVE_DIR}/<key>/final --out {DRIVE_DIR}/<key>/merged`.
11. Son not: eval notebook'una (yeni bir runtime'da, vLLM icin) gecilmesini soyle.

## Degerlendirme notebook'u (eval_<key>)  -- TEMIZ runtime (unsloth ile ayni ortama KURULMAZ)
1. Baslik.
2. Ayar: `MODEL_KEY`, `DRIVE_DIR`, `ENGINE="vllm"`, `MAX_NEW_TOKENS=3072`, `GPU_MEM=0.90`, `MAX_NUM_SEQS=128`.
3. Kurulum: Drive, `!pip install -q vllm rapidfuzz qwen-vl-utils` ve gerekirse `-U transformers`; HF token; repo clone (egitim notebook'undaki gibi); dataset zip ac.
4. Baz model: `!python eval_vllm.py --model $MODEL_KEY --tag base --out {DRIVE_DIR}/eval/<key> ...`.
5. LoRA (birlestirilmis): `--model-path {DRIVE_DIR}/<key>/merged --tag lora`.
6. Rapor: `!python report.py --eval-dir {DRIVE_DIR}/eval/<key>` ve `report.md`'yi goster (`IPython.display.Markdown`).
7. Not: vLLM sorun cikarirsa `ENGINE="hf"` yap.

Notebook dosyalari gecerli nbformat v4 olsun; her hucreye `metadata.colab`/gpu ayari ekleme gerekmez ama `nb.metadata["accelerator"]="GPU"` ve
`nb.metadata["colab"]={"gpuType":"A100"}` koy. Sonda uretilen dosyalari listele.
