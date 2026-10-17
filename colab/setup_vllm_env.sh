#!/bin/bash
# vLLM'i unsloth'tan ayri bir venv'e kurar (ayni ortama kurulmaz). Colab'da calistirilir.
# Kullanim: bash colab/setup_vllm_env.sh   (nice ile arka planda calistirmak egitimi etkilemez)
set -x
pip install -q uv
uv venv /content/vllm_env --python 3.12 --seed
uv pip install --python /content/vllm_env/bin/python vllm rapidfuzz qwen-vl-utils pillow numpy
/content/vllm_env/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
echo BITTI
