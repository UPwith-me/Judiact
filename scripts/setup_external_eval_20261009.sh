#!/usr/bin/env bash
set -e
ROOT=${DIAGSAFE_ROOT:-/disk/u20110017/diagsafe_v1}
BASE="$ROOT/work/answer_generation_env/bin/python"
ENV="$ROOT/work/external_eval_env"
SHIELD_ENV="$ROOT/work/shieldstral_eval_env"
LLAMA_ENV="$ROOT/work/llamaguard4_eval_env"
INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# Reuse the installed CUDA torch through system-site-packages; do not change training.
"$BASE" -m venv --system-site-packages "$ENV"
PARENT_SITE=$("$BASE" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$ENV/bin/python" -c 'import sys,sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "training_cuda_packages.pth").write_text(sys.argv[1]+"\n")' "$PARENT_SITE"
"$ENV/bin/python" -m pip install --index-url "$INDEX" 'transformers==4.57.1' \
    'qwen-vl-utils>=0.0.14' 'qwen-omni-utils' 'jsonschema' 'scipy' 'scikit-learn' 'spacy>=3.8,<3.9' 'datasketch' \
    'sentencepiece' 'protobuf' 'cryptography>=41'
"$ENV/bin/python" -m pip install --no-deps --index-url "$INDEX" 'modelscope==1.40.1' 'modelscope-hub==0.4.5'
"$ENV/bin/python" -m pip install --no-deps -e "$ROOT/work/external_refs/SafeAtlas-VL"
"$ENV/bin/python" -m spacy download en_core_web_sm-3.8.0 --direct

# Shieldstral uses the newer Mistral tokenizer backend and has its own environment.
"$BASE" -m venv --system-site-packages "$SHIELD_ENV"
"$SHIELD_ENV/bin/python" -c 'import sys,sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "training_cuda_packages.pth").write_text(sys.argv[1]+"\n")' "$PARENT_SITE"
"$SHIELD_ENV/bin/python" -m pip install --index-url "$INDEX" \
    'transformers>=5' 'mistral-common>=1.11.5' 'sentencepiece'
"$SHIELD_ENV/bin/python" -m pip install --no-deps --index-url "$INDEX" 'opencv-python-headless==4.11.0.86'

# This stable HF release supports Guard's intentionally disabled chunked attention.
"$BASE" -m venv --system-site-packages "$LLAMA_ENV"
EXTERNAL_SITE=$("$ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$LLAMA_ENV/bin/python" -c 'import sys,sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "evaluation_packages.pth").write_text("\n".join(sys.argv[1:])+"\n")' "$PARENT_SITE" "$EXTERNAL_SITE"
"$LLAMA_ENV/bin/python" -m pip install --no-deps --index-url "$INDEX" 'tokenizers==0.21.4' 'transformers==4.52.4'
"$ENV/bin/python" -c 'import torch, transformers, ordinal_safety_vlm; print(torch.__version__, transformers.__version__)'
"$SHIELD_ENV/bin/python" -c 'from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend; import torch; print(torch.__version__)'
