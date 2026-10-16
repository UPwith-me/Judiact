#!/usr/bin/env bash
set -euo pipefail
ROOT=${DIAGSAFE_ROOT:-/disk/u20110017/diagsafe_v1}
CODE=${DIAGSAFE_CODE:-$ROOT/training_code/grounded_v6_compact_balanced_20261016}
PY="$ROOT/work/answer_generation_env/bin/python"
DATA="$ROOT/releases/visual_refusal_fill_20261015"
MODEL="$ROOT/raw/models/Qwen3-VL-4B-Instruct"
RUN=${DIAGSAFE_V6_RUN:-$ROOT/runs/grounded_v6_compact_balanced_20261016/seed42/formal}
export PYTHONPATH="$CODE:$ROOT/training_code/v1_20261007"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "$RUN/logs"

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v6 --model "$MODEL" --release "$DATA" --output "$RUN" \
  --epochs 3 --effective-roots 64 --record-batch-size 8 --native-batch-size 16 \
  --rollout-batch-size 8 --max-batch-tokens 16384 --native-checkpoint-layer-stride 2 \
  --fact-dispatch balanced --recovery-save-steps 100 \
  > "$RUN/logs/train.log" 2>&1

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.evaluate_v4 --protocol v6 --model "$MODEL" --adapter "$RUN/selected" \
  --release "$DATA" --output "$RUN/dev_eval" --initial-batch-size 8 --max-batch-size 8 \
  > "$RUN/logs/dev_eval.log" 2>&1
