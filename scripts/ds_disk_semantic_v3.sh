#!/usr/bin/env bash
ROOT=/disk/u20110017/diagsafe_v1
CODE="$ROOT/training_code/semantic_v3_20261008"
PY="$ROOT/work/answer_generation_env/bin/python"
DATA="$ROOT/releases/v3_train_dev_20261008"
REVIEWS="$ROOT/runs/method_v1_20261007/seed42"
RUN="$ROOT/runs/semantic_v3_20261008/seed42"
export PYTHONPATH="$CODE:$ROOT/training_code/v1_20261007"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m causal_diagsafe.cli.train_v3 --model "$ROOT/raw/models/Qwen3-VL-4B-Instruct" \
    --release "$DATA" --reviews "$REVIEWS" --output "$RUN/v3" --variant v3 --epochs 3 --base-dev
V3_STATUS=$?

# This independent comparison starts even if the other branch fails visibly.
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m causal_diagsafe.cli.train_v3 --model "$ROOT/raw/models/Qwen3-VL-4B-Instruct" \
    --release "$DATA" --reviews "$REVIEWS" --output "$RUN/no_relation" --variant no_relation --epochs 3
SFT_STATUS=$?

for variant in v3 no_relation; do
    status="$V3_STATUS"
    if [ "$variant" = no_relation ]; then status="$SFT_STATUS"; fi
    if [ "$status" -eq 0 ]; then
        "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
            -m causal_diagsafe.cli.evaluate_public_v3 \
            --raw "$ROOT/raw/external_lock" --policy-deck "$ROOT/work/public_eval_v1_20261007/policy_deck.json" \
            --prepared "$ROOT/work/public_eval_v1_20261007" --model "$ROOT/raw/models/Qwen3-VL-4B-Instruct" \
            --adapter "$RUN/$variant/selected" --output "$RUN/$variant/public_eval"
    fi
done
printf 'Training exit codes: v3=%s no_relation=%s\n' "$V3_STATUS" "$SFT_STATUS"
