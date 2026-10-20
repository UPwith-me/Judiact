#!/usr/bin/env bash
set -euo pipefail

ROOT=${DIAGSAFE_ROOT:?Set DIAGSAFE_ROOT to the data and run directory}
CODE=${DIAGSAFE_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RELEASE=${DIAGSAFE_RELEASE:-$ROOT/releases/visual_refusal_fill_20261015}
MODEL=${DIAGSAFE_MODEL:-$ROOT/raw/models/Qwen3-VL-4B-Instruct}
TEACHER=${DIAGSAFE_TEACHER:-$ROOT/raw/external_models/multimodal_judge32b}
BANK=${DIAGSAFE_BANK:-$ROOT/work/v7_latent_bank_20261020}
RUN=${DIAGSAFE_RUN:-$ROOT/runs/latent_v7_20261020/seed42}
PY=${DIAGSAFE_PYTHON:-$ROOT/work/answer_generation_env/bin/python}
VPY=${DIAGSAFE_VLLM_PYTHON:-$ROOT/work/vllm_eval_env/bin/python}

export PYTHONPATH="$CODE"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# vLLM custom all-reduce uses CUDA IPC handles, incompatible with expandable segments here.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTHONUNBUFFERED=1
mkdir -p "$BANK" "$RUN"

printf '%s bank generation: two unquantized TP2 teacher replicas\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0,1 "$VPY" -m causal_diagsafe.cli.build_v7_latent_bank \
  --model "$TEACHER" --release "$RELEASE" --output "$BANK" \
  --rank 0 --world-size 2 --tensor-parallel-size 2 --memory 0.86 --window 32 \
  > "$RUN/bank.rank0.log" 2>&1 &
teacher0=$!
CUDA_VISIBLE_DEVICES=2,3 "$VPY" -m causal_diagsafe.cli.build_v7_latent_bank \
  --model "$TEACHER" --release "$RELEASE" --output "$BANK" \
  --rank 1 --world-size 2 --tensor-parallel-size 2 --memory 0.86 --window 32 \
  > "$RUN/bank.rank1.log" 2>&1 &
teacher1=$!
status=0
wait "$teacher0" || status=$?
wait "$teacher1" || status=$?
if (( status != 0 )); then
  exit "$status"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(--model "$MODEL" --release "$RELEASE" --bank "$BANK" --epochs 2
        --seed 42 --effective-roots 64 --student-draws 2 --max-new-tokens 2048
        --checkpoint-stride 1 --max-batch-tokens 24576
        --class-batch-size 16 --text-batch-size 2 --native-batch-size 32 --rollout-batch-size 16)
printf '%s SFT: two full epochs, natural dev after each epoch\n' "$(date -Is)"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v7_latent "${common[@]}" \
  --phase sft --output "$RUN/sft" > "$RUN/sft.log" 2>&1
printf '%s latent continuation: two full epochs\n' "$(date -Is)"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m causal_diagsafe.cli.train_v7_latent "${common[@]}" \
  --phase latent --adapter "$RUN/sft/final" --output "$RUN/latent" > "$RUN/latent.log" 2>&1
printf '%s completed; full-epoch checkpoints and autonomous dev outputs retained\n' "$(date -Is)"
