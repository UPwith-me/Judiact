# Training and reproduction

[Back to JudiAct](../README.md)

## Evaluate on JudiBench

Install the `audit` extra as described in the README. Set `MODEL` to a local Qwen3-VL-4B-Instruct snapshot and `ADAPTER` to a compatible JudiAct checkpoint directory. Keep the adapter weights and audit configuration together. Either the selected Stage 1 checkpoint or the completed Stage 2 checkpoint can be evaluated; report which one you used.

Obtain the [JudiBench snapshot](https://huggingface.co/datasets/Devin6666/JudiBench) under its access terms. If login is required, run `hf auth login` with a token that can read the dataset. Keep the media directory next to the JSONL files at their published paths.

```bash
export JUDIBENCH="$PWD/data/JudiBench-snapshot"
hf download Devin6666/JudiBench --repo-type dataset --local-dir "$JUDIBENCH" \
  --include 'data/test.jsonl' 'media/*' 'relations/test_verified.jsonl' 'quartets/test.jsonl'

judiact-prepare-data --snapshot "$JUDIBENCH" --output data/judibench
judiact-evaluate --model "$MODEL" --adapter "$ADAPTER" --release data/judibench \
  --output runs/judibench --attention-backend sdpa --batch-size 4 --include-mechanism
```

For the distributed FlashAttention recipe, use `torchrun --standalone --nproc-per-node=4 -m judiact.cli.evaluate` with the same paths and `--attention-backend flash_attention_2`. To continue a partially completed run, use `--resume` with its original inputs, checkpoint, and output directory.

- **Classification:** `summary.json` scores request and candidate-answer four-action Macro-F1 separately on the 5,274 primary audits. `diagnostic_macro_F1` uses the fixed label set `0`, `1`, and `uncertain`, with field-specific masks. A failed prediction on a valid target is not dropped.
- **Contextual comparisons:** `--include-mechanism` predicts additional members needed for the published pairs and quartets without changing the classification denominator. It reports exact pair and quartet scores. The 960 matched candidate-answer comparisons in the dataset card are a separate analysis; this CLI does not import that index.
- **Run artifacts:** `predictions.jsonl` retains audit traces; `evaluation_metrics.json` contains the scalar test metrics. Keep the dataset revision, checkpoint, and `settings.json` with reported scores.

## Train JudiAct

The paper trains on four A800 80 GB GPUs, with BF16 backbone computation and FP32 language-attention LoRA parameters. Use Linux, Python 3.11, and the PyTorch/CUDA versions in the README. Install `flash-attn==2.8.3.post2` against that PyTorch/CUDA build (using `--no-build-isolation` if building from source). The cost snapshot command uses FlashAttention. Stage 2 also needs the `rl` extra:

```bash
python -m pip install -e '.[rl]'
```

Set `MODEL` to a local Qwen3-VL-4B-Instruct snapshot and `TRAIN_DATA` to the prepared bundle below. Run the stages in order because the second stage uses the selected SFT adapter and its frozen training-set error costs.

```bash
torchrun --standalone --nproc-per-node=4 -m judiact.cli.train_sft \
  --model "$MODEL" --release "$TRAIN_DATA" --output runs/sft

SFT=$(python -c 'import json; print(json.load(open("runs/sft/selection.json"))["selected_checkpoint"])')

torchrun --standalone --nproc-per-node=4 -m judiact.cli.prepare_costs \
  --model "$MODEL" --adapter "$SFT" --release "$TRAIN_DATA" --output runs/costs

torchrun --standalone --nproc-per-node=4 -m judiact.cli.train_rl \
  --model "$MODEL" --checkpoint "$SFT" --release "$TRAIN_DATA" \
  --costs runs/costs/action_costs.json --output runs/rl
```

Stage 1 selects among three complete epochs using the equal mean of request and answer Macro-F1 on the 2,419 model-selection audits. Stage 2 starts a fresh optimizer and trains for one epoch. Each update compares the greedy audit with two independently sampled replacements for one short review item. Use its fixed completed endpoint from `runs/rl/training_result.json`; neither stage selects a checkpoint on test. Both training commands accept `--resume` with a full-state checkpoint, retaining the original output directory and training inputs.

### Training bundle

[JudiAct-Data](https://huggingface.co/datasets/Devin6666/JudiAct-Data) is the source for the released annotations, auxiliary cases, and relation indices. Its Hub export is not a drop-in `--release` directory. `judiact-prepare-data` prepares **JudiBench evaluation only**. Exact training requires the original reviewed supervision and frozen train-only reference bank; do not recreate unobserved conditions or infer them from final actions.

| Path in `--release` | Contents |
| --- | --- |
| `data/core_request/train.jsonl`, `data/s5_response/train.jsonl` | The main request and answer audits. Each row has `sample_id`, `group_id`, `split`, `task_type`, `model_input`, `gold`, and `source`. |
| `data/natural_request/dev.jsonl`, `data/natural_response/dev.jsonl` | The fixed model-selection views (958 requests and 1,461 answers). |
| `data/causal/{verified_train,noncausal_train,verified_s2_quartets_train}.jsonl` | Pair and quartet indices. Only eligible relations provide conditional matching targets. |
| `data/auxiliary/{native_text_train,native_image_train}.jsonl` | Original source-native records, expanded by the loader. They are not mapped to four-action labels. |
| `reviews/{supervision,projection,conditions,historical_conditions}.jsonl` | Reviewed masks, rule-revision applicability, and observed condition references. |
| `reviews/{confirmed_core_image_aliases,confirmed_image_root_families}.jsonl` | Image family information for grouping related cases by source. |
| `process_references/bank.rank0.jsonl` | Frozen train-only references with `sample_id`, `source_identity`, `policy_identity`, and `candidate`. |

Image paths must resolve on the training machine. The reader accepts the canonical review paths above or an unambiguous corresponding legacy filename. If you set `--bank`, use the same bank in all three commands. Preserve the original source-root splits and field masks; keep test cases out of the training bundle.
