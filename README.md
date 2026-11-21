# JudiAct

JudiAct is a four-action multimodal safety guard. It generates three short review items (source fact, condition, and rule link), then audits the original inputs and rules alongside its own reviews. This repository contains only training and evaluation code. Data, media, model weights, and upstream benchmark repositories are separate inputs.

## Install

Use Python 3.11 on Linux with a CUDA-capable GPU. The training recipe uses four A800 GPUs, BF16 backbone computation, and FP32 language-attention LoRA parameters. Install the PyTorch 2.8 CUDA wheel appropriate to your driver. For the FlashAttention recipe, install `flash-attn==2.8.3.post2` against that PyTorch/CUDA build (`--no-build-isolation` when building from source).

```bash
pip install -e '.[audit]'
# In the Stage 2 environment:
pip install -e '.[rl]'
```

Use local snapshots of `Qwen/Qwen3-VL-4B-Instruct` for training and auditing. No command downloads weights or starts a service.

## Train

Set `MODEL` to the local base-model directory and `TRAIN_DATA` to a prepared training bundle. The loader reads train/dev only. It requires the annotated main members, auxiliary records with source-native labels, verified relations, reviewed field masks, and the frozen train-only process reference bank. See the input layout below.

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

Stage 1 selects among three complete epochs on dev using the equal mean of request and answer four-action Macro-F1. Stage 2 uses the selected weights and a fresh optimizer. Each update uses the greedy audit and two sampled replacements of one short item. The scheduled one-epoch endpoint is used without dev or test selection. Both stages accept `--resume` for full-state checkpoints; retain the original output directory and data order. Stage 2 records its endpoint in `training_result.json`.

The `--release` bundle has the following paths. Prepare the bundle from your source data before running training; the public Hub viewer export is not the trainer's layout. For exact training reproduction, retain the original supervision sidecars and bank rather than regenerating or guessing missing supervision.

| Path | Input |
| --- | --- |
| `data/core_request/train.jsonl`, `data/s5_response/train.jsonl` | Main request and answer records, with `sample_id`, `group_id`, `split`, `task_type`, `model_input`, `gold`, and `source` |
| `data/natural_request/dev.jsonl`, `data/natural_response/dev.jsonl` | Fixed natural dev views, with the same record fields |
| `data/causal/{verified_train,noncausal_train,verified_s2_quartets_train}.jsonl` | Explicit pair and four-corner indices; only eligible relations supervise matching |
| `data/auxiliary/{native_text_train,native_image_train}.jsonl` | Source-native auxiliary records; not projected into four actions |
| `reviews/{supervision,projection,conditions,historical_conditions}.jsonl` | Field masks, rule-revision applicability and observed condition references |
| `reviews/{confirmed_core_image_aliases,confirmed_image_root_families}.jsonl` | Image family information for source-root grouping |
| `process_references/bank.rank0.jsonl` | Reference bank with `sample_id`, `source_identity`, `policy_identity`, and `candidate`; `--bank` overrides the path |

Image paths in a prepared training bundle must resolve on the training machine. The reader accepts canonical review filenames or an unambiguous corresponding filename in the original bundle.

## Evaluate JudiBench

Use a local [JudiBench](https://huggingface.co/datasets/Devin6666/JudiBench) snapshot containing its JSONL and referenced media. Access is governed by the dataset owner. Set `ADAPTER` to a trained checkpoint directory.

```bash
judiact-prepare-data --snapshot "$JUDIBENCH" --output data/judibench
torchrun --standalone --nproc-per-node=4 -m judiact.cli.evaluate \
  --model "$MODEL" --adapter "$ADAPTER" --release data/judibench \
  --output runs/judibench --include-mechanism
```

`summary.json` reports four-action Macro-F1 on the natural request and answer views, fixed three-class diagnostic Macro-F1, and (when requested) exact pair and quartet scores. Withheld gold is not scored. An unparseable prediction is not silently removed, and an uncertain model output is not turned into a binary corner for an interaction score. Localization is separate from the action decision.

## Evaluate public benchmarks

`judiact-public-eval` runs on caller-provided normalized benchmark manifests and trusted upstream rubric checkouts. The manifest directory has one JSONL per task, named as in [`public_protocol.json`](judiact/public_protocol.json). Each row supplies `sample_id`, `root_id`, `audit_target`, `native_harm`, `subset`, `category`, and `modality`; `visible_json` encodes `query`, `images`, `ocr_text`, `policy_deck`, and `response_candidate`. Images can use paths relative to the manifest. `provenance_json` carries source-only rubric inputs where required (e.g., the SIUO references). It is not shown to the answerer or guard. Keep source variants and rubric metadata unchanged when preparing these records.

```bash
judiact-public-eval prepare --inputs "$INPUTS" --output runs/public
judiact-public-eval generate --model "$MODEL" --output runs/public
judiact-public-eval guard --model "$MODEL" --adapter "$ADAPTER" --output runs/public
judiact-public-eval assemble --output runs/public
judiact-public-eval controlled --model "$MODEL" --output runs/public
judiact-public-eval materialize --output runs/public
judiact-public-eval translate --judge-model "$JUDGE" --output runs/public  # for SORRY-Bench
judiact-public-eval judge --judge-model "$JUDGE" --references "$REFERENCES" \
  --harmbench-classifier "$HARMBENCH_JUDGE" --output runs/public
judiact-public-eval score --output runs/public
```

`JUDGE` is a local `Qwen/Qwen3-VL-32B-Instruct` snapshot. `REFERENCES` contains the upstream rubric checkouts under the directory names read by `benchmark_official.py` and `harmbench.py`. For the official HarmBench classifier and copyright matching, install `.[harmbench]` and supply the released classifier and reference hashes. The optional `--backend vllm` is for greedy judge/translation runs in a separate compatible vLLM environment; it is not used for sampled answer generation. Inference steps support `--rank`/`--world-size` sharding. Use a distinct `--name` for every adapter compared in one run.

The systems share one unguarded answer pool; Post audits actual candidates. A Post `warn` requests one revision, and the delivered response is judged rather than assumed safe. `score` uses benchmark-specific aggregations and records unresolved judgments separately. For native binary manifests, run `guard --phases native` and `score-native`; the fixed four-action projection is distinct from the trained binary auxiliary interface (`--native-harm`).
