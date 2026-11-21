# Public benchmark evaluation

[Back to JudiAct](../README.md)

This runner evaluates delivered answers with a guard before or after generation, and also supports binary classification on public benchmarks. It consumes prepared manifests; it does not download or convert the upstream benchmark datasets. See [public_protocol.json](../judiact/public_protocol.json) for supported task names, decoding settings, and the controller prompts.

## Required inputs

| Path or variable | Contents |
| --- | --- |
| `INPUTS` | A local directory with a JSONL manifest per task. Use the task name in `public_protocol.json` as the filename stem. |
| `MODEL`, `ADAPTER` | Local Qwen3-VL-4B-Instruct and JudiAct checkpoint directories. |
| `JUDGE` | A local Qwen3-VL-32B-Instruct snapshot for source-rubric judging. |
| `REFERENCES` | Trusted upstream rubric checkouts. The relevant paths are listed below. |
| `HARMBENCH_JUDGE` | Local official HarmBench classifier, required for HarmBench only. Install `.[harmbench]` and retain the upstream copyright reference hashes. |

Manifest rows carry `sample_id`, `root_id`, `audit_target`, `native_harm`, `subset`, `category`, and `modality`. `visible_json` encodes the query, images, OCR, rule cards, and candidate answer with the same keys as `model_input`. Image paths may be relative to the manifest. `provenance_json` carries judge-only task inputs, such as the reference answer for SIUO. It is not shown to the guard or answerer. Preserve original variants, IDs, and rubric metadata.

For the tasks being scored, place upstream files under these paths relative to `REFERENCES`:

| Benchmark | Files read |
| --- | --- |
| MSSBench | `MSSBench/utils/gpt4_eval.py` |
| MM-SafetyBench++ | `EchoSafe/evaluation.py` |
| VLSBench | `VLSBench/eval_utils.py` |
| MM-SafetyBench | `MM-SafetyBench/evaluation.py` |
| SIUO | `SIUO/eval/gpt-eval.py` |
| XSTest | `XSTest/evaluation/classify_completions_gpt.py` |
| SORRY-Bench | `SORRYBench/data/sorry_bench/judge_prompts.jsonl` and its upstream response preprocessing files |
| MSTS | `MSTS/annotation_guidelines.txt` |
| HarmBench | `HarmBench/eval_utils.py` and its upstream copyright reference assets |

These are local evaluations using the source rubrics, not reproductions of the original hosted judge outputs. Record the upstream revisions and local checkpoints with the run.

## Delivered answers

```bash
RUN=runs/public
judiact-public-eval prepare --inputs "$INPUTS" --output "$RUN"
judiact-public-eval generate --model "$MODEL" --output "$RUN"
judiact-public-eval guard --model "$MODEL" --adapter "$ADAPTER" \
  --name judiact --phases pre post --output "$RUN"
judiact-public-eval assemble --output "$RUN"
judiact-public-eval controlled --model "$MODEL" --output "$RUN"
judiact-public-eval materialize --output "$RUN"
```

If the run includes SORRY-Bench, run `judiact-public-eval translate --judge-model "$JUDGE" --output "$RUN"` before judging. This step processes saved responses when the source rubric calls for translation.

```bash
judiact-public-eval judge --judge-model "$JUDGE" --references "$REFERENCES" \
  --harmbench-classifier "$HARMBENCH_JUDGE" --output "$RUN"
judiact-public-eval score --output "$RUN"
```

Omit `--harmbench-classifier` when not evaluating HarmBench. Use `--tasks` on `prepare` to select a subset of available manifests. Use a distinct `--name` for each checkpoint before assembly to share the same unguarded answer pool. Inference can be sharded with `--rank` and `--world-size`; complete the shards of a step before the next step consumes them. The optional `--backend vllm` applies to judge and translation runs in a separate compatible environment, not answer generation.

Post audits the actual candidate answer. A Post `warn` requests one revision, whose delivered text is then judged. The runner writes per-task reports under `reports/` with benchmark-specific aggregation and unresolved judgment counts.

## Native binary benchmarks

Prepare the native manifests in the same way, then use:

```bash
judiact-public-eval guard --model "$MODEL" --adapter "$ADAPTER" \
  --name judiact --phases native --output "$RUN"
judiact-public-eval score-native --output "$RUN"
```

The default projects four actions with `p(review) + p(refuse) >= 0.5`. Passing `--native-harm` selects the separately trained source-native binary interface instead. Report these readouts separately.
