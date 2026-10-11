#!/usr/bin/env bash
# Real dependency only: leave the whole existing v3 launcher and its GPU jobs alone.
ROOT=${DIAGSAFE_ROOT:-/disk/u20110017/diagsafe_v1}
CODE="$ROOT/training_code/external_baselines_20261009"
PY="$ROOT/work/external_eval_env/bin/python"
SHIELD_PY="$ROOT/work/shieldstral_eval_env/bin/python"
LLAMA_PY="$ROOT/work/llamaguard4_eval_env/bin/python"
CONFIG="$CODE/configs/external_baselines_20261009.json"
DATA="$ROOT/work/external_baselines_20261009/prepared"
RUN="$ROOT/runs/external_baselines_20261009"
REFERENCES="$ROOT/work/external_refs"
export PYTHONPATH="$CODE:$ROOT/training_code/v1_20261007"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$RUN/logs"

wait_existing() {
    local pid="$1" start="$2" current
    if [ -z "$pid" ]; then return; fi
    while [ -r "/proc/$pid/stat" ]; do
        current=$(awk '{print $22}' "/proc/$pid/stat")
        if [ "$current" != "$start" ]; then break; fi
        sleep 20
    done
}

# Capture identities, not just PIDs, so PID reuse cannot attach us to another experiment.
if [ -n "$WAIT_EXPERIMENT_PID" ]; then
    printf 'Waiting for complete current experiment launcher PID %s\n' "$WAIT_EXPERIMENT_PID"
    wait_existing "$WAIT_EXPERIMENT_PID" "$WAIT_EXPERIMENT_START"
fi
for kind in DATA; do
    pid_var="WAIT_${kind}_PID" start_var="WAIT_${kind}_START"
    wait_existing "${!pid_var}" "${!start_var}"
done
printf 'Existing experiment and public-data preparation dependencies finished; starting external evaluation.\n'

export HF_HUB_OFFLINE=1
if [ "${1:-all}" = all ]; then
    "$PY" -m causal_diagsafe.cli.prepare_external_benchmarks --root "$ROOT" --config "$CONFIG" \
        --deck "$ROOT/work/public_eval_v1_20261007/policy_deck.json" --references "$REFERENCES" --output "$DATA" \
        > "$RUN/logs/prepare.log" 2>&1
    printf 'prepare exit=%s\n' "$?"
fi

wait_weight_download() {
    local model="$1" kind="$2" producer_name="WAIT_${2}_PID" start_name="WAIT_${2}_START"
    local producer="${!producer_name}" start="${!start_name}" source="$ROOT/raw/external_models/$1/download_source.json"
    local report="$ROOT/work/external_baselines_20261009/fetch_${kind,,}.json"
    if [ "$model" = safeatlas4b ] && [ -n "$WAIT_SAFEATLAS_PID" ]; then
        producer="$WAIT_SAFEATLAS_PID" start="$WAIT_SAFEATLAS_START"
        report="$ROOT/work/external_baselines_20261009/fetch_models_safeatlas4b.json"
    fi
    if [ "$model" = llamaguard4 ] && [ -n "$WAIT_LLAMA_PID" ]; then
        producer="$WAIT_LLAMA_PID" start="$WAIT_LLAMA_START"
        report="$ROOT/work/external_baselines_20261009/fetch_models_llamaguard4.json"
    fi
    if [ "$model" = qwen3vl4b ] || [[ "$model" == diagsafe_v3* ]]; then return; fi
    while [ ! -f "$source" ] && [ -n "$producer" ] && [ -r "/proc/$producer/stat" ]; do
        if [ "$(awk '{print $22}' "/proc/$producer/stat")" != "$start" ]; then break; fi
        if "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=json.load(p.open()) if p.exists() else []; sys.exit(0 if any(r["name"]==sys.argv[2] and r["status"]=="unavailable" for r in rows) else 1)' \
            "$report" "$model"; then break; fi
        sleep 20
    done
}

guard() {
    local model="$1" gpu="$2" data="$3" out="$4" phase="$5" protocol="$6" interpreter="$PY"
    local tasks=() adapter=()
    mapfile -t tasks < <("$PY" -c 'import json,sys; from pathlib import Path; c=json.load(open(sys.argv[1])); tasks=c[sys.argv[2]+"_tasks"]; allowed=set(c["system_tasks"]); p=Path(sys.argv[4])/sys.argv[5]/sys.argv[6]/"summary.json"; done=json.load(p.open()) if p.exists() else {}; print("\n".join(t for t in tasks if (sys.argv[3]!="post" or t in allowed) and done.get(t,{}).get("status")!="evaluated"))' "$CONFIG" "$phase" "$(basename "$out")" "$out" "$model" "$protocol")
    if [ "${#tasks[@]}" -eq 0 ]; then return; fi
    if [ "$model" = shieldstral3b ] || [ "$model" = guardreasoner_omni7b ]; then interpreter="$SHIELD_PY"; fi
    if [ "$model" = llamaguard4 ]; then interpreter="$LLAMA_PY"; fi
    wait_weight_download "$model" MODELS
    if [[ "$model" == diagsafe_v3* ]]; then
        local variant=v3
        if [ "$model" = diagsafe_v3_no_relation ]; then variant=no_relation; fi
        local selected="$ROOT/runs/semantic_v3_20261008/seed42/$variant/selected"
        if [ ! -d "$selected" ]; then printf 'No selected adapter for %s; recording unavailability.\n' "$model"; return; fi
        adapter=(--adapter "$selected")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$interpreter" -m causal_diagsafe.cli.evaluate_external_baselines \
        --root "$ROOT" --config "$CONFIG" --model "$model" --protocol "$protocol" "${adapter[@]}" \
        --data "$data" --output "$out/$model/$protocol" --tasks "${tasks[@]}" \
        > "$RUN/logs/${model}_${protocol}_${phase}_$(basename "$out").log" 2>&1
    local exit_status=$?
    printf '%s %s %s %s exit=%s\n' "$model" "$protocol" "$phase" "$(basename "$out")" "$exit_status"
}

wave() {
    local data="$1" out="$2" phase="$3" protocol="$4"
    shift 4
    local gpu model slot
    local pids=()
    for model in "$@"; do
        gpu=-1
        while [ "$gpu" -lt 0 ]; do
            for slot in 0 1 2 3; do
                if [ -z "${pids[$slot]}" ]; then
                    gpu="$slot"
                    break
                fi
                if ! kill -0 "${pids[$slot]}" 2>/dev/null; then
                    wait "${pids[$slot]}"
                    unset 'pids[slot]'
                    gpu="$slot"
                    break
                fi
            done
            if [ "$gpu" -lt 0 ]; then sleep 2; fi
        done
        guard "$model" "$gpu" "$data" "$out" "$phase" "$protocol" &
        pids[$gpu]="$!"
    done
    for slot in 0 1 2 3; do if [ -n "${pids[$slot]}" ]; then wait "${pids[$slot]}"; fi; done
}

native_phase() {
    local phase="$1" data="$2" out="$3"
    wave "$data" "$out" "$phase" native qwen3vl4b guardreasoner_vl_eco3b safeatlas4b shieldstral3b \
        guardreasoner_vl7b safeatlas8b guardreasoner_omni7b llamaguard4 qwen3vl8b wildguard7b qwen3guard4b
    wave "$data" "$out" "$phase" shared_deck qwen3vl8b qwen3vl4b shieldstral3b diagsafe_v3 diagsafe_v3_no_relation
}

system_phase() {
    local phase="$1" tasks=() pids=() rank
    mapfile -t tasks < <("$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); allowed=set(c["system_tasks"]); print("\n".join(t for t in c[sys.argv[2]+"_tasks"] if t in allowed))' "$CONFIG" "$phase")
    for rank in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="$rank" "$PY" -m causal_diagsafe.cli.generate_external_answers \
            --root "$ROOT" --config "$CONFIG" --data "$DATA" --output "$RUN/answers" \
            --rank "$rank" --world-size 4 --tasks "${tasks[@]}" \
            > "$RUN/logs/answer_${phase}_rank${rank}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; printf 'answer job %s exit=%s\n' "$pid" "$?"; done
    "$PY" -m causal_diagsafe.cli.prepare_generated_responses --data "$DATA" --answers "$RUN/answers" \
        --output "$RUN/generated_s5" --tasks "${tasks[@]}" > "$RUN/logs/join_answers_${phase}.log" 2>&1
    # Post guards audit the very same cached answers, including pre-guard held requests.
    native_phase "$phase" "$RUN/generated_s5" "$RUN/post"
    local multimodal=() task
    for task in "${tasks[@]}"; do if [ "$task" != harmbench_test ]; then multimodal+=("$task"); fi; done
    wait_weight_download multimodal_judge32b JUDGES
    pids=()
    for rank in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="$rank" "$PY" -m causal_diagsafe.cli.judge_external_responses \
            --root "$ROOT" --references "$REFERENCES" --data "$RUN/generated_s5" --output "$RUN/judges/multimodal" \
            --kind multimodal --rank "$rank" --world-size 4 --tasks "${multimodal[@]}" \
            > "$RUN/logs/judge_${phase}_rank${rank}.log" 2>&1 &
        pids+=("$!")
    done
    for rank in "${pids[@]}"; do wait "$rank"; printf 'open judge worker %s exit=%s\n' "$rank" "$?"; done
    "$PY" -m causal_diagsafe.cli.judge_external_responses \
        --root "$ROOT" --references "$REFERENCES" --data "$RUN/generated_s5" --output "$RUN/judges/multimodal" \
        --kind multimodal --merge-only --world-size 4 --tasks "${multimodal[@]}"
    printf 'open judge %s merge exit=%s\n' "$phase" "$?"
    if [[ " ${tasks[*]} " == *" harmbench_test "* ]]; then
        wait_weight_download harmbench_judge13b JUDGES
        CUDA_VISIBLE_DEVICES=0 "$PY" -m causal_diagsafe.cli.judge_external_responses \
            --root "$ROOT" --references "$REFERENCES" --data "$RUN/generated_s5" --output "$RUN/judges/harmbench" \
            --kind harmbench --tasks harmbench_test > "$RUN/logs/harmbench_judge.log" 2>&1
        printf 'HarmBench official judge exit=%s\n' "$?"
    fi
    "$PY" -m causal_diagsafe.cli.summarize_external_systems --config "$CONFIG" --data "$DATA" \
        --generated "$RUN/generated_s5" --native "$RUN/native" --post "$RUN/post" \
        --judges "$RUN/judges" --output "$RUN/systems" --tasks "${tasks[@]}"
}

# Small full suites first; large attack/over-refusal extensions follow without test tuning.
if [ "${1:-all}" = all ]; then
    native_phase primary "$DATA" "$RUN/native"
    system_phase primary
fi
native_phase extended "$DATA" "$RUN/native"
system_phase extended
mapfile -t selected_system_tasks < <("$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["system_tasks"]))' "$CONFIG")
"$PY" -m causal_diagsafe.cli.summarize_external_systems --config "$CONFIG" --data "$DATA" \
    --generated "$RUN/generated_s5" --native "$RUN/native" --post "$RUN/post" \
    --judges "$RUN/judges" --output "$RUN/systems" --tasks "${selected_system_tasks[@]}"
printf 'External baseline sequence finished. Missing branches remain explicit in logs and denominators.\n'
