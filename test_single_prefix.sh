#!/usr/bin/env bash
# Single-prefix TTFT/KV-I/O measurement using run_full.sh machine settings.
# Usage: bash test_single_prefix.sh [--dry-run]
# Example: MODELS=opt-30b PREFIX_INDEX=0 bash test_single_prefix.sh
# Override PYTHON, WEIGHT_PATH, TOKENIZER_PATH, LONGBENCH_FILE, OFFLOAD_ROOT,
# OUTPUT_ROOT, MODELS, PREFIX_INDEX, MAX_REQUESTS, or exported runtime controls.
# POSITION_WEIGHTS optionally selects a position table (use with one model).
# Request 0 initializes the layout and is excluded from reuse summaries.
# The retained runtime uses guarded_hot_growth + access_control, not the old
# run_full.sh incremental strategy or its --reorder-topk-ratio override.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

dry_run=false
if [[ $# -eq 1 && "$1" == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash test_single_prefix.sh [--dry-run]" >&2
    exit 2
fi

export PYTHONPATH="$PWD/library/kvstore:$PWD/library/pqcode:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export KVSTORE_READ_BPS="${KVSTORE_READ_BPS:-7000000000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export KVSTORE_IOURING_QD="${KVSTORE_IOURING_QD:-32}"
export KVSTORE_REORDER_IOURING_QD="${KVSTORE_REORDER_IOURING_QD:-8}"
export FLEXGEN_FOREGROUND_CPUS="${FLEXGEN_FOREGROUND_CPUS:-0-47}"
export FLEXGEN_REORDER_CPUS="${FLEXGEN_REORDER_CPUS:-48-55}"
unset GOMP_CPU_AFFINITY OMP_PROC_BIND
# Preserve the single-prefix runner's defaults for settings absent in run_full.sh.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export KVSTORE_REORDER_SYNC="${KVSTORE_REORDER_SYNC:-1}"

PYTHON="${PYTHON:-python3}"
WEIGHT_PATH="${WEIGHT_PATH:-/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/opt_weights}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/hyk/param/opt-30b}"
LONGBENCH_FILE="${LONGBENCH_FILE:-/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/lqy/HF_HOME/datasets/THUDM___long_bench/data/narrativeqa.jsonl}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/ssd/nsccgz_zgchen_6/lqy/flexllmgen_offload_dir/single_prefix}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/pqcode_log/single_prefix}"
read -r -a models <<< "${MODELS:-opt-6.7b opt-30b opt-66b}"
if [[ ${#models[@]} -eq 0 ]]; then
    echo "MODELS must select at least one OPT model" >&2
    exit 2
fi
for model in "${models[@]}"; do
    case "${model#facebook/}" in
        opt-6.7b|opt-30b|opt-66b) ;;
        *) echo "Unsupported model: $model; choose opt-6.7b, opt-30b, opt-66b" >&2; exit 2 ;;
    esac
done

if "$dry_run"; then
    RUN_DIR="$OUTPUT_ROOT/DRY_RUN"
    RUN_OFFLOAD="$OFFLOAD_ROOT/DRY_RUN"
else
    [[ -r "$LONGBENCH_FILE" ]] || { echo "Cannot read $LONGBENCH_FILE" >&2; exit 1; }
    mkdir -p "$OUTPUT_ROOT" "$OFFLOAD_ROOT"
    RUN_DIR=$(mktemp -d "$OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU available"; print(torch.cuda.get_device_properties(0))' \
        > "$RUN_DIR/preflight.log" 2>&1 || {
        echo "GPU preflight failed; see $RUN_DIR/preflight.log" >&2
        exit 1
    }
    RUN_OFFLOAD=$(mktemp -d "$OFFLOAD_ROOT/run_XXXXXX")
fi

# Record effective machine controls alongside the commands for comparisons.
runtime_vars=(CUDA_VISIBLE_DEVICES KVSTORE_READ_BPS KVSTORE_IOURING_QD
    KVSTORE_REORDER_IOURING_QD FLEXGEN_FOREGROUND_CPUS FLEXGEN_REORDER_CPUS
    OMP_NUM_THREADS MKL_NUM_THREADS KVSTORE_REORDER_SYNC)
for name in "${runtime_vars[@]}"; do
    if "$dry_run"; then
        printf 'export %s=%q\n' "$name" "${!name}"
    else
        printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/environment.sh"
    fi
done

failed=0
for model in "${models[@]}"; do
    model="${model#facebook/}"
    percent=(100 0 100 0 100 0)
    if [[ "$model" == opt-66b ]]; then
        percent=(25 75 100 0 100 0)
    fi
    name="${model}_longbench_prefix_${PREFIX_INDEX:-0}"
    command=("$PYTHON" -u "$PWD/scripts/measure_adakvs_single_prefix.py"
        --model "facebook/$model" --percent "${percent[@]}"
        --path "$WEIGHT_PATH" --tokenizer-path "$TOKENIZER_PATH"
        --longbench-file "$LONGBENCH_FILE"
        --prefix-index "${PREFIX_INDEX:-0}" --min-questions "${MIN_QUESTIONS:-10}"
        --max-requests "${MAX_REQUESTS:-10}"
        --prefix-len "${PREFIX_LEN:-8000}" --max-length "${MAX_LENGTH:-8192}"
        --adakvs-retention "${RETENTION_RATIO:-0.25}"
        --adakvs-cache-gib "${CACHE_GIB:-1}"
        --prefill-query-chunk-size "${PREFILL_QUERY_CHUNK_SIZE:-1024}"
        --offload-dir "$RUN_OFFLOAD/$name"
        --output "$RUN_DIR/$name/requests.jsonl")
    # One output token, overlap=True and merge budget .15 are runner defaults.
    # Without an override, each model uses its own extended position table.
    if [[ -n "${POSITION_WEIGHTS:-}" ]]; then
        command+=(--position-weights "$POSITION_WEIGHTS")
    fi
    if "$dry_run"; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    mkdir -p "$RUN_DIR/$name" "$RUN_OFFLOAD/$name"
    printf '%q ' "${command[@]}" > "$RUN_DIR/$name/command.txt"
    printf '\n' >> "$RUN_DIR/$name/command.txt"
    echo "Running $name; log: $RUN_DIR/$name/run.log"
    if "${command[@]}" > "$RUN_DIR/$name/run.log" 2>&1; then
        printf '%s\tOK\n' "$name" >> "$RUN_DIR/status.tsv"
        echo "Completed $name; timings: $RUN_DIR/$name/requests.jsonl"
    else
        status=$?
        printf '%s\tFAILED\t%s\n' "$name" "$status" >> "$RUN_DIR/status.tsv"
        echo "Failed $name (exit $status); see $RUN_DIR/$name/run.log" >&2
        failed=1
    fi
done
exit "$failed"
