#!/usr/bin/env bash
# Usage: bash scripts/run_adakvs.sh [additional flexllmgen.flex_opt options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${WEIGHT_PATH:?Set WEIGHT_PATH to the FlexLLMGen weight root}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the tokenizer directory or model ID}"
: "${PREFIX_FILE:?Set PREFIX_FILE to the shared UTF-8 prefix}"
: "${OFFLOAD_DIR:?Set OFFLOAD_DIR to a writable SSD directory}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
exec "${PYTHON:-python3}" -m flexllmgen.flex_opt --adakvs \
  --model "${MODEL:-facebook/opt-30b}" --path "$WEIGHT_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" --offload-dir "$OFFLOAD_DIR" \
  --prefix-file "$PREFIX_FILE" --suffix ' Question: What is the topic? Answer:' \
  --suffix ' Question: What is the conclusion? Answer:' \
  --gpu-batch-size 1 --num-gpu-batches 1 --gen-len 1 \
  --percent 100 0 100 0 100 0 --adakvs-retention .25 --adakvs-cache-gib 1 \
  "$@"
