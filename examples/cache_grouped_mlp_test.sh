#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/cnic/work/liuql/Megatron-LM
export PYTHONPATH=/cnic/work/liuql/flash-attention/hopper:$PYTHONPATH
export NVTE_FLASH_ATTN_2=1
# Example launcher for examples/cache_grouped_mlp_test.py
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./examples/cache_grouped_mlp_test.sh
#   CUDA_VISIBLE_DEVICES=0 ACTIVATION_OFFLOAD=1 ./examples/cache_grouped_mlp_test.sh
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ./examples/cache_grouped_mlp_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=1}"
: "${ENABLE_NSYS:=1}"
: "${NSYS_OUT:=/cnic/work/liuql/Megatron-LM/nsys_cache_grouped_mlp}"
: "${ACTIVATION_OFFLOAD:=1}"

NSYS_CMD=()
if [[ "${ENABLE_NSYS}" == "1" ]]; then
  NSYS_CMD=(
    nsys profile
    -t cuda,nvtx,osrt
    --force-overwrite=true
    -o "${NSYS_OUT}"
  )
fi

ACTIVATION_OFFLOAD_CMD=()
if [[ "${ACTIVATION_OFFLOAD}" == "1" ]]; then
  ACTIVATION_OFFLOAD_CMD=(--activation-offload)
fi

${NSYS_CMD[@]} torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/cache_grouped_mlp_test.py" \
  --iters 12 \
  --num-global-experts 64 \
  --num-sets 8 \
  --hidden-size 7168 \
  --ffn-hidden-size 2048 \
  --tokens-per-expert 6256 \
  --bf16 \
  --trace-offload \
  "${ACTIVATION_OFFLOAD_CMD[@]}"