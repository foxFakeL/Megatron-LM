#!/usr/bin/env bash
set -euo pipefail

# Example launcher for examples/moe_fwd_bwd_only.py
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./examples/moe_fwd_bwd_only.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=1}"
: "${ENABLE_NSYS:=0}"
: "${NSYS_OUT:=/workspace/nsys_moe_fwd_bwd}"

NSYS_CMD=()
if [[ "${ENABLE_NSYS}" == "1" ]]; then
  NSYS_CMD=(
    nsys profile
    -t cuda,nvtx,osrt
    --force-overwrite=true
    -o "${NSYS_OUT}"
  )
fi

${NSYS_CMD[@]} torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/moe_fwd_bwd_only.py" \
  --iters 1 \
  --micro-batch-size 1 \
  --seq-length 50000 \
  --vocab-size 128 \
  --num-layers 5 \
  --hidden-size 7168 \
  --ffn-hidden-size 2408 \
  --num-attention-heads 128 \
  --bf16 \
  --use-transformer-engine \
  --use-flash-attn \
  --num-local-experts 64 \
  --moe-router-topk 8 \
  --moe-token-dispatcher-type alltoall \
  --trace-offload \
  --bf16
