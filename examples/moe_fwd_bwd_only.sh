#!/usr/bin/env bash
set -euo pipefail

# Example launcher for examples/moe_fwd_bwd_only.py
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./examples/moe_fwd_bwd_only.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=1}"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/moe_fwd_bwd_only.py" \
  --iters 1 \
  --micro-batch-size 2 \
  --seq-length 64 \
  --vocab-size 128 \
  --num-layers 5 \
  --hidden-size 2048 \
  --ffn-hidden-size 10240 \
  --num-attention-heads 4 \
  --num-local-experts 64 \
  --moe-router-topk 8 \
  --moe-token-dispatcher-type alltoall \
  --trace-offload
