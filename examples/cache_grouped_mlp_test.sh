#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/cnic/work/liuql/Megatron-LM
export PYTHONPATH=/cnic/work/liuql/flash-attention/hopper:$PYTHONPATH
export NVTE_FLASH_ATTN_2=1
# Example launcher for examples/cache_grouped_mlp_test.py
#
# Usage:
#   # Single-GPU test with CacheGroupedMLP (default)
#   CUDA_VISIBLE_DEVICES=0 ./examples/cache_grouped_mlp_test.sh
#
#   # Multi-GPU test (EP=2)
#   CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 ./examples/cache_grouped_mlp_test.sh
#
#   # With activation offload
#   CUDA_VISIBLE_DEVICES=0 ACTIVATION_OFFLOAD=1 ./examples/cache_grouped_mlp_test.sh
#
#   # With GPTModel mode (SequentialMLP with Attention)
#   CUDA_VISIBLE_DEVICES=0 ./examples/cache_grouped_mlp_test.sh --test-mode gpt_model
#
#   # Custom batch_size and seq_len
#   CUDA_VISIBLE_DEVICES=0 ./examples/cache_grouped_mlp_test.sh --num-layers 4 --batch-size 2 --seq-len 4096

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=1}"
: "${ENABLE_NSYS:=1}"
: "${NSYS_OUT:=/cnic/work/liuql/Megatron-LM/nsys_cache_grouped_mlp}"
: "${ACTIVATION_OFFLOAD:=1}"
: "${NUM_LAYERS:=1}"
: "${BATCH_SIZE:=1}"
: "${SEQ_LEN:=4096}"
: "${TEST_MODE:=cache_grouped_mlp}"

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
  --test-mode "${TEST_MODE}" \
  --num-global-experts 64 \
  --num-local-experts 64 \
  --moe-router-topk 8 \
  --moe-token-dispatcher-type alltoall \
  --hidden-size 7168 \
  --ffn-hidden-size 2048 \
  --num-attention-heads 8 \
  --num-layers "${NUM_LAYERS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --bf16 \
  --trace-offload \
  --use-flash-attn \
  --use-transformer-engine \
  --num-attention-heads 128 \
  "${ACTIVATION_OFFLOAD_CMD[@]}" \
  "$@"