#!/bin/bash
# Test script for FusedDispatcherCacheGroupedMLP
#
# Usage:
#   Single GPU:
#     bash examples/run_fused_dispatcher_test.sh
#
#   Multi-GPU (EP=2):
#     CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash examples/run_fused_dispatcher_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${NNODES:=1}"
: "${NPROC_PER_NODE:=1}"
: "${ENABLE_NSYS:=1}"
: "${NSYS_OUT:=/data/home/scwb466/run/liuql/nsys_fused_dispatcher_mlp}"
: "${ACTIVATION_OFFLOAD:=1}"
: "${NUM_LAYERS:=1}"
: "${BATCH_SIZE:=1}"
: "${SEQ_LEN:=4096}"
: "${TEST_MODE:=fused_dispatcher_mlp}"

echo "========================================"
echo "FusedDispatcherCacheGroupedMLP Test"
echo "========================================"
echo "GPUs: $NPROC_PER_NODE"
echo "Global experts: 64"
echo "Batch size: $BATCH_SIZE"
echo "Seq length: $SEQ_LEN"
echo "Iterations: $ITERS"
echo "========================================"


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
  --master_port "${MASTER_PORT:-29500}" \
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
  --bf16 \
  --trace-offload \
  --use-flash-attn \
  --use-transformer-engine \
  --num-attention-heads 128 \
  "${ACTIVATION_OFFLOAD_CMD[@]}" \
  "$@"

# # Run the test
# python examples/cache_grouped_mlp_test.py \
#     --test-mode fused_dispatcher_mlp \
#     --num-global-experts 64 \
#     --batch-size $BATCH_SIZE \
#     --seq-len $SEQ_LEN \
#     --hidden-size $HIDDEN_SIZE \
#     --ffn-hidden-size $FFN_HIDDEN_SIZE \
#     --bf16 \
#     --iters $ITERS

# echo "========================================"
# echo "Test completed!"
# echo "========================================"