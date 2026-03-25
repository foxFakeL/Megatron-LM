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

# Default to single GPU if not specified
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# Test configuration
NUM_GLOBAL_EXPERTS=${NUM_GLOBAL_EXPERTS:-64}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-4096}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-10240}
ITERS=${ITERS:-5}

echo "========================================"
echo "FusedDispatcherCacheGroupedMLP Test"
echo "========================================"
echo "GPUs: $NPROC_PER_NODE"
echo "Global experts: $NUM_GLOBAL_EXPERTS"
echo "Batch size: $BATCH_SIZE"
echo "Seq length: $SEQ_LEN"
echo "Iterations: $ITERS"
echo "========================================"

# Run the test
python examples/cache_grouped_mlp_test.py \
    --test-mode fused_dispatcher_mlp \
    --num-global-experts $NUM_GLOBAL_EXPERTS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --bf16 \
    --iters $ITERS

echo "========================================"
echo "Test completed!"
echo "========================================"