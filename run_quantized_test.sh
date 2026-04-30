#!/bin/bash
# Quick test script for quantized optimizer training with QuantizedDispatcher

# Add grouped_gemm to PYTHONPATH (installed locally at /cnic/work/liuql/grouped_gemm)
export PYTHONPATH="/cnic/work/liuql/grouped_gemm:$PYTHONPATH"

# Single GPU test with minimal iterations
echo "Running single GPU test..."
MASTER_ADDR=localhost MASTER_PORT=29500 torchrun --nproc_per_node=1 pretrain_qwen3_moe_quantized.py \
    --num-layers 2 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --num-experts 64 \
    --moe-ffn-hidden-size 10240 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --train-iters 50 \
    --lr 1e-4 \
    --warmup-steps 5 \
    --log-interval 5 \
    --quant-group-size 128 \
    --score-update-interval 10 \
    --top-bf16-ratio 0.05 \
    --top-int8-ratio 0.30 \
    --lr-quant 1e-4 \
    --timing-warmup-iters 3 \
    --no-persist-layer-norm \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --no-gradient-accumulation-fusion \
    --vocab-size 151936 \
    --make-vocab-size-divisible-by 64 \
    --tokenizer-type NullTokenizer \
    --seed 42 \
    --bf16

echo "Test completed!"