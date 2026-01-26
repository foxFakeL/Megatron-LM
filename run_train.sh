#!/usr/bin/env bash
#!/usr/bin/env bash
set -euo pipefail

# Single-node torchrun launcher.
# Override any of these via env vars.

DATA_DIR=${DATA_DIR:-/dataset}
DATA_PREFIX=${DATA_PREFIX:-${DATA_DIR}/gpt2_text_document_text_document}
VOCAB_FILE=${VOCAB_FILE:-${DATA_DIR}/vocab.json}
MERGE_FILE=${MERGE_FILE:-${DATA_DIR}/merges.txt}

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6000}

if command -v nvidia-smi >/dev/null 2>&1; then
  GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
else
  GPUS_PER_NODE=${GPUS_PER_NODE:-1}
fi

# 如果设置了CUDA_VISIBLE_DEVICES，则根据其数量设置GPUS_PER_NODE
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS_PER_NODE=${#cuda_devices[@]}
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# Default EP uses all local ranks.
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-${GPUS_PER_NODE}}

# MoE weight cache toggle (1=enable, 0=disable)
USE_MOE_CACHE=${USE_MOE_CACHE:-1}

SAVE_DIR=${SAVE_DIR:-./checkpoints/moe_cache_demo}
mkdir -p "${SAVE_DIR}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

DATA_ARGS=(
  --legacy-tokenizer
  --tokenizer-type GPT2BPETokenizer
  --vocab-file "${VOCAB_FILE}"
  --merge-file "${MERGE_FILE}"
  --data-path "${DATA_PREFIX}"
  --split 99990,8,2
)

MODEL_ARGS=(
  --use-mcore-models
  --transformer-impl local
  --seq-length 2048
  --max-position-embeddings 2048
  --num-layers 8
  --hidden-size 1024
  --ffn-hidden-size 5120
  --num-attention-heads 8
  --disable-bias-linear
  --swiglu
)

RECOMPUTE_ARGS=(
  --recompute-granularity selective
  --recompute-modules mlp
)

MOE_ARGS=(
  --num-experts 32
  --moe-layer-freq 2
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --moe-router-topk 8
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 1e-2
  --moe-use-legacy-grouped-gemm
  --moe-grouped-gemm
  --moe-token-dispatcher-type alltoall
)

if [[ "${USE_MOE_CACHE}" == "1" ]]; then
  MOE_ARGS+=(--moe-enable-expert-weight-cache)
else
  MOE_ARGS+=(--no-moe-enable-expert-weight-cache)
fi

TRAINING_ARGS=(
  --micro-batch-size 10
  --global-batch-size 100
  --train-iters 1000
  --lr 1.0e-4
  --min-lr 1.0e-5
  --lr-decay-style cosine
  --lr-warmup-iters 100
  --weight-decay 0.1
  --clip-grad 1.0
  --bf16
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
)

LOGGING_ARGS=(
  --log-interval 10
  --save-interval 200
  --eval-interval 200
  --eval-iters 5
  --save "${SAVE_DIR}"
  --load "${SAVE_DIR}"
  --tensorboard-dir "${SAVE_DIR}/tensorboard"
  --no-load-optim
  --no-load-rng
)

torchrun "${DISTRIBUTED_ARGS[@]}" train.py \
  "${MODEL_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${RECOMPUTE_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${LOGGING_ARGS[@]}"
