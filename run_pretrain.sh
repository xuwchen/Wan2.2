#!/bin/bash
# =============================================================================
# Megatron-FSDP Pretrain Script for Wan2.2-TI2V-5B (Mock Data Only)
# =============================================================================
# Model: Wan-AI/Wan2.2-TI2V-5B (Dense 5B parameters)
# Uses mock random data for testing the training pipeline.
# =============================================================================

set -e

# ==================== Megatron-LM Path ====================
MEGATRON_PATH="${MEGATRON_PATH:-/lustre/fsw/portfolios/coreai/users/xuwenc/code/wan22/Megatron-LM}"

if [ -d "${MEGATRON_PATH}" ]; then
    export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"
    echo "Megatron-LM: ${MEGATRON_PATH}"
else
    echo "ERROR: Megatron-LM not found at ${MEGATRON_PATH}"
    exit 1
fi

# ==================== Install Dependencies ====================
if [ "${SKIP_DEPS_INSTALL:-false}" != "true" ]; then
    echo "Installing dependencies..."
    pip install -q easydict diffusers ftfy regex sentencepiece transformers accelerate --root-user-action=ignore 2>/dev/null || true
fi

# ==================== Configuration ====================
# Model checkpoint (download: huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/xuwenc/code/wan22/Wan2.2-TI2V-5B}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/wan22_pretrain}"

# ==================== Training Config (User-Friendly) ====================
# Set these 3 values to control training:
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"       # Per-GPU batch size (limited by GPU memory)
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"     # Total samples per optimizer step
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"         # Total training steps

# Distributed config
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_GPUS=$((NNODES * NPROC_PER_NODE))

# Auto-calculate gradient accumulation and num_samples
GRADIENT_ACCUMULATION=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_GPUS)))
NUM_SAMPLES=$((NUM_ITERATIONS * GLOBAL_BATCH_SIZE))

# Validation
EXPECTED_GLOBAL=$((MICRO_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))
if [ ${EXPECTED_GLOBAL} -ne ${GLOBAL_BATCH_SIZE} ]; then
    echo "ERROR: global_batch_size (${GLOBAL_BATCH_SIZE}) must be divisible by (micro_batch_size × num_gpus)"
    echo "  micro_batch_size=${MICRO_BATCH_SIZE}, num_gpus=${NUM_GPUS}"
    echo "  Valid global_batch_size: $((MICRO_BATCH_SIZE * NUM_GPUS)), $((MICRO_BATCH_SIZE * NUM_GPUS * 2)), $((MICRO_BATCH_SIZE * NUM_GPUS * 4)), ..."
    exit 1
fi

if [ ${GRADIENT_ACCUMULATION} -lt 1 ]; then
    echo "ERROR: gradient_accumulation must be >= 1"
    echo "  Increase global_batch_size or decrease micro_batch_size"
    exit 1
fi

# ==================== Other Hyperparameters ====================
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

# Video config (smaller for 80GB GPU with ZeRO-1)
# ZeRO-1 keeps full model on each GPU, so we need smaller resolution
# seq_len = (frames/4) × (H/16) × (W/16)
# 17 frames, 256×256: seq_len = 5 × 16 × 16 = 1280
FRAME_NUM="${FRAME_NUM:-17}"
RESOLUTION="${RESOLUTION:-256 256}"

# Network
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

# Megatron-FSDP
DP_OUTER_SIZE="${DP_OUTER_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
# Note: fuse_wgrad_accumulation requires ZeRO-1 (optim), NOT ZeRO-3 (optim_grads_params)
# ZeRO-3 shards parameters which breaks cuBLAS GEMM for wgrad accumulation
ZERO_DP_STRATEGY="${ZERO_DP_STRATEGY:-optim_grads_params}"
OUTER_DP_STRATEGY="${OUTER_DP_STRATEGY:-no_shard}"
PARAM_DTYPE="${PARAM_DTYPE:-bf16}"

# TransformerEngine (gradient_accumulation_fusion)
# TE is installed system-wide via: pip install . --no-build-isolation
USE_TE_LINEAR="${USE_TE_LINEAR:-true}"   # Replace nn.Linear with TE Linear
# NOTE: fuse_wgrad_accumulation is INCOMPATIBLE with FSDP!
# TE's internal wgrad GEMM path uses cuBLAS algorithms that don't work with FSDP tensor layouts.
# If you need fuse_wgrad, you must use Megatron-Core DDP + DistributedOptimizer instead of FSDP.
TE_FUSE_WGRAD="${TE_FUSE_WGRAD:-true}"  # Enable TE gradient accumulation fusion (Megatron-FSDP overwrite_main_grad=False)

# Checkpointing
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_STEPS="${LOG_STEPS:-10}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# ==================== Launch ====================
echo "=============================================="
echo "Wan2.2-TI2V-5B Pretrain (Mock Data)"
echo "=============================================="
echo "Training Config:"
echo "  micro_batch_size:    ${MICRO_BATCH_SIZE}"
echo "  global_batch_size:   ${GLOBAL_BATCH_SIZE} (= ${MICRO_BATCH_SIZE} × ${NUM_GPUS} × ${GRADIENT_ACCUMULATION})"
echo "  num_iterations:      ${NUM_ITERATIONS}"
echo "  num_samples:         ${NUM_SAMPLES} (auto-calculated)"
echo "  gradient_accum:      ${GRADIENT_ACCUMULATION} (auto-calculated)"
echo ""
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NNODES}×${NPROC_PER_NODE} = ${NUM_GPUS}"
echo "Frame: ${FRAME_NUM}, Resolution: ${RESOLUTION}"
echo "ZeRO Strategy: ${ZERO_DP_STRATEGY}"
echo "TE Linear (grad_accum_fusion): ${USE_TE_LINEAR}"
echo "=============================================="

CMD="torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    pretrain_t2v.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_steps ${NUM_ITERATIONS} \
    --batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --num_samples ${NUM_SAMPLES} \
    --frame_num ${FRAME_NUM} \
    --resolution ${RESOLUTION} \
    --dp_outer_size ${DP_OUTER_SIZE} \
    --cp_size ${CP_SIZE} \
    --tp_size ${TP_SIZE} \
    --zero_dp_strategy ${ZERO_DP_STRATEGY} \
    --outer_dp_strategy ${OUTER_DP_STRATEGY} \
    --param_dtype ${PARAM_DTYPE} \
    --save_steps ${SAVE_STEPS} \
    --log_steps ${LOG_STEPS} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --preserve_fp32_weights \
    --overlap_grad_reduce \
    --overlap_param_gather"

# Optional args
[ -n "${DP_SHARD_SIZE}" ] && CMD="${CMD} --dp_shard_size ${DP_SHARD_SIZE}"
[ -n "${RESUME_FROM}" ] && CMD="${CMD} --resume_from ${RESUME_FROM}"
[ "${INIT_ON_META_DEVICE}" = "true" ] && CMD="${CMD} --init_on_meta_device"
[ "${USE_TE_LINEAR}" = "true" ] && CMD="${CMD} --use_te_linear"
[ "${TE_FUSE_WGRAD}" = "true" ] && CMD="${CMD} --te_fuse_wgrad"
[ "${GRAD_REDUCE_FP32}" = "true" ] && CMD="${CMD} --grad_reduce_in_fp32"

echo ""
echo "Command: ${CMD}"
echo ""

cd "$(dirname "$0")"
exec ${CMD}
