#!/bin/bash
# EruDiff on FLUX.1-dev: multi-GPU training run.
#
# Paper setting: LoRA rank 128, lr 5e-5, 28 inference steps, guidance 3.5,
# effective batch 64, 50k steps on 8 GPUs. The effective batch is
# train_batch_size * gradient_accumulation_steps * num_processes, and ACCUM is
# derived below so it stays at 64 for any GPU count.
#
# NO-RL is off unless --reward_model_path is passed, which trains with DK-DM
# alone (the paper does the same where no reward annotator is available).
#
# Usage:
#   FLUX=black-forest-labs/FLUX.1-dev DATA=data/train.jsonl bash scripts/train.sh
#   GPUS=0,1 STEPS=4000 bash scripts/train.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(awk -F, '{print NF}' <<< "$GPUS")"

FLUX="${FLUX:-black-forest-labs/FLUX.1-dev}"
DATA="${DATA:-data/train.jsonl}"
OUT="${OUT:-runs/erudiff}"

BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-$(( 64 / (BATCH * NPROC) ))}"

echo "GPUs=$GPUS (n=$NPROC)  batch=$BATCH  accum=$ACCUM  effective=$(( BATCH * ACCUM * NPROC ))"

export CUDA_VISIBLE_DEVICES="$GPUS"

accelerate launch \
    --num_processes "$NPROC" \
    $( [ "$NPROC" -gt 1 ] && echo --multi_gpu ) \
    --mixed_precision bf16 \
    train.py \
    --pretrained_model_name_or_path "$FLUX" \
    --train_data_path "$DATA" \
    --output_dir "$OUT" \
    --resolution 512 \
    --train_batch_size "$BATCH" \
    --gradient_accumulation_steps "$ACCUM" \
    --max_train_steps "${STEPS:-50000}" \
    --learning_rate 5e-5 \
    --adam_weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --rank 128 \
    --lora_alpha 128 \
    --num_inference_steps 28 \
    --guidance_scale 3.5 \
    --curriculum_decay 0.1 \
    --p_impl 0.8 --p_expl 0.1 --p_found 0.1 \
    --eman_decay 0.99 \
    --seed 42 \
    --logging_steps 10 \
    --checkpointing_steps 500 \
    --checkpoints_total_limit 3 \
    "$@"
