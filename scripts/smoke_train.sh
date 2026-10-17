#!/bin/bash
# Single-GPU smoke run: a handful of real optimizer steps to check the loop end
# to end. Tiny LoRA rank and few inference steps so it finishes quickly.
#
# Usage:
#   FLUX=black-forest-labs/FLUX.1-dev bash scripts/smoke_train.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

python -u train.py \
    --pretrained_model_name_or_path "${FLUX:-black-forest-labs/FLUX.1-dev}" \
    --train_data_path "${DATA:-data/sample_train.jsonl}" \
    --output_dir "${OUT:-runs/smoke}" \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_train_steps "${STEPS:-6}" \
    --rank 8 --lora_alpha 8 \
    --num_inference_steps 8 \
    --guidance_scale 3.5 \
    --logging_steps 1 \
    --checkpointing_steps 1000000 \
    --seed 0 \
    "$@"
