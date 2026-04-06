#!/bin/bash
#SBATCH --job-name=acai-bahraini-finetune
#SBATCH --output=/data/datasets/%u/acai/logs/train_%j.out
#SBATCH --error=/data/datasets/%u/acai/logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# ─── Confirm what we have ─────────────────────────────────────────────────────
echo "=============================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "=============================="

nvidia-smi

# ─── Go to project directory ──────────────────────────────────────────────────
cd /data/datasets/$USER/acai

# ─── Activate conda env ───────────────────────────────────────────────────────
source /opt/conda/etc/profile.d/conda.sh
conda activate acai-train

# ─── Set cache dirs (avoid filling /home — 5GB limit!) ────────────────────────
export HF_HOME=/data/datasets/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/data/datasets/$USER/.cache/huggingface/transformers
export TORCH_HOME=/data/datasets/$USER/.cache/torch
export XDG_CACHE_HOME=/data/datasets/$USER/.cache
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $TORCH_HOME

# ─── Run training ─────────────────────────────────────────────────────────────
python bahraini_qlora_train.py \
    --base_model    "Qwen/Qwen2.5-14B-Instruct" \
    --data_dir      /data/datasets/$USER/acai/data \
    --output_dir    /data/datasets/$USER/acai/models/acai-bahraini-v1 \
    --num_epochs    3 \
    --batch_size    2 \
    --grad_accum    8 \
    --lr            2e-4 \
    --lora_r        16 \
    --lora_alpha    32 \
    --max_seq_len   2048

echo "=============================="
echo "Finished:  $(date)"
echo "=============================="
