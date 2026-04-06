"""
ACAI — Bahraini Dialect QLoRA Fine-Tuning Script
=================================================
Fine-tunes Qwen2.5-14B-Instruct on Bahraini Arabic dialect data
using QLoRA (4-bit quantized LoRA) optimized for 2x A100 GPUs.

This script is designed to run on the Hayrat cluster at UOB.
Submit via SLURM — see slurm_job.sh at the bottom.

Architecture:
  Base model:  Qwen/Qwen2.5-14B-Instruct (HuggingFace)
  Method:      QLoRA (4-bit NF4 + double quantization)
  LoRA rank:   16 (good balance quality/VRAM)
  VRAM usage:  ~2x A100 40GB (with gradient checkpointing)

Dataset format (JSONL):
  Each line: {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}
  OR:        {"instruction": "...", "input": "...", "output": "..."}

Expected results after 3 epochs on 50K samples:
  - Bahraini dialect fluency: significantly improved
  - Standard Arabic capability: preserved (LoRA only modifies ~0.3%)
  - Training time: ~6-8 hours on 2x A100 40GB

Usage:
  # On Hayrat lab server:
  sbatch slurm_job.sh

  # Or directly (single GPU):
  python bahraini_qlora_train.py --data_dir /data/datasets/$USER/acai/data

  # Test with small sample first:
  python bahraini_qlora_train.py --max_samples 100 --num_epochs 1
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("acai.train")

# ─── Constants ─────────────────────────────────────────────────────────────────

BASE_MODEL  = "Qwen/Qwen2.5-14B-Instruct"   # Change to "inceptionai/jais-family-13b-chat" for Jais
OUTPUT_DIR  = "/data/datasets/$USER/acai/models/acai-bahraini-v1"
DATA_DIR    = "/data/datasets/$USER/acai/data"
CACHE_DIR   = "/data/datasets/$USER/.cache/huggingface"

# ─── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Bahraini Arabic")
    p.add_argument("--base_model",   default=BASE_MODEL)
    p.add_argument("--output_dir",   default=OUTPUT_DIR)
    p.add_argument("--data_dir",     default=DATA_DIR)
    p.add_argument("--train_file",   default="bahraini_train.jsonl")
    p.add_argument("--eval_file",    default="bahraini_eval.jsonl")
    p.add_argument("--max_samples",  type=int, default=None,
                   help="Limit samples (useful for quick tests)")
    p.add_argument("--num_epochs",   type=int, default=3)
    p.add_argument("--batch_size",   type=int, default=2)
    p.add_argument("--grad_accum",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lora_r",       type=int, default=16)
    p.add_argument("--lora_alpha",   type=int, default=32)
    p.add_argument("--max_seq_len",  type=int, default=2048)
    p.add_argument("--no_merge",     action="store_true",
                   help="Skip merging LoRA into base model after training")
    p.add_argument("--resume_from",  default=None,
                   help="Resume from checkpoint directory")
    return p.parse_args()


# ─── Dataset Preparation ───────────────────────────────────────────────────────

def load_dataset_from_jsonl(filepath: str, max_samples: Optional[int] = None):
    """
    Load JSONL dataset. Supports two formats:

    Format 1 — ChatML (preferred):
      {"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}

    Format 2 — Instruction:
      {"instruction": "...", "input": "...", "output": "..."}
    """
    from datasets import Dataset

    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)

                # Format 1: already has "text" field
                if "text" in item:
                    samples.append({"text": item["text"]})

                # Format 2: instruction format
                elif "instruction" in item and "output" in item:
                    instruction = item["instruction"]
                    input_text  = item.get("input", "")
                    output      = item["output"]

                    if input_text:
                        user_msg = f"{instruction}\n\n{input_text}"
                    else:
                        user_msg = instruction

                    # Convert to ChatML format
                    text = (
                        f"<|im_start|>system\n"
                        f"أنت ACAI، مساعد ذكي متخصص في اللهجة البحرينية والخليجية. "
                        f"أجب بنفس لهجة المستخدم.<|im_end|>\n"
                        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                        f"<|im_start|>assistant\n{output}<|im_end|>"
                    )
                    samples.append({"text": text})

                # Format 3: messages list
                elif "messages" in item:
                    msgs = item["messages"]
                    text = ""
                    for msg in msgs:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    samples.append({"text": text.strip()})

            except json.JSONDecodeError:
                continue

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Loaded {len(samples)} samples from {filepath}")
    return Dataset.from_list(samples)


def prepare_datasets(args):
    """Load and validate training data."""
    train_path = os.path.join(args.data_dir, args.train_file)
    eval_path  = os.path.join(args.data_dir, args.eval_file)

    if not os.path.exists(train_path):
        logger.error(f"Training file not found: {train_path}")
        logger.error("Create it by running: python finetune_bahraini.py --step prepare")
        sys.exit(1)

    train_dataset = load_dataset_from_jsonl(train_path, max_samples=args.max_samples)
    eval_dataset  = load_dataset_from_jsonl(eval_path,  max_samples=min(200, args.max_samples or 200))

    # Show sample
    logger.info(f"\nSample training example:")
    logger.info(f"  {train_dataset[0]['text'][:200]}...\n")

    return train_dataset, eval_dataset


# ─── Model Setup ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args):
    """Load base model with 4-bit quantization for QLoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading base model: {args.base_model}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory // 1024**3
        logger.info(f"  GPU {i}: {name} ({mem}GB)")

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",             # NF4 format — best for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16, # BF16 compute on A100
        bnb_4bit_use_double_quant=True,        # Extra compression
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",                     # Spread across all available GPUs
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Important for causal LM

    logger.info(f"Model loaded. Parameters: {model.num_parameters() / 1e9:.1f}B")
    return model, tokenizer


def setup_lora(model, args):
    """Apply LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare for QLoRA (freezes base, enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # LoRA config — targets attention + MLP layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # MLP (feed-forward)
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected: trainable params: ~40M of 14B = 0.28%

    return model


# ─── Training ──────────────────────────────────────────────────────────────────

def train(args, model, tokenizer, train_dataset, eval_dataset):
    """Run SFT training."""
    from trl import SFTTrainer, SFTConfig

    output_dir = os.path.expandvars(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,

        # Epochs and batching
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Effective batch = batch_size * grad_accum * num_gpus
        # = 2 * 8 * 2 = 32

        # Optimizer
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",   # Memory-efficient optimizer

        # Precision
        fp16=False,
        bf16=True,                  # BF16 on A100 for better numerical stability

        # Logging & saving
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # W&B reporting (optional — remove if not using wandb)
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else ["none"],
        run_name="acai-bahraini-v1",

        # Data
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,

        # Resume
        resume_from_checkpoint=args.resume_from,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Starting training")
    logger.info(f"  Model:      {args.base_model}")
    logger.info(f"  Train size: {len(train_dataset)}")
    logger.info(f"  Eval size:  {len(eval_dataset)}")
    logger.info(f"  Epochs:     {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size} × {args.grad_accum} grad accum")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"{'='*50}\n")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ Training complete. Model saved to {output_dir}")

    return output_dir


# ─── Merge LoRA into Base Model ────────────────────────────────────────────────

def merge_and_save(output_dir: str, base_model: str):
    """
    Merge LoRA adapters back into the base model.
    Required for deployment — you can't use LoRA adapters without PEFT library.
    After merging, the model can be converted to GGUF and used with Ollama.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = output_dir + "-merged"
    logger.info(f"Merging LoRA adapters → {merged_dir}")

    # Load base model in FP16 for merging
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base, output_dir)

    # Merge adapters into base model
    model = model.merge_and_unload()

    # Save merged model
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    logger.info(f"✅ Merged model saved to {merged_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Convert to GGUF:  python llama.cpp/convert_hf_to_gguf.py {merged_dir}")
    logger.info(f"  2. Quantize:         ./llama.cpp/llama-quantize model.gguf model-q4km.gguf Q4_K_M")
    logger.info(f"  3. Load in Ollama:   ollama create acai-bahraini:v1 -f Modelfile")
    logger.info(f"  4. Update .env:      SPECIALIST_MODEL=acai-bahraini:v1")

    return merged_dir


# ─── Quick Test After Training ─────────────────────────────────────────────────

def quick_test(model_dir: str):
    """Run a quick inference test on the trained model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    logger.info("Running quick inference test...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )

    test_prompts = [
        "شلونك يا خوي؟",                          # Bahraini greeting
        "كيف أفتح حساب مصرفي في البحرين؟",          # Banking in Arabic
        "Explain the CBB rulebook briefly.",       # English GRC
    ]

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=150, temperature=0.7)

    for prompt in test_prompts:
        msg = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        result = pipe(msg)[0]["generated_text"]
        print(f"\nQ: {prompt}")
        print(f"A: {result[len(msg):][:200]}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Check dependencies
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    # Load data
    train_dataset, eval_dataset = prepare_datasets(args)

    # Load model + apply LoRA
    model, tokenizer = load_model_and_tokenizer(args)
    model = setup_lora(model, args)

    # Train
    output_dir = train(args, model, tokenizer, train_dataset, eval_dataset)

    # Merge (unless skipped)
    if not args.no_merge:
        merged_dir = merge_and_save(output_dir, args.base_model)
        quick_test(merged_dir)
    else:
        logger.info("Skipping merge (--no_merge flag set)")

    logger.info("\n✅ Fine-tuning pipeline complete!")


if __name__ == "__main__":
    main()


# ─── SLURM Job Script ──────────────────────────────────────────────────────────
SLURM_SCRIPT = '''#!/bin/bash
#SBATCH --job-name=acai-bahraini-finetune
#SBATCH --output=/data/datasets/%u/acai/logs/train_%j.out
#SBATCH --error=/data/datasets/%u/acai/logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# ─── Setup ────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Move to project directory
cd /data/datasets/$USER/acai

# Activate conda environment
conda activate acai-train

# Set cache directories (avoid filling /home)
export HF_HOME=/data/datasets/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/data/datasets/$USER/.cache/huggingface/transformers
export TORCH_HOME=/data/datasets/$USER/.cache/torch

# Create log directory
mkdir -p /data/datasets/$USER/acai/logs

# ─── Install dependencies if needed ──────────────────────────────────────────
pip install -q transformers peft trl bitsandbytes accelerate datasets

# ─── Run training ─────────────────────────────────────────────────────────────
python bahraini_qlora_train.py \\
    --data_dir /data/datasets/$USER/acai/data \\
    --output_dir /data/datasets/$USER/acai/models/acai-bahraini-v1 \\
    --num_epochs 3 \\
    --batch_size 2 \\
    --grad_accum 8 \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --max_seq_len 2048

echo "Job finished: $(date)"
'''

# Write SLURM script to file if run directly
if __name__ != "__main__":
    slurm_path = Path(__file__).parent.parent.parent / "slurm_finetune.sh"
    if not slurm_path.exists():
        with open(slurm_path, 'w') as f:
            f.write(SLURM_SCRIPT)
'''

# ─── SETUP COMMANDS FOR HAYRAT LAB ────────────────────────────────────────────
SETUP_COMMANDS = """
=================================================================
HAYRAT LAB SETUP — Run these commands once after SSH login
=================================================================

# 1. Go to your data directory (NOT home — only 5GB!)
cd /data/datasets/$USER/acai

# 2. Create conda environment for training
conda create -n acai-train python=3.11 -y
conda activate acai-train

# 3. Install PyTorch with CUDA (check lab CUDA version first)
nvidia-smi | grep "CUDA Version"
# If CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install training packages
pip install transformers peft trl bitsandbytes accelerate datasets
pip install sentencepiece protobuf

# 5. Test GPU access
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"

# 6. Prepare your data
mkdir -p /data/datasets/$USER/acai/data
mkdir -p /data/datasets/$USER/acai/models
mkdir -p /data/datasets/$USER/acai/logs

# 7. Upload your data from Windows
# On Windows PowerShell:
# scp C:\\Users\\fatim\\arabic-cognitive-ai\\data\\*.jsonl user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/data/

# 8. Submit the training job
sbatch slurm_finetune.sh

# 9. Monitor
squeue --me           # Check job status
tail -f logs/train_*.out   # Watch training output
nvidia-smi            # GPU usage (on compute node)
=================================================================
"""
