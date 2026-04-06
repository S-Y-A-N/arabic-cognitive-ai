# ACAI Experiments Plan
## "Arabic Cognitive OS with Bahraini Dialect Specialization"
### Target venue: ACL / EMNLP 2025-2026

---

## What You Are Claiming (Your Novel Contribution)

> **ACAI is the first Arabic Cognitive OS that combines multi-agent reasoning, GraphRAG, and Bahraini dialect specialization into a single system — evaluated on ABBL, MADAR, and GCC policy benchmarks.**

This is novel because:
- No existing system combines multi-agent + Arabic GraphRAG + Bahraini dialect
- Most Arabic NLP papers do ONE thing (dialect OR RAG OR multi-agent)
- You have a working prototype + Bahraini-pro model + real GCC domain data

---

## Experiment Table (for the paper)

| Model | ABBL Acc | ABBL F1 | Bahraini Dial% | MSA Leak% | Avg Latency |
|-------|----------|---------|----------------|-----------|-------------|
| GPT-4o (baseline) | ~72% | ~70% | low | high | 1200ms |
| Jais-30B (baseline) | ~65% | ~63% | medium | medium | 800ms |
| Qwen2.5-14B base | TBD | TBD | low | high | 600ms |
| **ACAI-Qwen-14B (no finetune)** | TBD | TBD | low | high | 900ms |
| **ACAI-Qwen-14B + QLoRA (ours)** | TBD | TBD | **high** | **low** | 900ms |
| **ACAI-Qwen-14B + QLoRA + RAG (ours)** | TBD | TBD | **high** | **low** | 1100ms |

**Your goal:** Show that ACAI+QLoRA+RAG beats Jais-30B on ABBL AND has better Bahraini dialect control.

---

## Exact Steps to Run on A100s

### Step 1 — Data Preparation (your laptop, this week)

```bash
# Collect at minimum 5,000 Bahraini dialect samples
# Minimum viable dataset sizes:
#   Quick test:  1,000 samples   → 30 min training
#   Paper-ready: 50,000 samples  → 6 hours training
#   Strong:      200,000 samples → 24 hours training

# Data sources (priority order):
# 1. Your existing bahraini-pro training data ← use all of it
# 2. Generate synthetic data using this script:
python finetune_bahraini.py --step prepare --data_dir ./data

# 3. MADAR corpus (email camel@nyu.edu for access)
# 4. Bahraini news scraping (Al Ayam, Akhbar Al Khaleej)
```

**Minimum to start training:** 5,000 samples (will show dialect improvement, enough for paper)

**Target for strong paper:** 50,000-100,000 samples

---

### Step 2 — Upload to Lab (from Windows)

```powershell
# On your Windows machine:
scp C:\Users\fatim\arabic-cognitive-ai\data\*.jsonl user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/data/
scp C:\Users\fatim\arabic-cognitive-ai\backend\arabic\bahraini_qlora_train.py user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/
scp C:\Users\fatim\arabic-cognitive-ai\slurm_finetune.sh user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/
```

---

### Step 3 — Lab Setup (SSH, one time)

```bash
ssh user151@hayrat.uob.edu.bh
cd /data/datasets/$USER/acai

# Create conda env
conda create -n acai-train python=3.11 -y
conda activate acai-train

# Install (check CUDA version first with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.1 peft==0.14.0 trl==0.12.1
pip install bitsandbytes==0.45.0 accelerate==1.2.1 datasets==3.2.0
pip install sentencepiece protobuf wandb

# Test GPU
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

---

### Step 4 — Quick Test Run (before full training)

```bash
# ALWAYS test with 100 samples first to catch errors
conda activate acai-train
python bahraini_qlora_train.py \
    --max_samples 100 \
    --num_epochs 1 \
    --batch_size 1 \
    --no_merge

# If this works without errors → submit full job
```

---

### Step 5 — Full Training via SLURM

```bash
# Submit job
sbatch slurm_finetune.sh

# Monitor
squeue --me
tail -f /data/datasets/$USER/acai/logs/train_*.out

# Expected output every 25 steps:
# {'loss': 2.45, 'learning_rate': 0.0002, 'epoch': 0.05}
# Loss should decrease from ~2.5 to ~1.0 by end of training
```

---

### Step 6 — Benchmark Evaluation (after training)

```bash
# On your laptop (after training completes):
# 1. Download the model
scp -r user151@hayrat.uob.edu.bh:/data/datasets/user151/acai/models/acai-bahraini-v1-merged ./models/

# 2. Convert to GGUF for Ollama
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./models/acai-bahraini-v1-merged --outtype q4_k_m

# 3. Load into Ollama
ollama create acai-bahraini:v1 -f Modelfile

# 4. Run benchmarks
python eval/benchmark_harness.py --all --model acai-bahraini:v1
python eval/benchmark_harness.py --all --model qwen2.5:14b-instruct-q4_K_M

# 5. Compare results → fill in the experiment table above
```

---

## Hyperparameters Summary

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | Qwen2.5-14B-Instruct | Best Arabic reasoning at this size |
| Quantization | 4-bit NF4 | QLoRA — fits 2x A100 40GB |
| LoRA rank r | 16 | Good quality/VRAM balance |
| LoRA alpha | 32 | 2x rank = standard scaling |
| Target modules | q,k,v,o,gate,up,down | Full attention + MLP |
| Batch size | 2 per GPU × 8 grad accum | Effective batch = 32 |
| Learning rate | 2e-4 | Standard for QLoRA |
| Epochs | 3 | Enough for dialect adaptation |
| Max seq len | 2048 | Covers most Arabic conversations |
| Optimizer | paged_adamw_8bit | Memory efficient |
| Scheduler | cosine | Standard for LLM fine-tuning |

---

## Dialect Metrics (Novel Contribution)

These metrics are specific to YOUR paper and distinguish it from generic Arabic NLP work:

**Metric 1: Dialect Control Rate (DCR)**
```
DCR = (responses in correct dialect) / (total dialect queries)
```
Measure using lexical markers (same as benchmark_harness.py detect_dialect)

**Metric 2: MSA Leak Rate (MLR)**
```
MLR = (responses with ≥2 MSA markers when dialect expected) / total
```
Lower is better. Target: < 15% for your fine-tuned model.

**Metric 3: Dialect Fluency Score (DFS)**
```
DFS = human evaluation (1-5 scale) on 50 Bahraini responses
```
Ask 3 Bahraini Arabic speakers to rate naturalness 1-5.

---

## Timeline (4 weeks)

| Week | Task | Deliverable |
|------|------|-------------|
| Week 1 | Data collection + benchmark harness | 5K+ Bahraini samples, eval script running |
| Week 2 | QLoRA training on A100 | Trained model checkpoint |
| Week 3 | GraphRAG live (Weaviate + Neo4j) + evaluation | Benchmark results table |
| Week 4 | Paper writing + comparison table | ACL paper draft |

---

## Files You Need to Run

```
arabic-cognitive-ai/
├── eval/
│   ├── benchmark_harness.py      ← Run evaluations
│   └── data/
│       ├── abbl_extra.jsonl       ← Add more ABBL samples here
│       └── bahraini_custom.jsonl  ← Your Bahraini test set
├── backend/
│   ├── arabic/
│   │   └── bahraini_qlora_train.py  ← Training script
│   └── rag/
│       └── graphrag_impl.py          ← Live GraphRAG
├── data/
│   ├── bahraini_train.jsonl          ← Training data (50K+)
│   └── bahraini_eval.jsonl           ← Eval data (500+)
└── slurm_finetune.sh                 ← Submit to Hayrat cluster
```
