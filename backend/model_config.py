"""
ACAI v4 — Central Model Configuration
======================================
Change ONE value here to swap the entire model stack.
Supports: Ollama (local), vLLM (GPU server), Anthropic API (cloud fallback).

CURRENT STACK (recommended, flexible):
  PRIMARY:    Qwen2.5-72B-Instruct   — best reasoning, 128K context, tool-use
  SPECIALIST: Jais-family-30B-chat   — Arabic-first, GCC-native, fine-tune base
  ROUTER:     Qwen3-8B               — fast planner, hybrid think/fast mode

TO SWITCH TO OLLAMA LOCAL (no GPU server needed):
  Set INFERENCE_BACKEND = "ollama"
  Set PRIMARY_MODEL = "qwen2.5:72b" (or any ollama model name)

TO SWITCH TO FULL JAIS STACK:
  Set PRIMARY_MODEL = "inceptionai/jais-family-70b-chat"
  Set SPECIALIST_MODEL = "inceptionai/jais-family-13b-chat"
  Set ROUTER_MODEL = "inceptionai/jais-family-7b-chat"

TO USE ANTHROPIC AS FALLBACK ONLY:
  Set FALLBACK_TO_ANTHROPIC = True
  Set ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
"""

import os
from dataclasses import dataclass
from typing import Optional


# ─── INFERENCE BACKEND ────────────────────────────────────────────────────────
# Options: "ollama" | "vllm" | "anthropic" | "huggingface"
INFERENCE_BACKEND: str = os.getenv("INFERENCE_BACKEND", "ollama")

# ─── MODEL SELECTION ──────────────────────────────────────────────────────────

# Primary reasoning engine (5-agent pipeline backbone)
PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "qwen2.5:14b-instruct-q4_K_M")

# Arabic specialist (dialect analysis, Bahraini fine-tune target)
SPECIALIST_MODEL: str = os.getenv("SPECIALIST_MODEL", "bahraini-pro:latest")

# Fast router/planner (MoA routing, query classification)
ROUTER_MODEL: str = os.getenv("ROUTER_MODEL", "qwen2.5:7b-instruct")

# Embedding model (for RAG vector search)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# ─── BACKEND ENDPOINTS ────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL: str      = os.getenv("VLLM_BASE_URL", "http://localhost:8080")
ANTHROPIC_API_KEY: str  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str    = "claude-sonnet-4-20250514"

# ─── FALLBACK CHAIN ───────────────────────────────────────────────────────────
# If primary backend fails, fall through this chain
FALLBACK_TO_ANTHROPIC: bool = os.getenv("FALLBACK_TO_ANTHROPIC", "true").lower() == "true"
FALLBACK_CHAIN: list = ["ollama", "vllm", "anthropic"]

# ─── GENERATION PARAMETERS ────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS: int      = 2048
DEFAULT_TEMPERATURE: float   = 0.3   # Lower = more factual, better for GRC
ROUTER_TEMPERATURE: float    = 0.1   # Very deterministic for routing
CREATIVE_TEMPERATURE: float  = 0.7   # For synthesis/summarization
CONTEXT_WINDOW: int          = 32768 # Safe default across all models

# ─── AGENT-TO-MODEL MAPPING ───────────────────────────────────────────────────
# Each agent can use a different model. Modify freely.
AGENT_MODELS: dict = {
    "planner":       ROUTER_MODEL,      # Fast, cheap
    "research":      PRIMARY_MODEL,     # Needs tool-use
    "reasoning":     PRIMARY_MODEL,     # Needs CoT
    "verification":  PRIMARY_MODEL,     # Needs accuracy
    "synthesis":     PRIMARY_MODEL,     # Needs quality output
    "arabic_nlp":    SPECIALIST_MODEL,  # Arabic-first model
    "knowledge":     PRIMARY_MODEL,     # Tool + reasoning
    "grc":           PRIMARY_MODEL,     # Needs reasoning + citations
    "hallucination": ROUTER_MODEL,      # Fast binary checker
    "dialect":       SPECIALIST_MODEL,  # Arabic specialist
}

# ─── DIALECT FINE-TUNE CONFIG ─────────────────────────────────────────────────
DIALECT_FINETUNE = {
    "base_model":    "inceptionai/jais-family-30b-chat",
    "lora_r":        16,
    "lora_alpha":    32,
    "lora_dropout":  0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "dataset":       "bahraini_twitter_500m",
    "output_dir":    "./models/jais-bahraini-v1",
    "quantize":      "4bit",   # QLoRA for university GPUs
}

# ─── DATACLASS FOR TYPE SAFETY ────────────────────────────────────────────────
@dataclass
class ModelConfig:
    backend: str
    primary: str
    specialist: str
    router: str
    embedding: str
    ollama_url: str
    vllm_url: str
    max_tokens: int
    temperature: float
    context_window: int
    fallback_to_anthropic: bool

def get_config() -> ModelConfig:
    return ModelConfig(
        backend=INFERENCE_BACKEND,
        primary=PRIMARY_MODEL,
        specialist=SPECIALIST_MODEL,
        router=ROUTER_MODEL,
        embedding=EMBEDDING_MODEL,
        ollama_url=OLLAMA_BASE_URL,
        vllm_url=VLLM_BASE_URL,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        context_window=CONTEXT_WINDOW,
        fallback_to_anthropic=FALLBACK_TO_ANTHROPIC,
    )
