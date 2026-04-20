<div align="center">

# ACAI — Arabic Cognitive AI Engine
### محرك الذكاء الاصطناعي المعرفي العربي

**Private · On-Premise · Arabic-First · Research-Grade**

*University of Bahrain · College of Information Technology · Benefit AI Lab · 2026*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react)](https://react.dev)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

## Overview

ACAI is a production-grade, fully private Arabic AI platform built for Bahrain and the GCC. It enables users to interact with a cognitive multi-agent system using natural language in Arabic or English — with **zero data leaving the machine**.

No OpenAI. No cloud APIs. No data sovereignty risk.

The system is the first to specifically target **Bahraini Arabic dialect** — providing dialect identification, morphological analysis, MSA normalisation, and GCC regulatory knowledge in a single unified platform.

## Key Features

### 🧠 Multi-Agent Cognitive Pipeline
Six specialist agents, each with a distinct role. The orchestrator routes every query automatically — no manual agent selection.

| Agent | Arabic | Role |
|---|---|---|
| Researcher | باحث | Web search with source citations — zero hallucination policy |
| Reasoner | حكيم | 5-step chain-of-thought for complex analysis |
| GCC Advisor | مشير | CBB, SAMA, UAECB regulations and Vision 2030 |
| Arabic Expert | لغوي | Dialect identification, morphology, MSA normalisation |
| Fact Checker | مراقب | Claim verification — always runs last |
| Knowledge Builder | بانِ | Entity and relation extraction |

### 🗣️ Bahraini Arabic Specialisation
- Automatic dialect detection (Bahraini, Gulf, Saudi, Egyptian, Levantine, MSA)
- Morphological analysis — root → pattern → weight → meaning
- Code-switching detection (Arabic + English mixed input)
- 150-question Bahraini dialect benchmark (first of its kind)
- Fine-tuning pipeline for dialect adaptation (QLoRA on Hayrat A100)

### 📚 RAG — Document Intelligence
- Ingest CBB circulars, internal policies, regulatory documents
- AI answers grounded in your actual documents with citations
- SQLite FTS5 retrieval — no external vector database required
- Source cited in every regulatory response

### 🧠 Persistent Cross-Session Memory
- SQLite FTS5 full-text search over all past conversations
- Relevant context injected before every query automatically
- Survives server restarts — no Redis required
- Auto-generates reusable skill files from high-quality responses

### 🔒 Security
- All API keys in server `.env` only — zero secrets in browser
- JWT-style API key middleware on every endpoint
- Rate limiting (40 requests/min per IP)
- CORS handled at middleware level
- `.env` protected by `.gitignore`

### 📊 Novel Evaluation Metrics
- **DCR (Dialect Control Rate)** — measures dialect generation fidelity
- **MLR (MSA Leak Rate)** — measures formal Arabic contamination in dialect output
- Neither metric exists in any prior Arabic NLP publication

## Benchmark Results

*Evaluated April 2026 on Qwen2.5-14B running locally (CPU, no GPU)*

| Category | Base Model | After Fine-Tuning |
|---|---|---|
| MSA Normalisation | **100%** | 100% |
| Banking in Dialect | **100%** | 100% |
| Vocabulary | **90%** | ~95% |
| Dialect Identification | 50% | **~75% (target)** |
| Morphological Analysis | 40% | **~65% (target)** |
| **Overall Bahraini** | **76%** | **TBD** |
| **ABBL** | **87.5%** | TBD |

**Comparison with cloud models:**

| Model | ABBL | Deployment |
|---|---|---|
| GPT-4o | ~72% | ☁️ Cloud — sovereignty risk |
| Jais-30B | ~65% | ☁️ Cloud — sovereignty risk |
| **Qwen2.5-14B (ACAI)** | **87.5%** |  Local — data sovereign |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                React 18 Frontend (port 5173)                  │
│         Arabic RTL · Dark/Light Mode · Streaming SSE          │
│              Zero API keys in browser code                    │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼─────────────────────────────────────┐
│            FastAPI Gateway (port 8000)                        │
│   Auth · Rate Limiting · CORS · Audit · Health               │
└────┬──────────────┬───────────────┬──────────────────────────┘
     │              │               │
┌────▼────┐  ┌──────▼──────┐  ┌────▼────────────────────────┐
│ SQLite  │  │ Minimal RAG │  │     Orchestrator Engine      │
│ FTS5    │  │ CBB/SAMA    │  │  Intent → Pipeline → Merge   │
│ Memory  │  │ Doc Chunks  │  │  باحث·مشير·لغوي·حكيم·مراقب  │
└─────────┘  └─────────────┘  └────────────────┬────────────┘
                                                │
                               ┌────────────────▼────────────┐
                               │   Ollama (local inference)   │
                               │  qwen2.5:14b · bahraini-pro  │
                               └─────────────────────────────┘
```

## Tech Stack

### Backend
| Component | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| FastAPI | 0.115 | API framework |
| Uvicorn | 0.30 | ASGI server |
| SQLite FTS5 | built-in | Memory + RAG store |
| httpx | 0.27 | Async HTTP client |
| python-dotenv | 1.0 | Environment config |
| duckduckgo-search | 6.3 | Free web search |

### Frontend
| Component | Version | Purpose |
|---|---|---|
| React | 18.3 | UI framework |
| Vite | 5.3 | Build tool |
| Scheherazade New | font | Arabic typography |
| JetBrains Mono | font | Code display |

### Infrastructure
| Component | Purpose |
|---|---|
| Ollama | Local LLM inference (GPU or CPU) |
| qwen2.5:14b-instruct-q4_K_M | Primary reasoning model |
| bahraini-pro (fine-tuned) | Dialect specialist model |
| Hayrat A100 Cluster | QLoRA fine-tuning (University of Bahrain) |


## Project Structure

```
arabic-cognitive-ai/
├── backend/
│   ├── main_v5.py              # FastAPI app — orchestrator, memory, RAG, security
│   ├── orchestrator_v2.py      # Policy-based agent router with intent classifier
│   ├── orchestrator_logic.py   # Pure classification logic (no external deps)
│   ├── acai_memory.py          # SQLite FTS5 cross-session memory
│   ├── skill_generator.py      # Auto-generate agentskills.io skill files
│   ├── requirements.txt        # Pinned Python dependencies
│   └── .env.example            # Environment template
├── frontend-new/
│   └── src/App.jsx             # React 18 UI — all calls through backend
├── eval/
│   ├── experiments.py          # Benchmark runner — DCR, MLR, paper metrics
│   ├── bahraini_benchmark.py   # 150-question Bahraini dialect benchmark
│   └── memory_experiment.py    # Before/after memory experiment
├── tests/
│   └── test_acai.py            # 13 unit and integration tests (all passing)
├── paper/
│   └── acai_paper.tex          # ACL-format paper draft
├── data/
│   └── lughawi_train.jsonl     # Bahraini dialect training dataset
├── results/                    # Benchmark JSON outputs
├── slurm_finetune.sh           # QLoRA training job for Hayrat cluster
└── acai_lab_setup.sh           # Lab GPU deployment script
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- 16 GB RAM minimum

### 1. Pull the model
```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

### 2. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Edit .env — set API_KEY
uvicorn main_v5:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend
```bash
cd frontend-new
npm install
npm run dev
```

Open `http://localhost:5173`

## Environment Configuration

```env
# backend/.env
API_KEY=your-secret-key-here
PRIMARY_MODEL=qwen2.5:14b-instruct-q4_K_M
SPECIALIST_MODEL=bahraini-pro:latest
OLLAMA_BASE_URL=http://localhost:11434
ANTHROPIC_API_KEY=              # Optional — enables live web search for باحث
```

The `ANTHROPIC_API_KEY` is optional. Without it, the Researcher agent falls back to DuckDuckGo.

## API Reference

All endpoints require `X-API-Key` header. Health check is public.

### Chat (Streaming)
```http
POST /api/query/stream
X-API-Key: your-key
Content-Type: application/json

{
  "query": "ما أحدث أنظمة CBB للفنتك؟",
  "mode": "auto",
  "session_id": "user-session-1"
}
```

Response: Server-Sent Events stream

### Chat (Sync)
```http
POST /api/query
X-API-Key: your-key

{ "query": "شلون أفتح حساب؟", "mode": "auto" }
```

### Document Ingestion
```http
POST /api/rag/ingest
X-API-Key: your-key

{ "text": "...", "doc_name": "CBB_Circular_2025" }
```

### Health Check
```http
GET /api/health
# No auth required
```

### Full Endpoint List
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System status, models, memory stats |
| `/api/query/stream` | POST | Streaming orchestrated response |
| `/api/query` | POST | Non-streaming response |
| `/api/search` | POST | Web search (backend-only) |
| `/api/rag/ingest` | POST | Ingest document into RAG |
| `/api/rag/docs` | GET | List ingested documents |
| `/api/rag/retrieve` | POST | Retrieve relevant chunks |
| `/api/memory/stats` | GET | Memory database statistics |
| `/api/memory/search` | POST | Search past conversations |
| `/api/eval/dcr` | GET | Run DCR + MLR evaluation |

## Bahraini Dialect Support

The `لغوي` agent provides structured linguistic analysis for any Arabic input:

| Input | Analysis Provided |
|---|---|
| Dialect | Bahraini · Gulf · Saudi · Egyptian · Levantine · MSA |
| Confidence | Percentage confidence per classification |
| Morphology | Root → Pattern → Weight → Meaning (3 sample words) |
| Normalisation | Dialect → MSA equivalent |
| Code-switching | Arabic-English boundary detection |

**Sample dialect patterns handled:**

| Bahraini | MSA | English |
|---|---|---|
| وايد | كثير | very / a lot |
| حيل | جداً | extremely |
| الحين | الآن | now |
| مب / مو | ليس | not |
| صج | صحيح | true / really |
| تره | اعلم | note that |
| خوي | أخي | my brother (friendly) |
| باكر | غداً | tomorrow |
| شلون | كيف | how |
| وين | أين | where |

## Running Tests

```bash
python tests/test_acai.py -v
```

**13/13 tests passing:**
- Memory: save, retrieve, stats
- Orchestrator: intent routing, pipeline building, output merging
- Skill generator: quality threshold, step extraction
- Security: no keys in frontend

## Running Benchmarks

```bash
# Bahraini dialect benchmark (50 questions)
python eval/experiments.py --benchmark

# Dialect Control Rate + MSA Leak Rate
python eval/experiments.py --dcr

# Before/after memory experiment
python eval/memory_experiment.py

# Full paper table
python eval/experiments.py --all
```

Results saved to `results/` as JSON.

## Fine-Tuning

The `لغوي` agent is being fine-tuned on Bahraini dialect data using QLoRA on the University of Bahrain's Hayrat GPU cluster.

**Training configuration:**
- Base model: Qwen/Qwen2.5-14B-Instruct
- Method: QLoRA (4-bit NF4 + LoRA rank 16)
- Hardware: 2× A100 40GB
- Dataset: 10,500 curated samples + ~4,900 knowledge-distilled from 32B teacher
- Training time: ~6-8 hours

**Dataset categories:**
- Dialect identification (2,500 samples)
- Vocabulary in context (2,000 samples)
- MSA normalisation (2,000 samples)
- Morphological analysis (1,500 samples)
- Banking dialect (1,000 samples)
- Knowledge distillation from 32B teacher (~4,900 samples)

See `slurm_finetune.sh` for the complete training job.

## Research Context

**Target venue:** ArabicNLP 2026 workshop (co-located with EMNLP 2026)

**Novel contributions:**
1. First Bahraini Arabic evaluation benchmark (150 questions, 6 categories)
2. Dialect Control Rate (DCR) — new generation-level metric
3. MSA Leak Rate (MLR) — complementary dialect fidelity metric
4. Persistent cross-session memory with auto-skill generation
5. 87.5% ABBL accuracy with local deployment — surpasses cloud baselines

**Key references:**
- DialectalArabicMMLU (Oct 2025) — covers Syrian, Egyptian, Emirati, Saudi, Moroccan — Bahraini absent
- Evaluating Arabic LLMs Survey (Oct 2025) — 40+ benchmarks, critical gaps identified
- Fine-Tuning Arabic LLMs, PLOS ONE (Feb 2026) — validates knowledge distillation approach
- Hermes Agent, Nous Research (Feb 2026) — persistent memory architecture reference

## Security Notes

Before any institutional deployment:

- Change `API_KEY` in `.env` to a strong random value
- Never commit `.env` (protected by `.gitignore`)
- All AI API keys live server-side only — frontend has zero secrets
- Rate limiting prevents abuse (40 req/min per IP)

## Acknowledgements

University of Bahrain · College of Information Technology · Benefit AI Lab

Built with FastAPI, React, SQLite, Ollama, and the open-source Arabic NLP community.

## License

MIT License.
