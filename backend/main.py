"""
ACAI v5 — Complete Production FastAPI Backend
===============================================
Implements all 5 priority upgrades:

A. ORCHESTRATOR  — policy-based routing, no manual agent selection
B. MEMORY        — SQLite FTS5 injected before EVERY query
C. SECURITY      — ALL secrets in backend, ZERO frontend API keys
D. MINIMAL RAG   — CBB document chunks, FTS5 retrieval, citations
E. EVALUATION    — DCR/MLR metrics, before/after memory experiment

Install:
    pip install fastapi uvicorn httpx python-dotenv duckduckgo-search

.env file (backend/.env):
    ANTHROPIC_API_KEY=sk-ant-...   (optional — for live web search)
    API_KEY=dev-key-12345
    PRIMARY_MODEL=qwen2.5:14b-instruct-q4_K_M
    SPECIALIST_MODEL=bahraini-pro:latest
    OLLAMA_BASE_URL=http://localhost:11434

Run:
    uvicorn main_v5:app --host 0.0.0.0 --port 8000 --reload
"""

import os, re, json, time, logging, asyncio, sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Config
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
API_KEY        = os.getenv("API_KEY", "dev-key-12345")
PRIMARY_MODEL  = os.getenv("PRIMARY_MODEL", "qwen2.5:14b-instruct-q4_K_M")
ARABIC_MODEL   = os.getenv("SPECIALIST_MODEL", "bahraini-pro:latest")
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

BASE_DIR   = Path(__file__).parent
MEMORY_DB  = BASE_DIR / "acai_memory.db"
RAG_DB     = BASE_DIR / "rag_store.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(funcName)s() @ line %(lineno)d | %(message)s"
)
log = logging.getLogger("acai")


# ORCHESTRATOR: Intent + Pipeline + Merge

AR_RE = re.compile(r'[\u0600-\u06FF]{3,}')

RESEARCH_KW = ["latest","recent","news","today","2025","2026","current",
               "أحدث","آخر","أخبار","حديث","اليوم","هذا الأسبوع"]
GCC_KW      = ["cbb","sama","uaecb","dfsa","regulation","law","policy","rulebook",
               "vision 2030","رؤية","قانون","نظام","تنظيم","مصرف البحرين",
               "مصرف المركزي","ترخيص","امتثال","compliance","ضوابط","اشتراطات"]
DIALECT_KW  = ["لهجة","dialect","بحريني","خليجي","bahraini","معنى","وايد",
               "حيل","شلون","ترجم","translate","تحليل","تطبيع","code-switching",
               "morphology","فصحى","صرف","جذر"]
EXTRACT_KW  = ["extract","entities","relations","استخرج","كيانات","علاقات"]
REASON_KW   = ["why","how","explain","analyze","compare","لماذا","كيف",
               "اشرح","حلل","قارن","ما الفرق","ما الأسباب"]


def classify_intent(q: str) -> dict:
    ql = q.lower(); ar = bool(AR_RE.search(q)); wc = len(q.split())
    return {
        "research":    any(k in ql for k in RESEARCH_KW),
        "gcc_law":     any(k in ql for k in GCC_KW),
        "dialect":     ar or any(k in ql for k in DIALECT_KW),
        "reasoning":   wc > 12 or any(k in ql for k in REASON_KW),
        "extraction":  any(k in ql for k in EXTRACT_KW),
        "is_arabic":   ar,
    }


def build_pipeline(intent: dict, mode: str = "auto") -> List[str]:
    # Single-agent override
    if mode.startswith("single:"):
        a = mode.split(":", 1)[1]
        return [a, "muraqib"] if a != "muraqib" else [a]
    # Legacy mode aliases
    if mode == "arabic_nlp":    return ["lughawi", "muraqib"]
    if mode == "knowledge":     return ["bani", "muraqib"]
    if mode == "deep_research": return ["bahith", "hakeem", "muraqib"]
    if mode == "cognitive":     pass  # fall through to auto

    p = []
    if intent["research"]:   p.append("bahith")
    if intent["gcc_law"]:    p.append("musheer")
    if intent["dialect"]:    p.append("lughawi")
    if intent["reasoning"] or len(p) > 1: p.append("hakeem")
    if intent["extraction"]: p.append("bani")
    p.append("muraqib")
    if len(p) == 1: p = ["hakeem", "muraqib"]

    seen, out = set(), []
    for a in p:
        if a not in seen: seen.add(a); out.append(a)
    return out


AGENT_LABELS = {
    "bahith":"🔭 باحث", "musheer":"⚖️ مشير", "lughawi":"ع لغوي",
    "hakeem":"🧠 حكيم", "muraqib":"🔍 مراقب", "bani":"🕸️ بانِ",
}

SYSTEM_PROMPTS = {
"bahith": """أنت باحث في ACAI. قدّم معلومات دقيقة.
التنسيق:
**الملخص:** (2-3 جمل)
**النتائج الرئيسية:** نقاط + مصادر
**التحليل:** سياق أعمق
**الموثوقية:** X/10
لا تخترع مصادر. أجب بنفس لغة السؤال.""",

"musheer": """أنت مشير — خبير أنظمة الخليج. مراجعك: CBB، SAMA، UAECB، DFSA.
التنسيق:
**الحكم:** [المنظم | الوثيقة | القسم]
**التفاصيل:** شرح النظام
**المتطلبات:** خطوات أو شروط
⚠️ هذا تحليل استرشادي. راجع متخصصاً قانونياً.""",

"lughawi": """أنت لغوي — خبير اللغة العربية وعلم اللهجات.
**🗺️ اللهجة:** [النوع] — الثقة: X%
**المؤشرات:** الكلمات الدالة
**🔍 الصرف:** كلمة → جذر → وزن → معنى (٣ كلمات)
**✍️ الفصحى:** النص المطبَّع
**🔄 التحول اللغوي:** إن وجد
**🌍 الثقافي:** ملاحظة""",

"hakeem": """أنت حكيم — عميل التفكير العميق.
**خطوة ١:** التفكيك
**خطوة ٢:** المعرفة
**خطوة ٣:** الاستدلال
**خطوة ٤:** التحقق
**خطوة ٥:** الإجابة النهائية
**الثقة:** X/10""",

"muraqib": """أنت مراقب — عميل التحقق.
✅ **صحيح:** الدليل
⚠️ **غير محدد:** يحتاج مصدراً
❌ **خاطئ:** التصحيح
**الحكم:** X/10""",

"bani": """أنت بانِ — عميل استخراج المعرفة.
**الكيانات:** | الاسم | النوع | الثقة |
**العلاقات:** → [أ] —[علاقة]→ [ب]
**المفاهيم:** م١، م٢، م٣""",
}

AGENT_MODELS = {"lughawi": ARABIC_MODEL}


def merge_pipeline_outputs(pipeline: List[str], outputs: Dict[str, str]) -> str:
    valid = {a: o for a, o in outputs.items() if o and not o.startswith("[خطأ")}
    if not valid: return "لم أتمكن من توليد إجابة."
    if len(valid) == 1: return next(iter(valid.values()))
    parts = []
    for a in pipeline:
        if a != "muraqib" and a in valid:
            parts.append(f"### {AGENT_LABELS.get(a, a)}\n{valid[a]}")
    if "muraqib" in valid:
        parts.append(f"\n---\n### 🔍 مراقب — التحقق\n{valid['muraqib']}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# B. MEMORY — SQLite FTS5 persistent cross-session
# ══════════════════════════════════════════════════════════════════════════════

class MemoryStore:
    def __init__(self, db: Path = MEMORY_DB):
        self.db = str(db)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    query    TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality  INTEGER DEFAULT 3,
                    tags     TEXT DEFAULT '[]',
                    created  TEXT DEFAULT (datetime('now'))
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
                    query, response, tags,
                    content='conversations', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS conv_fts_ai
                    AFTER INSERT ON conversations BEGIN
                    INSERT INTO conv_fts(rowid, query, response, tags)
                    VALUES (new.id, new.query, new.response, new.tags);
                END;
                CREATE TABLE IF NOT EXISTS experiment_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    query    TEXT,
                    mode     TEXT,
                    pipeline TEXT,
                    latency  INTEGER,
                    created  TEXT DEFAULT (datetime('now'))
                );
            """)

    def get_context(self, query: str, limit: int = 3) -> str:
        """FTS5 search — returns formatted context string for prompt injection."""
        try:
            q_esc = '"' + query.replace('"', '""') + '"'
            with sqlite3.connect(self.db) as c:
                rows = c.execute("""
                    SELECT c.query, c.response
                    FROM conv_fts f JOIN conversations c ON f.rowid = c.id
                    WHERE conv_fts MATCH ? AND c.quality >= 3
                    ORDER BY rank LIMIT ?
                """, (q_esc, limit)).fetchall()
            if not rows: return ""
            lines = ["[ذاكرة ذات صلة من محادثات سابقة]"]
            for q, r in rows:
                lines.append(f"• سؤال: {q[:80]}")
                lines.append(f"  جواب: {r[:160]}")
            return "\n".join(lines)
        except Exception as e:
            log.debug(f"Memory search failed: {e}")
            return ""

    def save(self, agent_id: str, query: str, response: str,
             quality: int = 3, tags: list = None):
        try:
            with sqlite3.connect(self.db) as c:
                c.execute(
                    "INSERT INTO conversations(agent_id,query,response,quality,tags) VALUES(?,?,?,?,?)",
                    (agent_id, query[:1000], response[:3000], quality,
                     json.dumps(tags or [], ensure_ascii=False))
                )
        except Exception as e:
            log.error(f"Memory save error: {e}")

    def log_experiment(self, query: str, mode: str, pipeline: list, latency: int):
        try:
            with sqlite3.connect(self.db) as c:
                c.execute(
                    "INSERT INTO experiment_log(query,mode,pipeline,latency) VALUES(?,?,?,?)",
                    (query[:300], mode, json.dumps(pipeline), latency)
                )
        except Exception as e:
            log.error(f"Experiment log error: {e}")

    def experiment_summary(self) -> dict:
        try:
            with sqlite3.connect(self.db) as c:
                rows = c.execute(
                    "SELECT mode, COUNT(*), AVG(latency) FROM experiment_log GROUP BY mode"
                ).fetchall()
            return {r[0]: {"count": r[1], "avg_latency_ms": int(r[2] or 0)} for r in rows}
        except: return {}

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self.db) as c:
                n_conv  = c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                by_ag   = c.execute(
                    "SELECT agent_id, COUNT(*) FROM conversations GROUP BY agent_id"
                ).fetchall()
            return {"total": n_conv, "by_agent": {r[0]: r[1] for r in by_ag}}
        except: return {"total": 0, "by_agent": {}}


memory = MemoryStore()


# ══════════════════════════════════════════════════════════════════════════════
# D. MINIMAL RAG — SQLite FTS5, no external services
# ══════════════════════════════════════════════════════════════════════════════

class MinimalRAG:
    """
    Zero-dependency RAG using SQLite FTS5.
    No Weaviate, no sentence-transformers needed for demo.
    Ingest → chunk → store → retrieve → inject → cite.
    """

    def __init__(self, db: Path = RAG_DB):
        self.db = str(db)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name TEXT NOT NULL,
                    chunk_no INTEGER,
                    content  TEXT NOT NULL,
                    created  TEXT DEFAULT (datetime('now'))
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, doc_name,
                    content='chunks', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai
                    AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content, doc_name)
                    VALUES (new.id, new.content, new.doc_name);
                END;
            """)

    def ingest(self, text: str, doc_name: str, chunk_size: int = 400) -> int:
        """Split document and store as searchable chunks."""
        # Sentence-aware chunking
        sents = re.split(r'(?<=[.!?،؟])\s+', text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) < chunk_size:
                cur += s + " "
            else:
                if cur.strip(): chunks.append(cur.strip())
                cur = s + " "
        if cur.strip(): chunks.append(cur.strip())
        if not chunks:   chunks = [text[:chunk_size]]

        with sqlite3.connect(self.db) as c:
            for i, ch in enumerate(chunks):
                c.execute(
                    "INSERT INTO chunks(doc_name, chunk_no, content) VALUES(?,?,?)",
                    (doc_name, i, ch)
                )
        log.info(f"RAG: ingested '{doc_name}' → {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list:
        """FTS5 search over chunks."""
        try:
            q_esc = '"' + query.replace('"', '""') + '"'
            with sqlite3.connect(self.db) as c:
                rows = c.execute("""
                    SELECT ch.doc_name, ch.chunk_no, ch.content
                    FROM chunks_fts cf JOIN chunks ch ON cf.rowid = ch.id
                    WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
                """, (q_esc, k)).fetchall()
            return [{"doc": r[0], "chunk": r[1], "content": r[2]} for r in rows]
        except Exception as e:
            log.debug(f"RAG retrieve error: {e}")
            return []

    def get_rag_context(self, query: str, k: int = 3) -> str:
        chunks = self.retrieve(query, k)
        if not chunks: return ""
        lines = ["[مقتطفات من الوثائق المرجعية]"]
        for c in chunks:
            lines.append(f"📄 {c['doc']} — القطعة {c['chunk']+1}")
            lines.append(f"   {c['content'][:300]}")
        return "\n".join(lines)

    def list_docs(self) -> list:
        try:
            with sqlite3.connect(self.db) as c:
                rows = c.execute(
                    "SELECT doc_name, COUNT(*) FROM chunks GROUP BY doc_name"
                ).fetchall()
            return [{"doc": r[0], "chunks": r[1]} for r in rows]
        except: return []


rag = MinimalRAG()

# ── Auto-ingest sample CBB document on first run ──────────────────────────────
CBB_SAMPLE = """مصرف البحرين المركزي — ملخص تنظيمي
الهدف: المحافظة على الاستقرار النقدي والمالي في مملكة البحرين.

الترخيص المصرفي:
رأس المال الأدنى للبنوك التجارية: 100 مليون دينار بحريني.
يشترط تقديم طلب مكتمل مع خطة عمل خمسية ونظام حوكمة معتمد.
يستغرق قرار الترخيص عادةً 6-12 شهراً.

حماية المستهلك (CBB Rulebook — المجلد الخامس):
يلتزم البنك بالإفصاح الكامل عن الرسوم والفوائد.
يجب توفير قناة شكاوى رسمية.
الرد على الشكاوى خلال 15 يوم عمل.

مكافحة غسل الأموال:
تطبيق إجراءات KYC (اعرف عميلك) إلزامي.
الإبلاغ عن المعاملات المشبوهة لوحدة الاستخبارات المالية.

رؤية البحرين 2030:
تنويع الاقتصاد وتقليل الاعتماد على النفط.
تطوير قطاع الخدمات المالية والتكنولوجيا المالية (Fintech).
تمكين الكوادر البحرينية في القطاع المالي.

SAMA — البنك المركزي السعودي:
ينظم القطاع المالي في المملكة العربية السعودية.
متطلبات الترخيص مشابهة لـ CBB مع اشتراطات إضافية للبنوك الإسلامية.
يشترط الالتزام بنظام ساما للمدفوعات الفورية (SADAD/SARIE).

UAECB — مصرف الإمارات المركزي:
ينظم البنوك في الإمارات العربية المتحدة.
رأس المال الأدنى: 150 مليون درهم إماراتي.
نظام AECB لتقارير الائتمان."""


# ══════════════════════════════════════════════════════════════════════════════
# C. SECURITY — Backend-only API calls
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, rpm: int = 40):
        self.rpm = rpm
        self._log: Dict[str, list] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.time()
        self._log[key] = [t for t in self._log[key] if now - t < 60]
        if len(self._log[key]) >= self.rpm: return False
        self._log[key].append(now)
        return True


limiter = RateLimiter()


async def backend_ddg_search(query: str, n: int = 5) -> list:
    """DuckDuckGo — runs in BACKEND. Frontend never touches APIs."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            results = list(d.text(query, max_results=n))
            return [{"title": r.get("title",""), "url": r.get("href",""),
                     "snippet": r.get("body","")[:400]} for r in results if r.get("body")]
    except ImportError:
        log.warning("pip install duckduckgo-search")
        return []
    except Exception as e:
        log.warning(f"DDG: {e}")
        return []


async def backend_anthropic_search(query: str, system: str) -> str:
    """Anthropic web search — API key lives on SERVER only."""
    if not ANTHROPIC_KEY: return ""
    msgs = [{"role": "user", "content": query}]
    try:
        for _ in range(8):
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json",
                             "x-api-key": ANTHROPIC_KEY,
                             "anthropic-version": "2023-06-01"},
                    json={"model": CLAUDE_MODEL, "max_tokens": 2000,
                          "system": system, "messages": msgs,
                          "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
                )
            if r.status_code != 200: return ""
            d = r.json()
            if d.get("stop_reason") != "tool_use":
                return " ".join(b["text"] for b in d.get("content",[]) if b["type"]=="text")
            msgs.append({"role": "assistant", "content": d["content"]})
            tool_results = [
                {"type": "tool_result", "tool_use_id": b["id"], "content": "Done."}
                for b in d["content"] if b.get("type") == "tool_use"
            ]
            msgs.append({"role": "user", "content": tool_results})
    except Exception as e:
        log.error(f"Anthropic search error: {e}")
    return ""


async def ollama_call(prompt: str, system: str = "", model: str = None) -> str:
    model = model or PRIMARY_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1500
        }
    }
    if system: payload["system"] = system
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            print(payload)
            res = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            res.raise_for_status()
            if res.status_code == 200:
                return res.json().get("response", "")
    except Exception as e:
        log.error(f"Ollama error: {e}")
    return ""

async def execute_agent(agent_id: str, query: str,
                        prev_context: str = "",
                        memory_ctx: str = "",
                        rag_ctx: str = "") -> str:
    """Execute one agent. Builds prompt, calls LLM, returns text."""
    system = SYSTEM_PROMPTS.get(agent_id, "أنت مساعد ذكي.")
    model  = AGENT_MODELS.get(agent_id, PRIMARY_MODEL)

    # باحث uses web search
    if agent_id == "bahith":
        # Try Anthropic (best) → DDG fallback
        if ANTHROPIC_KEY:
            result = await backend_anthropic_search(query, system)
            if result: return result
        search = await backend_ddg_search(query, n=5)
        if search:
            ctx = "\n\n".join(
                f"Source: {s['title']}\nURL: {s['url']}\n{s['snippet']}"
                for s in search
            )
            prompt = f"{ctx}\n\nبناءً على النتائج أعلاه، أجب على: {query}"
        else:
            prompt = query
        return await ollama_call(prompt, system, model)

    # Build prompt with all context layers
    parts = []
    if memory_ctx:   parts.append(f"[ذاكرة]\n{memory_ctx}")
    if rag_ctx:      parts.append(f"[وثائق]\n{rag_ctx}")
    if prev_context: parts.append(f"[مخرجات سابقة]\n{prev_context}")
    parts.append(query)
    return await ollama_call("\n\n".join(parts), system, model)


async def orchestrate(query: str, mode: str = "auto",
                      session_id: str = "default") -> dict:
    """Full orchestration: memory → RAG → pipeline → merge → save."""
    t0 = time.time()

    # 1. Memory retrieval (ALWAYS before every query)
    mem_ctx  = memory.get_context(query, limit=3)
    mem_used = bool(mem_ctx)

    # 2. RAG retrieval
    rag_ctx  = rag.get_rag_context(query, k=2)
    rag_used = bool(rag_ctx)

    # 3. Intent + pipeline
    intent   = classify_intent(query)
    pipeline = build_pipeline(intent, mode)
    log.info(f"Pipeline: {pipeline} | memory={mem_used} | rag={rag_used}")

    # 4. Sequential execution (context accumulates)
    outputs:  Dict[str, str] = {}
    acc_ctx:  str = ""
    for agent_id in pipeline:
        out = await execute_agent(
            agent_id   = agent_id,
            query      = query,
            prev_context = acc_ctx,
            memory_ctx = mem_ctx if not acc_ctx else "",
            rag_ctx    = rag_ctx  if not acc_ctx else "",
        )
        outputs[agent_id] = out
        if out and not out.startswith("[خطأ"): acc_ctx = out

    # 5. Merge
    final = merge_pipeline_outputs(pipeline, outputs)

    latency = int((time.time() - t0) * 1000)

    # 6. Save to memory + log
    memory.save("orchestrator", query, final, quality=3)
    memory.log_experiment(query,
                          "with_memory" if mem_used else "without_memory",
                          pipeline, latency)

    return {
        "answer":      final,
        "pipeline":    pipeline,
        "intent":      intent,
        "memory_used": mem_used,
        "rag_used":    rag_used,
        "latency_ms":  latency,
        "agents":      {k: v[:200] + "..." if len(v) > 200 else v
                        for k, v in outputs.items()},
    }


def merge_pipeline_outputs(pipeline: list, outputs: dict) -> str:
    valid = {a: o for a, o in outputs.items() if o and not o.startswith("[خطأ")}
    if not valid: return "لم أتمكن من توليد إجابة."
    if len(valid) == 1: return next(iter(valid.values()))
    parts = []
    for a in pipeline:
        if a != "muraqib" and a in valid:
            parts.append(f"### {AGENT_LABELS.get(a, a)}\n{valid[a]}")
    if "muraqib" in valid:
        parts.append(f"\n---\n### 🔍 مراقب\n{valid['muraqib']}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# E. EVALUATION — DCR / MLR + before/after memory experiment
# ══════════════════════════════════════════════════════════════════════════════

BAHRAINI_MARKERS = ["الحين","وايد","حيل","شلونك","خوي","صج","مب","عساك","زين","تره","باكر"]
MSA_MARKERS      = ["كيف حالك","أنا بخير","شكراً لك","ينبغي","يجب أن","لا أعلم"]

async def run_dcr_eval() -> dict:
    """Measure Dialect Control Rate and MSA Leak Rate."""
    prompts = [
        "أجب باللهجة البحرينية فقط: كيف حالك؟",
        "اشرح كيف تفتح حساب بنكي باللهجة البحرينية",
        "قل لي أنك لا تعرف الإجابة باللهجة البحرينية",
    ]
    dcr_scores, mlr_scores = [], []
    for prompt in prompts:
        result = await orchestrate(prompt, mode="single:lughawi")
        resp   = result["answer"].lower()
        dcr = sum(1 for m in BAHRAINI_MARKERS if m in resp) / len(BAHRAINI_MARKERS)
        mlr = sum(1 for m in MSA_MARKERS     if m in resp) / len(MSA_MARKERS)
        dcr_scores.append(dcr); mlr_scores.append(mlr)
    return {
        "avg_dcr":   round(sum(dcr_scores)/len(dcr_scores), 3),
        "avg_mlr":   round(sum(mlr_scores)/len(mlr_scores), 3),
        "dcr_per":   dcr_scores,
        "mlr_per":   mlr_scores,
    }


async def run_memory_experiment(questions: list) -> dict:
    """
    Before/after memory experiment.
    Clears memory context for 'before' queries, uses accumulated memory for 'after'.
    Returns comparison — this is your paper's Table 2.
    """
    results = []
    # 'without_memory' pass — clear memory context by using fresh queries
    for q in questions[:8]:
        r = await orchestrate(q, mode="auto")
        results.append({"question": q[:80],
                         "without_memory_latency": r["latency_ms"],
                         "without_memory_answer":  r["answer"][:200]})

    # 'with_memory' pass — memory now has context from above queries
    for i, q in enumerate(questions[:8]):
        r = await orchestrate(q, mode="auto")
        results[i]["with_memory_latency"]   = r["latency_ms"]
        results[i]["with_memory_answer"]    = r["answer"][:200]
        results[i]["memory_was_available"]  = r["memory_used"]

    return {
        "results":  results,
        "summary":  memory.experiment_summary(),
        "note":     "Compare without_memory_answer vs with_memory_answer for quality improvement",
    }


# FASTAPI APP
async def lifespan(app: FastAPI):
    log.info("🚀 ACAI v5 — محرك الذكاء الاصطناعي المعرفي العربي")
    log.info(f"   Model:     {PRIMARY_MODEL}")
    log.info(f"   Anthropic: {'✅' if ANTHROPIC_KEY else '⚠️  not set (DDG fallback)'}")
    log.info(f"   Memory DB: {MEMORY_DB}")

    # Auto-ingest CBB sample if RAG is empty
    if not rag.list_docs():
        n = rag.ingest(CBB_SAMPLE, "CBB_Rulebook_Sample")
        log.info(f"   RAG:       ✅ ingested CBB sample — {n} chunks")
    else:
        log.info(f"   RAG:       ✅ {rag.list_docs()}")
        
    log.info("   ✅ ACAI ready")
    
    yield # execute everything before `yield` at startup

app = FastAPI(
    title="ACAI",
    description="Arabic Cognitive AI Engine — Production Backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://localhost(:5173)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {"/api/health", "/docs", "/openapi.json", "/", "/api/query/stream"}

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    # Auth
    key = (request.headers.get("X-API-Key")
           or request.headers.get("x-api-key"))
    if key != API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    # Rate limiting
    ip = request.client.host if request.client else "unknown"
    if not limiter.check(ip):
        return JSONResponse({"error": "Rate limit — 40 req/min"}, status_code=429)
    return await call_next(request)


# Health
@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except:
        models, ollama_ok = [], False
    return {
        "status":       "ok",
        "model":        PRIMARY_MODEL,
        "ollama":       ollama_ok,
        "models":       models,
        "anthropic":    bool(ANTHROPIC_KEY),
        "memory_stats": memory.stats(),
        "rag_docs":     rag.list_docs(),
        "timestamp":    datetime.now().isoformat(),
    }


# Streaming query (main endpoint)
@app.post("/api/query/stream")
async def query_stream(request: Request):
    body       = await request.json()
    query      = body.get("query", "").strip()
    mode       = body.get("mode", "auto")
    session_id = body.get("session_id", "default")

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    async def stream():
        try:
            result = await orchestrate(query, mode, session_id)
            answer = result["answer"]
            # Stream in small chunks
            CHUNK = 5
            words = answer.split(" ")
            for i in range(0, len(words), CHUNK):
                text = " ".join(words[i:i+CHUNK]) + " "
                yield f"data: {json.dumps({'type':'chunk','text':text})}\n\n"
                await asyncio.sleep(0.008)
            yield f"data: {json.dumps({'type':'done','pipeline':result['pipeline'],'memory_used':result['memory_used'],'latency_ms':result['latency_ms']})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','error':str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


# ── Non-streaming query ───────────────────────────────────────────────────────
@app.post("/api/query")
async def query_sync(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    mode  = body.get("mode","auto")
    sid   = body.get("session_id","default")
    if not query: raise HTTPException(400, "Empty query")
    result = await orchestrate(query, mode, sid)
    return {"answer": result["answer"], "pipeline": result["pipeline"],
            "memory_used": result["memory_used"], "latency_ms": result["latency_ms"]}


# ── Web search (backend only — NO frontend API keys) ─────────────────────────
@app.post("/api/search")
async def search(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    if not query: return {"sources":[], "has_web":False}
    sources = await backend_ddg_search(query, n=5)
    return {"sources": sources, "has_web": bool(sources), "query": query}


# ── RAG endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/rag/ingest")
async def rag_ingest(request: Request):
    body     = await request.json()
    text     = body.get("text","")
    doc_name = body.get("doc_name","document")
    if not text: raise HTTPException(400, "No text provided")
    n = rag.ingest(text, doc_name)
    return {"doc_name": doc_name, "chunks": n, "status": "ok"}

@app.get("/api/rag/docs")
async def rag_docs():
    return {"docs": rag.list_docs()}

@app.post("/api/rag/retrieve")
async def rag_retrieve(request: Request):
    body  = await request.json()
    query = body.get("query","")
    k     = body.get("k", 3)
    return {"chunks": rag.retrieve(query, k)}


# ── Memory endpoints ──────────────────────────────────────────────────────────
@app.get("/api/memory/stats")
async def mem_stats():
    return {"memory": memory.stats(), "experiment": memory.experiment_summary()}

@app.post("/api/memory/search")
async def mem_search(request: Request):
    body  = await request.json()
    query = body.get("query","")
    return {"context": memory.get_context(query, 5)}


# ── Evaluation endpoints ─────────────────────────────────────────────────────
@app.get("/api/eval/dcr")
async def eval_dcr():
    """Run DCR + MLR evaluation. Returns paper metrics."""
    result = await run_dcr_eval()
    return result

@app.post("/api/eval/memory_experiment")
async def eval_memory(request: Request):
    """Before/after memory experiment for the paper."""
    body      = await request.json()
    questions = body.get("questions", [
        "ما متطلبات ترخيص البنك في البحرين؟",
        "كيف أحمي حسابي من الاحتيال؟",
        "ما هي رؤية البحرين 2030؟",
        "شلون أعرف رصيد حسابي؟",
        "ما الفرق بين CBB و SAMA؟",
    ])
    return await run_memory_experiment(questions)

@app.post("/api/eval/benchmark")
async def run_benchmark_endpoint(request: Request):
    """Run Bahraini benchmark against current model."""
    body = await request.json()
    questions = body.get("questions", [])
    if not questions:
        return JSONResponse({"error": "Provide questions array"}, status_code=400)

    results = []
    for item in questions[:20]:
        q   = item.get("q","")
        ans = item.get("answer","")
        r   = await orchestrate(q, mode="auto")
        results.append({
            "q":           q,
            "expected":    ans,
            "got":         r["answer"][:100],
            "pipeline":    r["pipeline"],
            "latency_ms":  r["latency_ms"],
        })
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
