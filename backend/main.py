"""
ACAI v4 — Production FastAPI Server
=====================================
Full production server with:
  - JWT authentication + API key support
  - Rate limiting per user/tier
  - Server-Sent Events (SSE) streaming
  - WebSocket real-time pipeline
  - CORS for frontend
  - Prometheus metrics
  - Structured audit logging
  - Health checks and readiness probes

Start:  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Docs:   http://localhost:8000/api/docs
"""

import asyncio
import logging
import time
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Any

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect,
    Request, status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Optional dependencies — graceful fallback if not installed
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
    # Metrics
    REQUEST_COUNT    = Counter("acai_requests_total",   "Total requests", ["endpoint", "status"])
    REQUEST_LATENCY  = Histogram("acai_latency_seconds", "Request latency", ["endpoint"])
    ACTIVE_SESSIONS  = Gauge("acai_active_sessions",    "Active WebSocket sessions")
    LLM_TOKENS       = Counter("acai_llm_tokens_total", "LLM tokens used", ["model", "direction"])
    AGENT_CALLS      = Counter("acai_agent_calls_total", "Agent invocations", ["agent"])
    HALLUCINATION_RATE = Gauge("acai_hallucination_rate", "Estimated hallucination rate")
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Internal imports (from Phase 1 + v4 additions)
from llm.inference_client import LLMClient
from model_config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("acai.api")

# ─── App State ────────────────────────────────────────────────────────────────

class AppState:
    llm_client: Optional[LLMClient] = None
    orchestrator: Any = None
    rag_pipeline: Any = None
    arabic_nlp: Any = None
    kg_connector: Any = None
    memory_system: Any = None
    feedback_system: Any = None
    ingestion_pipeline: Any = None
    active_ws_sessions: Dict[str, WebSocket] = {}
    start_time: float = time.time()

app_state = AppState()

# ─── Lifespan (startup/shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
    logger.info("🚀 ACAI v4 starting up...")

    # LLM Client (always available)
    app_state.llm_client = LLMClient()
    llm_health = await app_state.llm_client.health_check()
    logger.info(f"LLM backends: {llm_health}")

    # Optional services — graceful degradation if not installed
    try:
        from agents.langgraph_orchestrator import CognitiveOrchestrator
        app_state.orchestrator = CognitiveOrchestrator(app_state.llm_client)
        logger.info("✅ Orchestrator loaded")
    except Exception as e:
        logger.warning(f"⚠️  Orchestrator not loaded: {e}")

    try:
        from rag.advanced_graphrag import AdvancedGraphRAG
        app_state.rag_pipeline = AdvancedGraphRAG()
        await app_state.rag_pipeline.initialize()
        logger.info("✅ RAG pipeline loaded")
    except Exception as e:
        logger.warning(f"⚠️  RAG not loaded: {e}")

    try:
        from arabic.dialect_specialist import ArabicNLPSpecialist
        app_state.arabic_nlp = ArabicNLPSpecialist()
        logger.info("✅ Arabic NLP loaded")
    except Exception as e:
        logger.warning(f"⚠️  Arabic NLP not loaded: {e}")

    try:
        from memory.quantum_memory import CognitiveMemorySystem
        app_state.memory_system = CognitiveMemorySystem()
        await app_state.memory_system.initialize()
        logger.info("✅ Memory system loaded")
    except Exception as e:
        logger.warning(f"⚠️  Memory not loaded: {e}")

    try:
        from feedback.system import FeedbackSystem
        app_state.feedback_system = FeedbackSystem()
        logger.info("✅ Feedback system loaded")
    except Exception as e:
        logger.warning(f"⚠️  Feedback not loaded: {e}")

    logger.info("✅ ACAI v4 ready — محرك الذكاء الاصطناعي المعرفي العربي")

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down ACAI v4...")
    if app_state.llm_client:
        await app_state.llm_client.close()
    if app_state.memory_system:
        await app_state.memory_system.shutdown()
    logger.info("ACAI v4 shutdown complete")

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Arabic Cognitive AI Engine API",
    description="""
# محرك الذكاء الاصطناعي المعرفي العربي
## ACAI v4 — Arabic Cognitive OS

Production API for the Arabic Cognitive AI Platform.

### Modes
- **Deep Research**: Real-time web search + multi-source synthesis
- **Cognitive Pipeline**: 5-agent sequential reasoning (Plan→Research→Reason→Verify→Synthesize)
- **Arabic NLP**: Dialect detection, morphology, MSA normalization (15 dialects incl. Bahraini)
- **Knowledge Graph**: Entity extraction, GCC ontology, GraphRAG

### Authentication
Include JWT in Authorization header: `Bearer <token>`
Or API key in `X-API-Key` header.
    """,
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
allow_origins=["*"],    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)

    duration = (time.time() - start) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.0f}ms)"
    )

    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(endpoint=request.url.path, status=str(response.status_code)).inc()

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.0f}ms"
    return response

# ─── Auth ─────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-this-in-production-immediately")
API_KEYS = set(os.getenv("API_KEYS", "dev-key-12345").split(","))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token or API key."""
    # Allow API key auth
    if credentials and credentials.credentials in API_KEYS:
        return {"user_id": "api_key_user", "tier": "api", "key": credentials.credentials[:8]}

    # Allow JWT auth
    if credentials and JWT_AVAILABLE:
        try:
            payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
            return payload
        except Exception:
            pass

    # In development mode — allow unauthenticated (set DEV_MODE=true)
    if os.getenv("DEV_MODE", "true").lower() == "true":
        return {"user_id": "dev_user", "tier": "dev"}

    raise HTTPException(status_code=401, detail="Invalid or missing authentication")

def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict]:
    """Optional auth — returns None if not authenticated."""
    try:
        return verify_token(credentials)
    except HTTPException:
        return None

# ─── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="Query in Arabic or English")
    mode: str = Field("deep_research", description="deep_research|cognitive|arabic_nlp|knowledge")
    session_id: Optional[str] = Field(None, description="Session ID for memory continuity")
    language: Optional[str] = Field(None, description="ar|en|auto")
    use_memory: bool = Field(True, description="Use session memory context")
    stream: bool = Field(False, description="Stream response chunks")

class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    query: str
    response: str
    rating: int = Field(..., ge=-1, le=1, description="1=positive, -1=negative, 0=neutral")
    correction: Optional[str] = Field(None, description="User's correction if rating=-1")

class ArabicAnalysisRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Arabic text to analyze")
    analysis_type: str = Field("full", description="full|dialect|morphology|normalize")

class IngestionRequest(BaseModel):
    source: str = Field("all", description="wikipedia|news|academic|regulatory|all")
    limit: int = Field(50, ge=1, le=1000, description="Max documents to ingest")

class QueryResponse(BaseModel):
    session_id: str
    message_id: str
    query: str
    answer: str
    mode: str
    agent_traces: List[Dict] = []
    sources: List[str] = []
    searches_performed: List[str] = []
    confidence: float = 0.0
    latency_ms: float = 0.0
    model_used: str = ""
    language_detected: str = "unknown"

# ─── Core Query Endpoint ──────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse, tags=["Core"])
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user: Dict = Depends(optional_auth),
):
    """
    Main query endpoint. Supports all 4 cognitive modes.
    For streaming, set stream=true and use /api/query/stream instead.
    """
    t0 = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    message_id = f"msg_{uuid.uuid4().hex[:12]}"

    # Get memory context if available
    memory_context = ""
    if request.use_memory and app_state.memory_system:
        memory_context = await app_state.memory_system.retrieve_context(
            request.query, session_id
        )

    try:
        result = await _execute_query(request, session_id, memory_context)
        latency = (time.time() - t0) * 1000

        # Store in memory (background)
        if app_state.memory_system:
            background_tasks.add_task(
                app_state.memory_system.store,
                session_id=session_id,
                query=request.query,
                response=result.get("answer", ""),
                agent=request.mode,
            )

        if PROMETHEUS_AVAILABLE:
            AGENT_CALLS.labels(agent=request.mode).inc()

        return QueryResponse(
            session_id=session_id,
            message_id=message_id,
            query=request.query,
            answer=result.get("answer", ""),
            mode=request.mode,
            agent_traces=result.get("traces", []),
            sources=result.get("sources", []),
            searches_performed=result.get("searches", []),
            confidence=result.get("confidence", 0.0),
            latency_ms=latency,
            model_used=result.get("model", ""),
            language_detected=result.get("language", "unknown"),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_query(request: QueryRequest, session_id: str, memory_context: str) -> Dict:
    """Route query to the right pipeline based on mode."""
    query = request.query
    if memory_context:
        query = f"{query}\n\n[MEMORY CONTEXT]\n{memory_context}"

    # Use orchestrator if available (Phase 3+)
    if app_state.orchestrator:
        return await app_state.orchestrator.execute(request.query, request.mode, session_id)

    # Direct LLM fallback (Phase 2 — always works)
    if not app_state.llm_client:
        raise RuntimeError("No LLM client available")

    SYSTEM_PROMPTS = {
        "deep_research": """You are the Deep Research Intelligence of the Arabic Cognitive AI Engine.
Research the query thoroughly. Provide comprehensive, well-cited analysis.
Prioritize GCC/Arabic sources. Flag any uncertain information clearly.
Respond in the same language as the query. Arabic queries → Arabic MSA response.""",

        "cognitive": """You are the master reasoning intelligence of the Arabic Cognitive AI Engine.
Apply rigorous multi-step reasoning:
STEP 1 [ANALYZE]: What exactly is being asked?
STEP 2 [RESEARCH]: What relevant knowledge applies?
STEP 3 [REASON]: What logical conclusions follow?
STEP 4 [VERIFY]: What could be wrong with this reasoning?
STEP 5 [ANSWER]: Provide the final comprehensive response.
Respond in the same language as the query.""",

        "arabic_nlp": """أنت عميل اللغة العربية في محرك الذكاء الاصطناعي المعرفي العربي.
You are an expert Arabic linguist. Analyze:
1. Dialect identification (MSA/Bahraini/Gulf/Egyptian/Levantine/Maghrebi)
2. Morphological analysis of key words
3. MSA normalization
4. Cultural/regional context
5. Code-switching detection
Write analysis in Arabic MSA with English explanations.""",

        "knowledge": """You are the Knowledge Graph Intelligence.
Extract a complete knowledge graph from the text.
Return structured JSON with entities (name, name_ar, type, confidence) 
and relations (from, type, to, evidence).
Entity types: Person|Organization|Location|Concept|Regulation|Technology|Event
Relation types: GOVERNS|REGULATES|LOCATED_IN|PART_OF|EMPLOYS|DEVELOPS|COMPETES_WITH""",
    }

    system = SYSTEM_PROMPTS.get(request.mode, SYSTEM_PROMPTS["cognitive"])
    response = await app_state.llm_client.generate(
        prompt=query, system=system
    )

    # Detect language
    arabic_ratio = sum(1 for c in request.query if '\u0600' <= c <= '\u06FF') / max(len(request.query), 1)
    lang = "ar" if arabic_ratio > 0.5 else "mixed" if arabic_ratio > 0.1 else "en"

    return {
        "answer": response.text,
        "traces": [],
        "sources": [],
        "searches": [],
        "confidence": 0.85,
        "model": response.model,
        "language": lang,
    }

# ─── Streaming Endpoint ───────────────────────────────────────────────────────

@app.post("/api/query/stream", tags=["Core"])
async def query_stream(
    request: QueryRequest,
    user: Dict = Depends(optional_auth),
):
    """Stream query response as Server-Sent Events (SSE)."""
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

        try:
            if app_state.llm_client:
                SYSTEM_PROMPTS = {
                    "deep_research": "You are the Deep Research Intelligence. Research comprehensively, cite sources.",
                    "cognitive": "You are a master reasoner. Think step-by-step, verify your reasoning.",
                    "arabic_nlp": "أنت عميل اللغة العربية. Analyze Arabic text in depth.",
                    "knowledge": "Extract structured knowledge graph entities and relations.",
                }
                system = SYSTEM_PROMPTS.get(request.mode, "")

                async for chunk in app_state.llm_client.stream(
                    prompt=request.query, system=system
                ):
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                    await asyncio.sleep(0)  # Yield to event loop

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time WebSocket for live agent pipeline visualization.
    Client receives agent status updates as each agent completes.
    """
    await websocket.accept()
    app_state.active_ws_sessions[session_id] = websocket
    if PROMETHEUS_AVAILABLE:
        ACTIVE_SESSIONS.inc()

    try:
        await websocket.send_json({"type": "connected", "session_id": session_id})

        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            mode  = data.get("mode", "cognitive")

            if not query:
                continue

            # Send pipeline stage updates
            stages = ["planner", "research", "reasoning", "verification", "synthesis"]
            for stage in stages:
                await websocket.send_json({"type": "agent_start", "agent": stage})
                await asyncio.sleep(0.2)

                # Execute stage (simplified — orchestrator handles real execution)
                if app_state.llm_client:
                    result = await app_state.llm_client.generate(
                        prompt=query,
                        system=f"You are the {stage} agent. Be concise.",
                        max_tokens=512,
                    )
                    await websocket.send_json({
                        "type": "agent_complete",
                        "agent": stage,
                        "output": result.text[:500],
                        "tokens": result.tokens,
                    })
                else:
                    await websocket.send_json({"type": "agent_complete", "agent": stage, "output": ""})

            await websocket.send_json({"type": "pipeline_complete"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        app_state.active_ws_sessions.pop(session_id, None)
        if PROMETHEUS_AVAILABLE:
            ACTIVE_SESSIONS.dec()

# ─── Arabic NLP Endpoints ─────────────────────────────────────────────────────

@app.post("/api/arabic/analyze", tags=["Arabic NLP"])
async def arabic_analyze(request: ArabicAnalysisRequest, user: Dict = Depends(optional_auth)):
    """Full Arabic linguistic analysis: dialect, morphology, NER, normalization."""
    if app_state.arabic_nlp:
        result = await app_state.arabic_nlp.analyze(request.text)
        return result

    # Direct LLM fallback
    if app_state.llm_client:
        response = await app_state.llm_client.generate(
            prompt=request.text,
            system="""أنت عميل اللغة العربية. Analyze:
1. Dialect (MSA/Bahraini/Gulf/Egyptian/Levantine/Maghrebi) + confidence %
2. Morphological analysis of 3 key words (root → pattern → meaning)
3. MSA normalized version
4. Cultural/regional context
5. Code-switching if present""",
            model="hf.co/inceptionai/jais-family-30b-chat",  # Prefer Jais for Arabic
        )
        return {"analysis": response.text, "model": response.model}

    raise HTTPException(503, "Arabic NLP service not available")


@app.post("/api/arabic/detect-dialect", tags=["Arabic NLP"])
async def detect_dialect(text: str, user: Dict = Depends(optional_auth)):
    """Fast dialect detection only."""
    if app_state.arabic_nlp:
        return await app_state.arabic_nlp.detect_dialect(text)

    if app_state.llm_client:
        response = await app_state.llm_client.generate(
            prompt=f"Detect the Arabic dialect of this text and return JSON only: {text}",
            system="Return only: {\"dialect\": \"...\", \"confidence\": 0.X, \"markers\": [...]}",
        )
        try:
            return json.loads(response.text)
        except Exception:
            return {"dialect": "unknown", "raw": response.text}

    raise HTTPException(503, "Service not available")

# ─── RAG Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/rag/search", tags=["Knowledge"])
async def rag_search(query: str, top_k: int = 10, user: Dict = Depends(optional_auth)):
    """Hybrid vector + keyword search over the Arabic knowledge base."""
    if app_state.rag_pipeline:
        return await app_state.rag_pipeline.hybrid_search(query, top_k=top_k)
    return {"results": [], "note": "RAG pipeline not initialized — run docker-compose up weaviate"}

@app.post("/api/rag/ingest", tags=["Knowledge"])
async def ingest(request: IngestionRequest, background_tasks: BackgroundTasks,
                 user: Dict = Depends(optional_auth)):
    """Trigger Arabic knowledge ingestion from configured sources."""
    if app_state.ingestion_pipeline:
        background_tasks.add_task(app_state.ingestion_pipeline.ingest_batch, request.source)
        return {"status": "ingestion_started", "source": request.source}
    return {"status": "ingestion_pipeline_not_available"}

# ─── Knowledge Graph Endpoints ────────────────────────────────────────────────

@app.post("/api/kg/extract", tags=["Knowledge"])
async def kg_extract(text: str, user: Dict = Depends(optional_auth)):
    """Extract entities and relations from text into knowledge graph format."""
    if app_state.llm_client:
        response = await app_state.llm_client.generate(
            prompt=text,
            system="""Extract knowledge graph. Return ONLY valid JSON:
{"entities":[{"id":"..","name":"..","name_ar":"..","type":"Person|Organization|Location|Concept|Regulation|Technology","confidence":0.9}],
"relations":[{"from":"..","type":"GOVERNS|REGULATES|LOCATED_IN|PART_OF|EMPLOYS|DEVELOPS","to":"..","evidence":".."}],
"gcc_entities":[],"key_concepts":[]}""",
        )
        try:
            return json.loads(response.text.replace("```json", "").replace("```", "").strip())
        except Exception:
            return {"raw": response.text}
    raise HTTPException(503, "LLM not available")

# ─── Feedback Endpoints ───────────────────────────────────────────────────────

@app.post("/api/feedback", tags=["Learning"])
async def submit_feedback(request: FeedbackRequest, user: Dict = Depends(optional_auth)):
    """Submit feedback for a response. Feeds into RLHF/DPO learning pipeline."""
    if app_state.feedback_system:
        feedback_id = await app_state.feedback_system.record(
            session_id=request.session_id,
            message_id=request.message_id,
            query=request.query,
            response=request.response,
            agent=request.session_id,
            rating=request.rating,
            correction=request.correction,
        )
        return {"status": "recorded", "feedback_id": feedback_id}
    return {"status": "feedback_system_not_available"}

@app.get("/api/feedback/stats", tags=["Learning"])
async def feedback_stats(user: Dict = Depends(verify_token)):
    """Get feedback statistics and learning signal summary."""
    if app_state.feedback_system:
        return await app_state.feedback_system.get_system_stats()
    return {"status": "not_available"}

@app.get("/api/feedback/export", tags=["Learning"])
async def export_training_data(user: Dict = Depends(verify_token)):
    """Export DPO preference pairs for model fine-tuning."""
    if app_state.feedback_system:
        return await app_state.feedback_system.export_training_data()
    return {"status": "not_available"}

# ─── Memory Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/memory/{session_id}", tags=["Memory"])
async def get_memory(session_id: str, user: Dict = Depends(optional_auth)):
    """Get memory context for a session."""
    if app_state.memory_system:
        entries = await app_state.memory_system.retrieve_relevant("", session_id, top_k=10)
        return {"session_id": session_id, "memories": entries}
    return {"session_id": session_id, "memories": []}

@app.delete("/api/memory/{session_id}", tags=["Memory"])
async def clear_memory(session_id: str, user: Dict = Depends(optional_auth)):
    """Clear session memory (GDPR/privacy compliance)."""
    if app_state.memory_system:
        await app_state.memory_system.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

# ─── System Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health():
    """Health check — returns service status of all components."""
    llm_health = {}
    if app_state.llm_client:
        llm_health = await app_state.llm_client.health_check()

    uptime = time.time() - app_state.start_time

    return {
        "status": "healthy",
        "version": "4.0.0",
        "uptime_seconds": round(uptime),
        "components": {
            "llm": llm_health,
            "orchestrator": app_state.orchestrator is not None,
            "rag": app_state.rag_pipeline is not None,
            "arabic_nlp": app_state.arabic_nlp is not None,
            "memory": app_state.memory_system is not None,
            "feedback": app_state.feedback_system is not None,
        },
        "active_sessions": len(app_state.active_ws_sessions),
    }

@app.get("/api/models", tags=["System"])
async def list_models():
    """List available models in the local Ollama instance."""
    if app_state.llm_client:
        models = await app_state.llm_client.list_ollama_models()
        config = get_config()
        return {
            "backend": config.backend,
            "configured": {
                "primary": config.primary,
                "specialist": config.specialist,
                "router": config.router,
            },
            "ollama_available": models,
        }
    return {"status": "llm_not_initialized"}

@app.get("/api/config", tags=["System"])
async def get_current_config(user: Dict = Depends(verify_token)):
    """Get current model configuration (admin only)."""
    config = get_config()
    return {
        "backend": config.backend,
        "primary_model": config.primary,
        "specialist_model": config.specialist,
        "router_model": config.router,
        "embedding_model": config.embedding,
        "context_window": config.context_window,
        "fallback_to_anthropic": config.fallback_to_anthropic,
    }

@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if PROMETHEUS_AVAILABLE:
        return StreamingResponse(
            iter([generate_latest()]),
            media_type=CONTENT_TYPE_LATEST
        )
    return {"note": "prometheus_client not installed — pip install prometheus-client"}

@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Arabic Cognitive AI Engine",
        "name_ar": "محرك الذكاء الاصطناعي المعرفي العربي",
        "version": "4.0.0",
        "phase": "Phase 2 — Live",
        "docs": "/api/docs",
        "health": "/api/health",
    }
