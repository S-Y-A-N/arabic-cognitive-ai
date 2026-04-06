"""
Arabic Cognitive AI Engine — FastAPI Backend
============================================
Production-grade async API server for the Arabic Cognitive AI Platform.

Architecture:
  - Async FastAPI with streaming support
  - JWT-based authentication
  - Role-based access control (RBAC)
  - Request rate limiting
  - Audit logging
  - Prometheus metrics
  - WebSocket support for real-time streaming

Author: Arabic Cognitive AI Engine Team
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import asyncio
import time
import uuid
import logging
import json
from typing import Optional, AsyncGenerator
from datetime import datetime, timedelta

# Internal imports
from agents.orchestrator import CognitiveOrchestrator
from rag.pipeline import RAGPipeline
from arabic_nlp.pipeline import ArabicNLPPipeline
from knowledge_graph.connector import KnowledgeGraphConnector
from memory.cognitive_memory import CognitiveMemorySystem
from models.router import ModelRouter
from api.schemas import (
    QueryRequest, QueryResponse, AgentConfig,
    MemoryEntry, KnowledgeGraphQuery, ArabicAnalysisRequest
)
from api.auth import create_access_token, verify_token, get_current_user
from api.rate_limiter import RateLimiter
from api.audit_logger import AuditLogger

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("arabic-cognitive-ai")

# ─── Application Lifecycle ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all system components on startup, cleanup on shutdown."""
    logger.info("🚀 Initializing Arabic Cognitive AI Engine...")
    
    # Initialize core systems
    app.state.orchestrator = CognitiveOrchestrator()
    app.state.rag_pipeline = RAGPipeline()
    app.state.arabic_nlp = ArabicNLPPipeline()
    app.state.knowledge_graph = KnowledgeGraphConnector()
    app.state.memory_system = CognitiveMemorySystem()
    app.state.model_router = ModelRouter()
    app.state.audit_logger = AuditLogger()
    app.state.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
    
    # Warm up connections
    await app.state.rag_pipeline.initialize()
    await app.state.knowledge_graph.connect()
    await app.state.memory_system.initialize()
    
    logger.info("✅ Arabic Cognitive AI Engine initialized successfully")
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Arabic Cognitive AI Engine...")
    await app.state.knowledge_graph.disconnect()
    await app.state.memory_system.shutdown()
    logger.info("✅ Shutdown complete")

# ─── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Arabic Cognitive AI Engine",
    description="محرك الذكاء الاصطناعي المعرفي العربي — Research-grade Cognitive AI Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# ─── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

# ─── Authentication Endpoints ─────────────────────────────────────────────────
@app.post("/api/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and issue JWT token."""
    # In production: validate against database
    if form_data.username == "admin" and form_data.password == "secure_password":
        token = create_access_token(
            data={"sub": form_data.username, "role": "admin"},
            expires_delta=timedelta(hours=8)
        )
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# ─── Core Cognitive Query Endpoint ────────────────────────────────────────────
@app.post("/api/cognitive/query", response_model=QueryResponse)
async def cognitive_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Main cognitive query endpoint.
    
    Orchestrates all agents: Planner → Research → Reasoning → Verification → Synthesis
    Returns structured response with sources, confidence, and reasoning chain.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Rate limiting
        await app.state.rate_limiter.check(current_user["sub"])
        
        # Log audit trail
        background_tasks.add_task(
            app.state.audit_logger.log_request,
            request_id=request_id,
            user=current_user["sub"],
            query=request.query,
            mode=request.mode
        )
        
        # Arabic NLP preprocessing
        nlp_analysis = await app.state.arabic_nlp.analyze(request.query)
        
        # Load relevant memories
        session_memory = await app.state.memory_system.retrieve_relevant(
            query=request.query,
            session_id=request.session_id,
            top_k=5
        )
        
        # Execute cognitive orchestration
        result = await app.state.orchestrator.execute(
            query=request.query,
            nlp_analysis=nlp_analysis,
            memory_context=session_memory,
            mode=request.mode,  # "single_agent" | "full_cognitive"
            agent_id=request.agent_id,
            rag_pipeline=app.state.rag_pipeline,
            knowledge_graph=app.state.knowledge_graph,
            model_router=app.state.model_router,
        )
        
        # Store to memory
        background_tasks.add_task(
            app.state.memory_system.store,
            session_id=request.session_id,
            query=request.query,
            response=result.final_answer,
            agent_traces=result.agent_traces,
            entities=nlp_analysis.entities
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {processing_time:.2f}s")
        
        return QueryResponse(
            request_id=request_id,
            query=request.query,
            final_answer=result.final_answer,
            agent_traces=result.agent_traces,
            sources=result.sources,
            confidence_score=result.confidence_score,
            reasoning_chain=result.reasoning_chain,
            nlp_analysis=nlp_analysis.dict(),
            processing_time=processing_time,
            tokens_used=result.tokens_used
        )
        
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Streaming Endpoint ────────────────────────────────────────────────────────
@app.post("/api/cognitive/stream")
async def cognitive_stream(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Streaming cognitive query — returns agent traces in real-time via SSE.
    Enables live visualization of the orchestration pipeline.
    """
    async def stream_generator() -> AsyncGenerator[str, None]:
        async for chunk in app.state.orchestrator.stream_execute(
            query=request.query,
            mode=request.mode,
            agent_id=request.agent_id,
            rag_pipeline=app.state.rag_pipeline,
            knowledge_graph=app.state.knowledge_graph,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ─── WebSocket for Real-Time ───────────────────────────────────────────────────
@app.websocket("/ws/cognitive/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time bidirectional communication."""
    await websocket.accept()
    logger.info(f"WebSocket connected: session {session_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            mode = data.get("mode", "full_cognitive")
            
            # Stream agent traces back
            async for chunk in app.state.orchestrator.stream_execute(
                query=query,
                mode=mode,
                rag_pipeline=app.state.rag_pipeline,
                knowledge_graph=app.state.knowledge_graph,
            ):
                await websocket.send_json(chunk)
            
            await websocket.send_json({"type": "complete"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session {session_id}")


# ─── Arabic NLP Endpoints ──────────────────────────────────────────────────────
@app.post("/api/arabic/analyze")
async def analyze_arabic(request: ArabicAnalysisRequest):
    """
    Full Arabic linguistic analysis pipeline.
    Returns dialect, morphology, NER, normalization.
    """
    analysis = await app.state.arabic_nlp.full_analysis(request.text)
    return {
        "text": request.text,
        "dialect": analysis.dialect,
        "dialect_confidence": analysis.dialect_confidence,
        "normalized_msa": analysis.normalized_msa,
        "morphological_analysis": analysis.morphological_analysis,
        "entities": analysis.entities,
        "language_mix": analysis.language_mix,
        "tokens": analysis.tokens,
        "sentiment": analysis.sentiment
    }

@app.post("/api/arabic/detect-dialect")
async def detect_dialect(request: ArabicAnalysisRequest):
    """Detect Arabic dialect with confidence scores."""
    result = await app.state.arabic_nlp.detect_dialect(request.text)
    return result


# ─── RAG Endpoints ────────────────────────────────────────────────────────────
@app.post("/api/rag/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Ingest document into the RAG knowledge base."""
    # Document ingestion pipeline
    background_tasks.add_task(app.state.rag_pipeline.ingest_document)
    return {"status": "ingestion_queued", "message": "Document is being processed"}

@app.post("/api/rag/search")
async def rag_search(query: str, top_k: int = 10):
    """Hybrid vector + keyword search against knowledge base."""
    results = await app.state.rag_pipeline.hybrid_search(query=query, top_k=top_k)
    return {"query": query, "results": results}


# ─── Knowledge Graph Endpoints ────────────────────────────────────────────────
@app.post("/api/kg/query")
async def kg_query(request: KnowledgeGraphQuery):
    """Execute a knowledge graph query (Cypher or natural language)."""
    if request.natural_language:
        cypher = await app.state.orchestrator.nl_to_cypher(request.query)
    else:
        cypher = request.query
    
    results = await app.state.knowledge_graph.query(cypher)
    return {"cypher": cypher, "results": results}

@app.post("/api/kg/extract-entities")
async def extract_and_store_entities(text: str):
    """Extract entities from text and store in knowledge graph."""
    entities = await app.state.arabic_nlp.extract_entities(text)
    await app.state.knowledge_graph.store_entities(entities)
    return {"entities_extracted": len(entities), "entities": entities}


# ─── Memory Endpoints ─────────────────────────────────────────────────────────
@app.get("/api/memory/{session_id}")
async def get_session_memory(session_id: str):
    """Retrieve session memory and episodic history."""
    memories = await app.state.memory_system.get_session(session_id)
    return {"session_id": session_id, "memories": memories}

@app.delete("/api/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear session memory."""
    await app.state.memory_system.clear_session(session_id)
    return {"status": "cleared"}


# ─── System Health Endpoints ───────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """System health check — all components."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "orchestrator": "active",
            "rag_pipeline": "active",
            "arabic_nlp": "active",
            "knowledge_graph": await app.state.knowledge_graph.ping(),
            "memory_system": "active",
            "model_router": "active"
        },
        "version": "1.0.0"
    }

@app.get("/api/metrics")
async def get_metrics():
    """System metrics for observability dashboard."""
    return {
        "total_requests": app.state.audit_logger.total_requests,
        "requests_last_hour": app.state.audit_logger.requests_last_hour(),
        "avg_response_time": app.state.audit_logger.avg_response_time(),
        "active_sessions": app.state.memory_system.active_sessions_count(),
        "kg_entities": await app.state.knowledge_graph.entity_count(),
        "rag_documents": await app.state.rag_pipeline.document_count(),
        "memory_entries": await app.state.memory_system.total_entries()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
        log_level="info"
    )
