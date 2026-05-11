import json, asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
# from sqlalchemy.orm import Session

# from app.schemas.query import QueryCreate
# from app.services.query import create_query
# from app.services.query import stream_query
# from app.dependencies import get_db
from app.services.orchestrator import orchestrate
from app.services.agent import ollama_call

router = APIRouter(prefix="/query", tags=["query"])

from app.core.logger import log
from app.core.config import PRIMARY_MODEL
import ollama
# streaming
@router.post("/stream", response_class=StreamingResponse)
async def stream_query(request: Request):
    body  = await request.json()
    prompt = body.get("query","").strip()
    mode  = body.get("mode","auto")
    sid   = body.get("session_id","default")
    if not prompt: raise HTTPException(400, "Empty prompt")
    
    model = PRIMARY_MODEL
    
    # orchestrator: returns pipeline, prmopt per agent
    orch = await orchestrate(prompt, mode, sid)
    agent_prompts = orch['prompts']
    return StreamingResponse(ollama_call(agent_prompts, model=model), media_type="text/plain")
   
# non-streaming 
@router.post("/")
async def query_sync(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    mode  = body.get("mode","auto")
    sid   = body.get("session_id","default")
    if not query: raise HTTPException(400, "Empty query")
    result = await orchestrate(query, mode, sid)
    return {"answer": result["answer"], "pipeline": result["pipeline"],
            "memory_used": result["memory_used"], "latency_ms": result["latency_ms"]}