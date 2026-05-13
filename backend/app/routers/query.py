from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.chat import ChatRepository
from app.services.orchestrator import orchestrate
from app.services.agent import ollama_call

from app.core.logger import log
from app.core.config import PRIMARY_MODEL

router = APIRouter(prefix="/query", tags=["query"])


# streaming
@router.post("/stream", response_class=StreamingResponse)
async def stream_query(request: Request, db: Session = Depends(get_db)):
    body   = await request.json()
    prompt = body.get("query","").strip()
    mode   = body.get("mode", "auto")
    sid    = body.get("session_id", "default")
    if not prompt: raise HTTPException(400, "Empty prompt")
    
    # Save query to chat
    chat_repo = ChatRepository(db)
    # TODO for now only one user, later implement multi-user
    chat = chat_repo.get_or_create_session(session_id=sid, user_id=1)
    saved_query = chat_repo.save_query(chat.chat_id, prompt, mode)
    model = PRIMARY_MODEL
    
    # Orchestrator: returns pipeline, prmopt per agent
    orch = await orchestrate(prompt, mode, sid)
    agent_prompts = orch['prompts']
    
    async def stream_and_save():
        full_response = []
        async for chunk in ollama_call(agent_prompts, model=model):
            full_response.append(chunk)
            yield chunk
        chat_repo.save_response(saved_query.query_id, "".join(full_response))
    
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