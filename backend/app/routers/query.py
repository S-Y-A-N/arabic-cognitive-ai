import json
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.chat import ChatRepository
from app.services.orchestrator import orchestrate
from app.services.agent import ollama_streaming
from app.core.config import PRIMARY_MODEL
from app.core.agent_config import AGENT_MODELS

router = APIRouter(prefix="/query", tags=["query"])

# streaming
@router.post("/stream", response_class=StreamingResponse)
async def stream_query(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    prompt = body.get("query","").strip()
    if not prompt: raise HTTPException(400, "Empty prompt")
    mode = body.get("mode", "auto")
    agent_id = mode.split(":")[1] # get agent id from mode 'single:agent_id'
    session_id = body.get("session_id")
    model = AGENT_MODELS.get(agent_id, PRIMARY_MODEL)
    
    
    chat_repo = ChatRepository(db)
    # TODO for now only one user, later implement multi-user
    chat = chat_repo.get_or_create_session(session_id=session_id, user_id=1)
    
    # 1. Get chat history before saving query
    message_history = chat_repo.get_history(chat.chat_id)
    
    # 2. Orchestrator:
    # if Auto -> returns final output [no streaming]
    # if Single -> returns Ollama `messages` (prompts) [streaming]
    orch = await orchestrate(query=prompt, mode=mode, message_history=message_history)
        
    messages = orch["messages"]    
    print(f"Prompts messages: {messages}")

    async def stream_and_save(messages, model):
        full_response = []
        for chunk in ollama_streaming(model=model, messages=messages):
            full_response.append(chunk)
            yield chunk
        # 3. Save query and response to database
        saved_query = chat_repo.save_query(chat.chat_id, prompt, session_id)
        chat_repo.save_response(saved_query.query_id, "".join(full_response), session_id)
    
    return StreamingResponse(stream_and_save(messages, model=model), media_type="text/plain")
   
# blocking (auto)
@router.post("", response_class=StreamingResponse)
async def query_sync(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    prompt = body.get("query","").strip()
    if not prompt: raise HTTPException(400, "Empty prompt")
    mode = body.get("mode", "auto")
    session_id = body.get("session_id")
    
    chat_repo = ChatRepository(db)
    
    # TODO for now only one user, later implement multi-user
    chat = chat_repo.get_or_create_session(session_id=session_id, user_id=1)
    
    # 1. Get chat history before saving query
    message_history = chat_repo.get_history(chat.chat_id)
    
    # 2. Orchestrator:
    # if Auto -> returns final output [no streaming]
    # if Single -> returns Ollama `messages` (prompts) [streaming]
    orch = await orchestrate(query=prompt, mode=mode, message_history=message_history)
        
    output = orch["output"]  
    pipeline = orch["pipeline"]  
    print(f"Output: {output}")
    
    async def stream_and_save(output):
        full_response = []
        for chunk in output:
            full_response.append(chunk)
            yield chunk
        # 3. Save query and response to database
        saved_query = chat_repo.save_query(chat.chat_id, prompt, session_id)
        chat_repo.save_response(saved_query.query_id, "".join(full_response), session_id)
        metadata = { "pipeline": pipeline }
        yield "\n__METADATA__" + json.dumps(metadata)
    
    return StreamingResponse(stream_and_save(output), media_type="text/plain")
