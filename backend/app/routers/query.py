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
    body = await request.json()
    prompt = body.get("query","").strip()
    mode = body.get("mode", "auto")
    session_id = body.get("session_id", "default")
    if not prompt: raise HTTPException(400, "Empty prompt")
    model = PRIMARY_MODEL
    
    chat_repo = ChatRepository(db)
    
    # TODO for now only one user, later implement multi-user
    chat = chat_repo.get_or_create_session(session_id=session_id, user_id=1)
    
    # 1. Get chat history before saving query
    message_history = chat_repo.get_history(chat.chat_id)
    
    # 2. Save query to database
    saved_query = chat_repo.save_query(chat.chat_id, prompt, session_id)
    
    # 3. Orchestrator:
    # if Auto -> returns final output [no streaming]
    # if Single -> returns Ollama `messages` (prompts) [streaming]
    orch = await orchestrate(query=prompt, mode=mode, message_history=message_history)
    
    if mode == "auto":
        return {"answer": orch["output"], "pipeline": orch["pipeline"]}  
      
    
    messages = orch["messages"]    
    print(f"Prompts messages: {messages}")

    async def stream_and_save(messages, model):
        full_response = []
        for chunk in ollama_call(model=model, messages=messages, stream=True):
            full_response.append(chunk)
            yield chunk
        chat_repo.save_response(saved_query.query_id, "".join(full_response), mode)
    
    return StreamingResponse(stream_and_save(messages, model=model), media_type="text/plain")
   
# non-streaming 
@router.post("/")
async def query_sync(request: Request):
    pass