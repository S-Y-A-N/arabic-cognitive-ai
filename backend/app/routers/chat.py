from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories.chat import ChatRepository

router = APIRouter(prefix="/chat", tags=["chat"])

# get chat history by session id (right now equals agent Id)
@router.get("/{session_id}")
def get_chat(session_id: str, db: Session = Depends(get_db)):
    chat_repo = ChatRepository(db)
    chat = chat_repo.get_or_create_session(session_id=session_id, user_id=1)
    messages = chat_repo.get_history(chat.chat_id)
    return {"messages": messages}

# delete chat history
@router.delete("/{session_id}")
def clear_chat(session_id: str, db: Session = Depends(get_db)):
    chat_repo = ChatRepository(db)
    chat_repo.clear_session(session_id)
    return {"status": "cleared"}