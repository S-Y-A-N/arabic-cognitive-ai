from sqlalchemy.orm import Session
from app.db.models.chat import Chat
from app.db.models.query import Query
from app.db.models.response import Response

class ChatRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_session(self, session_id: str, user_id: int) -> Chat:
        chat = self.db.query(Chat).filter(Chat.session_id == session_id).first()
        if not chat:
            chat = Chat(session_id=session_id, user_id=user_id)
            self.db.add(chat)
            self.db.commit()
            self.db.refresh(chat)
        return chat

    def save_query(self, chat_id: int, prompt: str, agent_id: int) -> Query:
        q = Query(chat_id=chat_id, query=prompt, agent_id=agent_id)
        self.db.add(q)
        self.db.commit()
        self.db.refresh(q)
        return q

    def save_response(self, query_id: int, content: str, agent_id: int) -> Response:
        r = Response(query_id=query_id, response=content, agent_id=agent_id)
        self.db.add(r)
        self.db.commit()
        self.db.refresh(r)
        return r

    def get_history(self, session_id: str, limit: int = 10) -> list[Query]:
        chat = self.db.query(Chat).filter(Chat.session_id == session_id).first()
        if not chat:
            return []
        return (self.db.query(Query)
                .filter(Query.chat_id == chat.chat_id)
                .order_by(Query.query_id.desc())
                .limit(limit).all())