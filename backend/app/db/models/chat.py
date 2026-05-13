from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.user import User
  from app.db.models.query import Query
  
from typing import List
from datetime import datetime
from sqlalchemy import ForeignKey, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column,relationship
from app.db.base import Base

class Chat(Base):  
  __tablename__ = "chats"
  
  chat_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  session_id: Mapped[str] = mapped_column(String, unique=True, index=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

  # foreign keys
  user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
  
  # relationships
  user: Mapped["User"] = relationship(back_populates="chats")
  queries: Mapped[List["Query"]] = relationship(back_populates="chat")