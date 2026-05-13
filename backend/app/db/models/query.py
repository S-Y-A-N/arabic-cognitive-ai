from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.chat import Chat
  from app.db.models.agent import Agent
  from app.db.models.response import Response
  
from datetime import datetime
from sqlalchemy import ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class Query(Base):  
  __tablename__ = "queries"
  
  query_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  query: Mapped[str] = mapped_column(index=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
  # foreign keys
  chat_id: Mapped[int] = mapped_column(ForeignKey("chats.chat_id"))
  agent_id: Mapped[int] = mapped_column(ForeignKey("agents.agent_id"))
  
  # relationships
  chat: Mapped["Chat"] = relationship(back_populates="queries")
  agent: Mapped["Agent"] = relationship(back_populates="queries")
  response: Mapped["Response"] = relationship(back_populates="query")
