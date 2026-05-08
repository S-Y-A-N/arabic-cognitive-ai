from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base

from app.models.chat import Chat
from app.models.agent import Agent
from app.models.response import Response

class Query(Base):
  __tablename__ = "queries"
  query_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  query: Mapped[str] = mapped_column(primary_key=True, index=True)

  # foreign keys
  chat_id: Mapped[int] = mapped_column(ForeignKey("chats.chat_id"))
  
  agent_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  agent: Mapped["Agent"] = relationship(back_populates="queries")
  
  # child
  response: Mapped["Response"] = relationship(back_populates="query")
  