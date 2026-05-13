from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.query import Query
  from app.db.models.agent import Agent
  
from datetime import datetime
from sqlalchemy import ForeignKey, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class Response(Base):  
  __tablename__ = "responses"
  
  response_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  response: Mapped[str]  = mapped_column(String, index=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
  
  # foreign keys
  query_id: Mapped[int]  = mapped_column(ForeignKey("queries.query_id"))
  agent_id: Mapped[int]  = mapped_column(ForeignKey("agents.agent_id"))
  
  # relationships
  query: Mapped["Query"] = relationship(back_populates="response")
  agent: Mapped["Agent"]  = relationship(back_populates="responses")
  