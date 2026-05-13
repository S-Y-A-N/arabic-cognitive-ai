from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.llm import LLM
  from app.db.models.query import Query
  from app.db.models.response import Response

from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class Agent(Base):  
  __tablename__ = "agents"
  
  agent_id: Mapped[str] = mapped_column(primary_key=True, index=True)
  agent_name: Mapped[str] = mapped_column(index=True)
  description: Mapped[str] = mapped_column(index=True)
  # foreign keys
  llm_id: Mapped[int] = mapped_column(ForeignKey("llms.llm_id"))
  
  # relationships
  llm: Mapped["LLM"] = relationship(back_populates="agents")
  queries: Mapped[List["Query"]] = relationship(back_populates="agent")
  responses: Mapped[List["Response"]] = relationship(back_populates="agent")
