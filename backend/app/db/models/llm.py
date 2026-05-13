from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.agent import Agent
  
from typing import List, Optional
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class LLM(Base):  
  __tablename__ = "llms"
  
  llm_id: Mapped[int] = mapped_column(primary_key=True)
  llm_name: Mapped[str] = mapped_column(index=True)
  gen_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
  
  # relattionships
  agents: Mapped[List["Agent"]] = relationship(back_populates="llm")

