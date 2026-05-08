from typing import List

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.models.agent import Agent

class LLM(Base):
  __tablename__ = "llms"
  
  llm_id: Mapped[int] = mapped_column(primary_key=True)
  llm_name: Mapped[str] = mapped_column(index=True)
  config: Mapped[dict] = mapped_column()
  
  # children
  agents: Mapped[List["Agent"]] = relationship()

