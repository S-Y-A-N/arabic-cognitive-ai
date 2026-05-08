from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.models.query import Query
from app.models.response import Response

class Agent(Base):
  __tablename__ = "agents"
  agent_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  agent_name: Mapped[str] = mapped_column(index=True)
  description: Mapped[str] = mapped_column(index=True)
  
  # foreign keys
  llm_id: Mapped[int] = relationship(ForeignKey("llms.llm_id"))
  
  # children
  queries: Mapped["Query"] = relationship(back_populates="agent")
  responses: Mapped["Response"] = relationship(back_populates="agent")
  
  