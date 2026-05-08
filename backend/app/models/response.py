from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base
from app.models.query import Query
from app.models.agent import Agent

class Response(Base):
  __tablename__ = "responses"
  response_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  response: Mapped[int]  = mapped_column(index=True)
  
  # foreign keys
  query_id: Mapped[int]  = mapped_column(ForeignKey("query.query_id"))
  query: Mapped["Query"] = relationship(back_populates="response")
  
  agent_id: Mapped[int]  = mapped_column(ForeignKey("agent.agent_id"))
  