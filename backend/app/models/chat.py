from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column,relationship

from app.db.database import Base
from app.models.query import Query

class Chat(Base):
  __tablename__ = "queries"
  chat_id: Mapped[int] = mapped_column(primary_key=True, index=True)
  
  # foreign keys
  user_id: Mapped[int] = relationship(ForeignKey("users.user_id"))
  
  # children
  queries: Mapped[List["Query"]] = relationship()