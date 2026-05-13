from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from app.db.models.chat import Chat
  
from typing import List
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

class User(Base):
  __tablename__ = "users"
  
  user_id: Mapped[int] = mapped_column(primary_key=True)
  name: Mapped[str] = mapped_column(String(30), nullable=False)
  email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
  password: Mapped[str] = mapped_column(String, nullable=False)
  
  # relationships
  chats: Mapped[List["Chat"]] = relationship(back_populates="user")
  
  
  
  
  