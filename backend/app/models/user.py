from typing import List

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base
from app.models.chat import Chat

class User(Base):
  __tablename__ = "users"
  
  user_id: Mapped[int] = mapped_column(primary_key=True)
  name: Mapped[int] = mapped_column(String(30), nullable=False)
  email: Mapped[int] = mapped_column(primary_key=True)
  password: Mapped[int] = mapped_column(hash='bcrypt')
  
  # children (has chats)
  chats: Mapped[List["Chat"]] = relationship()
  
  
  
  
  