from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker
from app.core.config import DATABASE_URL
from app.core.logger import log

# engine for connection
engine = create_engine(DATABASE_URL)
# session to interact with data
SessionLocal = sessionmaker(echo=True, autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
  pass

def init_db():
  from app.models.user import User
  from app.models.chat import Chat
  from app.models.query import Query
  from app.models.response import Response
  from app.models.agent import Agent
  from app.models.llm import LLM

  Base.metadata.create_all(engine, checkfirst=True)
  