from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.core.logger import log

def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
    