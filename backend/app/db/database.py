from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import DATABASE_URL

# engine for connection
engine = create_engine(
  DATABASE_URL,
  echo=True,
  connect_args={"check_same_thread": False},
)

# session to interact with data
SessionLocal = sessionmaker(
  autocommit=False,
  autoflush=False,
  bind=engine
)


def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()
