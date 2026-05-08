import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("API_KEY")
PRIMARY_MODEL  = os.getenv("PRIMARY_MODEL", "qwen2.5:14b-instruct-q4_K_M")
ARABIC_MODEL   = os.getenv("ARABIC_MODEL", "bahraini-pro:latest")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
DATABASE_URL   = os.getenv("DATABASE_URL")

BASE_DIR   = Path(__file__).parent

PUBLIC_PATHS = {"health", "/docs", "/openapi.json", "/", "query/stream"}

