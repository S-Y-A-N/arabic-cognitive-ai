# main entry point to application
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import ollama

from app.core.config import *
from app.core.logger import log
from app.core.security import limiter
from app.db.database import init_db

from app.routers import query, health


async def lifespan(app: FastAPI):
    log.info("🚀 ACAI: محرك الذكاء الاصطناعي المعرفي العربي")
    log.info("✅ ACAI ready")
    
    # database init
    try: init_db(); log.info("Database initizalized.")
    except Exception as e: log.error(f"Database initialization error: {e}")
    
    # model preloading
    try:
        ollama.generate(model=PRIMARY_MODEL)
    except Exception as e:
        log.error(f"Ollama preloading error: {e}")
    
    yield # execute everything before `yield` at startup

# application
app = FastAPI(
    title="ACAI",
    description="Arabic Cognitive AI Engine Backend",
    lifespan=lifespan
)

# middleware
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://localhost(:5173)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # check request path
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    
    # auth
    
    # rate limiting
    ip = request.client.host if request.client else "unknown"
    if not limiter.check(ip):
        return JSONResponse({"error": "Rate limit — 40 req/min"}, status_code=429)
    return await call_next(request)

# routers
app.include_router(health.router)
app.include_router(query.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
