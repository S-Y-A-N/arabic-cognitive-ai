from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.schemas.query import QueryCreate
from app.services.agent import ddg_search
from app.dependencies import get_db


router = APIRouter(prefix="/query", tags=["query"])

@router.post("search")
async def search(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    if not query: return {"sources":[], "has_web":False}
    sources = await ddg_search(query, n=5)
    return {"sources": sources, "has_web": bool(sources), "query": query}