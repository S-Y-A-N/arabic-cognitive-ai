import json, asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
# from sqlalchemy.orm import Session

# from app.schemas.query import QueryCreate
# from app.services.query import create_query
# from app.services.query import stream_query
# from app.dependencies import get_db
from app.services.orchestrator import orchestrate

router = APIRouter(prefix="/query", tags=["query"])

# streaming
@router.post("/stream")
async def stream_query(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    mode  = body.get("mode","auto")
    sid   = body.get("session_id","default")
    if not query: raise HTTPException(400, "Empty query")


    async def stream_query(query):
        try:
            response = await orchestrate(query, mode, sid)
            answer = response["answer"]
            # Stream in small chunks
            CHUNK = 5
            words = answer.split(" ")
            for i in range(0, len(words), CHUNK):
                text = " ".join(words[i:i+CHUNK]) + " "
                yield f"data: {json.dumps({'type':'chunk','text':text})}\n\n"
                await asyncio.sleep(0.008)
            yield f"data: {json.dumps({'type':'done', 'pipeline': response['pipeline'],})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error', 'error': str(e)})}\n\n"
            
    return StreamingResponse(
        stream_query(query),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )
   
# non-streaming 
@router.post("/")
async def query_sync(request: Request):
    body  = await request.json()
    query = body.get("query","").strip()
    mode  = body.get("mode","auto")
    sid   = body.get("session_id","default")
    if not query: raise HTTPException(400, "Empty query")
    result = await orchestrate(query, mode, sid)
    return {"answer": result["answer"], "pipeline": result["pipeline"],
            "memory_used": result["memory_used"], "latency_ms": result["latency_ms"]}