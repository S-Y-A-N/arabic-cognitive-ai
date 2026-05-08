import httpx
from typing import List, Dict
from app.core.logger import log
from app.core.config import *
from app.core.agent_config import *

async def ddg_search(query: str, n: int = 5) -> list:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            results = list(d.text(query, max_results=n))
            return [{"title": r.get("title",""), "url": r.get("href",""),
                     "snippet": r.get("body","")[:400]} for r in results if r.get("body")]
    except ImportError:
        log.warning("pip install ddgs")
        return []
    except Exception as e:
        log.warning(f"DDG: {e}")
        return []


async def ollama_call(prompt: str, system: str = "", model: str = None) -> str:
    model = model or PRIMARY_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1500
        }
    }
    if system: payload["system"] = system
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            print(payload)
            res = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            res.raise_for_status()
            if res.status_code == 200:
                return res.json().get("response", "")
    except Exception as e:
        log.error(f"Ollama error: {e}")
    return ""

async def execute_agent(agent_id: str, query: str,
                        prev_context: str = "",
                        memory_ctx: str = "",
                        rag_ctx: str = "") -> str:
    """Execute one agent. Builds prompt, calls LLM, returns text."""
    system = SYSTEM_PROMPTS.get(agent_id, "أنت مساعد ذكي.")
    model  = AGENT_MODELS.get(agent_id, PRIMARY_MODEL)

    # باحث uses web search
    if agent_id == "bahith":
        search = await ddg_search(query, n=5)
        if search:
            ctx = "\n\n".join(
                f"Source: {s['title']}\nURL: {s['url']}\n{s['snippet']}"
                for s in search
            )
            prompt = f"{ctx}\n\nبناءً على النتائج أعلاه، أجب على: {query}"
        else:
            prompt = query
        return await ollama_call(prompt, system, model)

    # Build prompt with all context layers
    parts = []
    if memory_ctx:   parts.append(f"[ذاكرة]\n{memory_ctx}")
    if rag_ctx:      parts.append(f"[وثائق]\n{rag_ctx}")
    if prev_context: parts.append(f"[مخرجات سابقة]\n{prev_context}")
    parts.append(query)
    return await ollama_call("\n\n".join(parts), system, model)



def merge_pipeline_outputs(pipeline: List[str], outputs: Dict[str, str]) -> str:
    valid = {a: o for a, o in outputs.items() if o and not o.startswith("[خطأ")}
    # if not valid: return "لم أتمكن من توليد إجابة."
    if len(valid) == 1: return next(iter(valid.values()))
    parts = []
    for a in pipeline:
        if a != "muraqib" and a in valid:
            parts.append(f"### {AGENT_LABELS.get(a, a)}\n{valid[a]}")
    if "muraqib" in valid:
        parts.append(f"\n---\n### 🔍 مراقب\n{valid['muraqib']}")
    return "\n\n".join(parts)
