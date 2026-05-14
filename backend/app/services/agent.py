from fastapi.responses import StreamingResponse
from typing import List, Dict
import json
from app.core.logger import log
from app.core.config import *
from app.core.agent_config import *
import ollama

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


def ollama_streaming(
        model: str = PRIMARY_MODEL,
        messages: list[dict] = [],
    ):
    try:
        # model='gemma3:1b-it-qat' created_at='2026-05-11T13:11:10.045441322Z' done=False done_reason=None total_duration=None load_duration=None prompt_eval_count=None prompt_eval_duration=None eval_count=None eval_duration=None message=Message(role='assistant', content='Hello', thinking=None, images=None, tool_name=None, tool_calls=None) logprobs=None
        response = ollama.chat(
            model=model,
            stream=True,
            messages=messages,
            options={
                'num_predict': 1000,
            }
        )
        for chunk in response:
            yield chunk['message']['content']
        if chunk['done']: return # TODO CHECK
    except Exception as e:
        log.error(f"Ollama error: {e}")
        
def ollama_blocking(
        model: str = PRIMARY_MODEL,
        messages: list[dict] = [],
    ):
    try:
        response = ollama.chat(
            model=model,
            stream=False,
            messages=messages,
            options={
                'num_predict': 1000,
            }
        )
        return response['message']['content']
    except Exception as e:
        log.error(f"Ollama error: {e}")
        

async def build_agent_prompt(
        agent_id: str,
        query: str,
        prev_context: str = "",
        rag_ctx: str = ""
    ):
    """Builds the full prompt using agent system prompt with the query, previous context, and RAG."""
    
    # 1. Get System Prompt
    system = SYSTEM_PROMPTS.get(agent_id, "أنت مساعد ذكي.")

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
                    
        return system, prompt

    # Build prompt with all context layers
    parts = []
    if prev_context: parts.append(f"[مخرجات سابقة]\n{rag_ctx}")
    if rag_ctx: parts.append(f"[وثائق]\n{rag_ctx}")
    parts.append(query)
    prompt = "\n\n".join(parts)

    return system, prompt


def build_ollama_messages(
        system: str | None = None,
        prompt: str | None = None,
        message_history: list[dict[str, str]] | None = None
    ) -> list[dict[str, str]]:
    messages = []
    
    if system:
        messages.append({'role': 'system', 'content': system})
        
    if message_history:
        messages.extend(message_history)
    
    if prompt:
        messages.append({'role': 'user', 'content': prompt})
        
    return messages
