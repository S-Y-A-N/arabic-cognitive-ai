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


def ollama_call(prompt: str | Dict[str, Dict[str, str]], system: str = "", model: str = None):
    model = model or PRIMARY_MODEL
    try:
        messages = []
        # model='gemma3:1b-it-qat' created_at='2026-05-11T13:11:10.045441322Z' done=False done_reason=None total_duration=None load_duration=None prompt_eval_count=None prompt_eval_duration=None eval_count=None eval_duration=None message=Message(role='assistant', content='Hello', thinking=None, images=None, tool_name=None, tool_calls=None) logprobs=None
        if system:
            messages.append({'role': 'system', 'content': system})
        if isinstance(prompt, Dict):
            for p in prompt.values():
                prompt = p['prompt']
                system = p['system']
                messages.extend([
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': prompt}
                ])
        print(messages)
        stream = ollama.chat(
            model=model,
            stream=True,
            messages=messages,
            options={
                'num_predict': 1000,
            }
        )
        for chunk in stream:
            print(chunk)
            yield chunk['message']['content']
        # if chunk['done']: return # TODO CHECK
    except Exception as e:
        log.error(f"Ollama error: {e}")
        

async def build_agent_prompt(agent_id: str, query: str,
                        prev_context: str = "",
                        memory_ctx: str = "",
                        rag_ctx: str = ""):
    """Builds prompt for an agent."""
    
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
            
        return {
            'system': system,
            'prompt': prompt
        }

    # Build prompt with all context layers
    parts = []
    if memory_ctx:   parts.append(f"[ذاكرة]\n{memory_ctx}")
    if rag_ctx:      parts.append(f"[وثائق]\n{rag_ctx}")
    if prev_context: parts.append(f"[مخرجات سابقة]\n{prev_context}")
    parts.append(query)
    prompt = "\n\n".join(parts)
    return {
        'system': system,
        'prompt': prompt
    }


