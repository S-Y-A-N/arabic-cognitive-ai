from typing import List, Dict

from app.core.logger import log
from app.core.orchestrator_config import *
from app.services.agent import build_agent_prompt
from app.core.agent_config import AGENT_LABELS


async def orchestrate(query: str, mode: str = "auto",
                      session_id: str = "default") -> dict:
    """Full orchestration: memory → RAG → pipeline → merge → save."""
    # 1. Memory retrieval
    # mem_ctx  = memory.get_context(query, limit=3)
    # mem_used = bool(mem_ctx)

    # 2. RAG retrieval
    # rag_ctx  = rag.get_rag_context(query, k=2)
    # rag_used = bool(rag_ctx)

    # 3. Intent + pipeline
    intent   = classify_intent(query)
    pipeline = build_pipeline(intent, mode)
    

    # 4. Sequential execution (context accumulates)
    prompts:  Dict[str, str] = {}
    acc_ctx:  str = ""
    for agent_id in pipeline:
        agent_prompt = await build_agent_prompt(
            agent_id   = agent_id,
            query      = query,
            prev_context = acc_ctx,
            # memory_ctx = mem_ctx if not acc_ctx else "",
            # rag_ctx    = rag_ctx  if not acc_ctx else "",
        )
        prompts[agent_id] = agent_prompt

    return {
        'pipeline': pipeline,
        'prompts': prompts,
        'intent': intent,
        # "memory_used": mem_used,
        # "rag_used":    rag_used,
        # "agents":      {k: v[:200] + "..." if len(v) > 200 else v
        #                 for k, v in streams.items()},
    }

    # 6. Save to memory + log
    # memory.save("orchestrator", query, final, quality=3)
    # memory.log_experiment(query,
    #                       "with_memory" if mem_used else "without_memory",
    #                       pipeline, latency)

    
def classify_intent(query: str) -> dict:
    q = query.lower()
    ar = bool(AR_RE.search(q))
    wc = len(q.split())
    return {
        "research":    any(k in q for k in RESEARCH_KW),
        "gcc_law":     any(k in q for k in GCC_KW),
        "dialect":     ar or any(k in q for k in DIALECT_KW),
        "reasoning":   wc > 12 or any(k in q for k in REASON_KW),
        "extraction":  any(k in q for k in EXTRACT_KW),
        "is_arabic":   ar,
    }


def build_pipeline(intent: dict, mode: str = "auto") -> List[str]:
    # Single-agent override
    if mode.startswith("single:"):
        a = mode.split(":", 1)[1]
        return [a, "muraqib"] if a != "muraqib" else [a]
    # Legacy mode aliases
    if mode == "arabic_nlp":    return ["lughawi", "muraqib"]
    if mode == "knowledge":     return ["bani", "muraqib"]
    if mode == "deep_research": return ["bahith", "hakeem", "muraqib"]
    if mode == "cognitive":     pass  # fall through to auto

    p = []
    if intent["research"]:   p.append("bahith")
    if intent["gcc_law"]:    p.append("musheer")
    if intent["dialect"]:    p.append("lughawi")
    if intent["reasoning"] or len(p) > 1: p.append("hakeem")
    if intent["extraction"]: p.append("bani")
    p.append("muraqib")
    if len(p) == 1: p = ["hakeem", "muraqib"]

    seen, out = set(), []
    for a in p:
        if a not in seen: seen.add(a); out.append(a)
    return out

def merge_pipeline_outputs(pipeline: List[str], outputs: Dict[str, str]) -> str:
    valid = {a: o for a, o in outputs.items() if o.status_code == 200}
    if not valid: return "لم أتمكن من توليد إجابة."
    if len(valid) == 1: return next(iter(valid.values()))
    parts = []
    for a in pipeline:
        if a != "muraqib" and a in valid:
            parts.append(f"### {AGENT_LABELS.get(a, a)}\n{valid[a]}")
    if "muraqib" in valid:
        parts.append(f"\n---\n### 🔍 مراقب\n{valid['muraqib']}")
    return "\n\n".join(parts)