from app.core.orchestrator_config import *
from app.services.agent import build_agent_prompt, build_ollama_messages, ollama_call
from app.core.agent_config import AGENT_LABELS, AGENT_MODELS
from app.core.config import PRIMARY_MODEL


async def orchestrate(
        query: str,
        message_history: list[dict] | None = None,
        mode: str = "auto"
    ) -> dict:
    """Full orchestration: memory → RAG → pipeline → merge → save."""
    
    # TODO RAG retrieval
    # rag_context  = rag.get_rag_context(query, k=2)
    # rag_used = bool(rag_context)

    # Auto: Intent + pipeline
    if mode == "auto":
        intent = classify_intent(query)
        pipeline = build_pipeline(intent, mode)
        
        outputs: dict[str, str] = {} # messages per agent id
        
        acc_context = "" # accumulated context
        for agent_id in pipeline:
            model = AGENT_MODELS.get(agent_id, PRIMARY_MODEL)
            # build agent prompt (return system & built prompt as Ollama messages)
            system, prompt = await build_agent_prompt(
                agent_id=agent_id,
                query=query,
                prev_context=acc_context
                # rag_context = rag_context if not acc_context else "",
            )

            # build the messages structure for ollama, then execute agent
            messages = build_ollama_messages(system=system, prompt=prompt, message_history=message_history)
            agent_response = ollama_call(model=model, messages=messages, stream=False)

            # save agent response as context for next agent, and save the response
            acc_context = agent_response
            outputs[agent_id] = agent_response

        # merge agents outputs
        final_output = merge_pipeline_outputs(outputs)
        
        return {
            'output': final_output,
            'pipeline': pipeline,
            'intent': intent,
        }
        
    # One agent, enables response streaming...
    
    agent_id = mode.split(":")[1]
    model = AGENT_MODELS.get(agent_id, PRIMARY_MODEL)
    
    # build agent prompt (return system & prompt)
    system, prompt = await build_agent_prompt(
        agent_id=agent_id,
        query=query,
    )
                
    # build the messages structure for ollama, then return it for streaming
    messages = build_ollama_messages(system=system, prompt=prompt, message_history=message_history)
    
    return {
        'messages': messages,
    }
        
    # TODO sequential agent pipeline
    # pipeline: first get built prompt then run first agent then get response and give to next agent, then stream response to user

    
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


def build_pipeline(intent: dict, mode: str = "auto") -> list[str]:
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

def merge_pipeline_outputs(outputs: dict[str, str]) -> str:
    valid_outputs = {a: o for a, o in outputs.items() if o}
    
    if not valid_outputs: return "لم أتمكن من توليد إجابة."
    if len(valid_outputs) == 1: return next(iter(valid_outputs.values()))

    parts = []
    for agent_id in outputs.keys():
        if agent_id != "muraqib":
            parts.append(f"### {AGENT_LABELS.get(agent_id, agent_id)}\n{valid_outputs[agent_id]}")
            
    if "muraqib" in valid_outputs:
        parts.append(f"\n---\n### 🔍 مراقب\n{valid_outputs['muraqib']}")
    
    merged_outputs = "\n\n".join(parts)
    return merged_outputs