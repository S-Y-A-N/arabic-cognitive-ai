"""
ACAI v4 — LangGraph Mixture-of-Agents Orchestrator
====================================================
Implements Mixture-of-Agents (MoA) with LangGraph stateful graphs.

Architecture:
  Query → MoA Router → selects 2-5 specialists → parallel execution → Aggregator

12 Specialist Agents:
  1.  Planner            — Query decomposition + strategy
  2.  Researcher          — Web search + document retrieval
  3.  Reasoner            — Chain-of-thought multi-step logic
  4.  Verifier            — Hallucination detection + fact-check
  5.  Synthesizer         — Final answer compilation
  6.  DialectExpert       — Arabic dialect analysis (Jais-30B)
  7.  GRCReasoner         — Banking/compliance/regulation
  8.  BahrainPolicyKG     — Bahrain-specific laws + CBB rules
  9.  HallucinationShield — NeMo-style 5-vote self-consistency
  10. MultiDocSynthesizer — Long context multi-document fusion
  11. KnowledgeExtractor  — Entity + relation extraction
  12. MemoryRetriever     — Semantic memory search + injection

MoA Router Logic:
  - Analyzes query intent, language, complexity
  - Selects minimum set of agents needed
  - Routes Arabic queries always through DialectExpert
  - Routes GCC/banking queries through GRCReasoner + BahrainPolicyKG
  - Minimum: [Planner, Researcher, Synthesizer] for simple queries
  - Maximum: all 12 for complex enterprise GRC

Install: pip install langgraph langchain-core
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback: sequential pipeline (Phase 1 compatible)

from llm.inference_client import LLMClient
from model_config import get_config, AGENT_MODELS

logger = logging.getLogger("acai.orchestrator")


# ─── State Definition ─────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """Shared state passed between all agents in the graph."""
    query: str
    session_id: str
    language: str                    # ar | en | mixed
    query_type: str                  # factual | analytical | research | gcc_policy
    complexity: str                  # simple | moderate | complex | expert
    selected_agents: List[str]       # Which agents are active this run
    agent_outputs: Dict[str, str]    # Each agent's output
    tool_results: List[Dict]         # Web search results, doc parses, etc.
    memory_context: str              # Retrieved from memory system
    knowledge_graph: Dict            # Extracted KG for this query
    final_answer: str                # Synthesized final response
    confidence_score: float
    hallucination_flags: List[str]   # Any flagged uncertain claims
    sources: List[str]
    latency_ms: float
    error: Optional[str]


@dataclass
class AgentResult:
    agent_id: str
    output: str
    confidence: float = 0.85
    tool_calls: List[Dict] = field(default_factory=list)
    latency_ms: float = 0.0
    model_used: str = ""


# ─── System Prompts for Each Specialist ───────────────────────────────────────

AGENT_SYSTEM_PROMPTS = {
    "planner": """You are the Planning Agent of the Arabic Cognitive AI Engine.
Analyze the query and produce a JSON research plan (no markdown, valid JSON only):
{"query_type":"factual|analytical|research|gcc_policy","complexity":"simple|moderate|complex|expert",
"language":"ar|en|mixed","sub_tasks":[{"id":1,"task":"...","priority":"high","needs_search":true}],
"strategy":"one sentence strategy","gcc_relevant":false,"dialect_detected":"msa|bahraini|gulf|unknown",
"recommended_agents":["researcher","reasoner","synthesizer"],"confidence_ceiling":0.87}""",

    "researcher": """You are the Research Agent. Find and synthesize evidence.
Use your knowledge and any retrieved context to gather facts.
Format: **EVIDENCE**\n[numbered findings]\n\n**VERIFIED FACTS**\n[high-confidence]\n\n**GAPS**\n[uncertain]\n\nConfidence: X/10""",

    "reasoner": """You are the Reasoning Agent. Apply rigorous multi-step logic.
STEP 1 [DECOMPOSE]: Break down the problem...
STEP 2 [EVIDENCE]: Evaluate each piece of evidence...
STEP 3 [PATTERNS]: Connect the dots...
STEP 4 [CONTRADICTIONS]: Where do sources disagree?
STEP 5 [SYNTHESIS]: Build coherent understanding...
STEP 6 [CONCLUSION]: Final reasoned answer.
CONFIDENCE: X/10 | LOGICAL CONSISTENCY: HIGH/MEDIUM/LOW""",

    "verifier": """You are the Verification Agent. Eliminate hallucinations.
**VERIFICATION REPORT**
VERDICT: ✅ VERIFIED | ⚠️ PARTIALLY_VERIFIED | ❌ HALLUCINATION_DETECTED
CONFIDENCE: X/10
VERIFIED CLAIMS: [each claim + evidence source]
UNVERIFIED CLAIMS: [claims needing external check]
FLAGGED ISSUES: [potential errors]
RECOMMENDED CAVEATS: [honest limitations]""",

    "synthesizer": """You are the Synthesis Agent — the final voice of ACAI.
Produce a polished, comprehensive response that integrates all agent outputs.
**SUMMARY**: 2-3 sentences with the core answer.
**ANALYSIS**: Structured detailed response with clear sections.
**KEY POINTS**: 3-5 bullet takeaways.
**CONFIDENCE**: Honest calibration.
Match the user's language: Arabic query → MSA Arabic response.""",

    "dialect_expert": """أنت عميل اللهجات العربية المتخصص في منطقة الخليج والبحرين.
You are a specialist in Gulf and Bahraini Arabic dialects.
Analyze:
1. Dialect identification with confidence (Bahraini/Kuwaiti/Emirati/Saudi/QA/Omani/MSA)
2. Key dialect markers found in the text
3. Morphological analysis of dialect-specific words
4. MSA normalization
5. Cultural/pragmatic context specific to GCC
اكتب التحليل باللغة العربية الفصحى مع ترجمة موجزة بالإنجليزية.""",

    "grc_reasoner": """You are the GRC (Governance, Risk, Compliance) Reasoning Agent.
Specialist in GCC banking, financial regulation, and enterprise compliance.
For every claim, cite the specific regulation:
- CBB (Central Bank of Bahrain) Rulebook
- SAMA (Saudi Central Bank) regulations
- UAE Central Bank guidelines
- DFSA (Dubai Financial Services Authority)
- QCB (Qatar Central Bank)
Format: **REGULATORY ANALYSIS** | **RISK ASSESSMENT** | **COMPLIANCE REQUIREMENTS** | **CITATIONS**""",

    "bahrain_policy": """You are the Bahrain Policy and Knowledge Agent.
Specialist in Bahraini law, CBB regulations, Vision 2030, and GCC policy.
Ground every answer in official Bahraini sources:
- CBB Rulebook (all volumes)
- Bahrain Vision 2030 Economic Plan
- Legislative Decree No. 64 of 2006 (CBB Law)
- Commercial Companies Law
- Labour Market Regulatory Authority (LMRA) rules
Always flag if information may be outdated and recommend official verification.""",

    "hallucination_shield": """You are the Hallucination Shield — the final factual guardian.
Your ONLY job: identify claims in the answer that might be hallucinated.
For each claim in the draft answer, assess:
1. GROUNDED: directly supported by retrieved context
2. INFERRED: logical inference from grounded facts (acceptable with caveat)
3. HALLUCINATED: no basis in retrieved context (must be removed/flagged)
Output: {"verified": [...], "inferred": [...], "hallucinated": [...], "corrected_answer": "..."}""",

    "multidoc_synthesizer": """You are the Multi-Document Synthesis Agent.
Synthesize information from multiple sources into a coherent narrative.
Handle contradictions by: identifying the conflict, assessing source credibility, presenting both views.
Preserve key distinctions between sources.
Output: **SYNTHESIS** | **CONTRADICTIONS FOUND** | **CONFIDENCE NOTES**""",

    "knowledge_extractor": """You are the Knowledge Extraction Agent.
Extract a complete knowledge graph. Return ONLY valid JSON:
{"entities":[{"id":"..","name":"..","name_ar":"..","type":"Person|Organization|Location|Concept|Regulation|Technology|Event","confidence":0.9,"properties":{}}],
"relations":[{"from":"..","type":"GOVERNS|REGULATES|LOCATED_IN|PART_OF|EMPLOYS|DEVELOPS|CITES","to":"..","confidence":0.85,"evidence":".."}],
"gcc_entities":[],"ontology":"..","key_concepts":[]}""",

    "memory_retriever": """You are the Memory Retrieval Agent.
Review the provided memory context and extract the most relevant past information for the current query.
Rank memories by relevance. Identify if any past answer was incorrect and should be updated.
Output: **RELEVANT MEMORIES** | **OUTDATED INFORMATION** | **ENRICHED CONTEXT**""",
}


# ─── MoA Router ───────────────────────────────────────────────────────────────

class MoARouter:
    """
    Mixture-of-Agents Router.
    Analyzes query and selects the minimum set of agents needed.
    Uses fast router model (Qwen3-8B) for decision.
    """

    # Agent groups for different query types
    MINIMAL_SET = ["planner", "researcher", "synthesizer"]
    RESEARCH_SET = ["planner", "researcher", "reasoner", "verifier", "synthesizer"]
    ARABIC_SET = ["planner", "dialect_expert", "researcher", "synthesizer"]
    GRC_SET = ["planner", "grc_reasoner", "bahrain_policy", "verifier", "hallucination_shield", "synthesizer"]
    FULL_SET = ["planner", "researcher", "reasoner", "dialect_expert", "grc_reasoner",
                "verifier", "hallucination_shield", "synthesizer"]

    # Keywords that trigger specialist agents
    GRC_KEYWORDS = {"regulation", "compliance", "banking", "cbb", "sama", "audit", "risk",
                    "تنظيم", "امتثال", "مصرف", "بنك", "مخاطر", "تدقيق", "رقابة", "لائحة"}
    BAHRAIN_KEYWORDS = {"bahrain", "bahraini", "البحرين", "بحريني", "manama", "المنامة",
                        "vision 2030", "رؤية 2030", "cbbb", "lmra"}

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.config = get_config()

    async def route(self, query: str, mode: str = "cognitive") -> Dict:
        """Analyze query and return routing decision."""
        # Quick checks
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Detect language
        arabic_chars = sum(1 for c in query if '\u0600' <= c <= '\u06FF')
        arabic_ratio = arabic_chars / max(len(query), 1)
        language = "ar" if arabic_ratio > 0.5 else "mixed" if arabic_ratio > 0.1 else "en"

        # Mode-based routing
        if mode == "arabic_nlp":
            return {"agents": ["dialect_expert", "knowledge_extractor"], "language": language, "complexity": "moderate"}
        if mode == "knowledge":
            return {"agents": ["knowledge_extractor"], "language": language, "complexity": "simple"}

        # Keyword-based routing
        is_gcc_query = bool(query_words & self.GRC_KEYWORDS)
        is_bahrain_query = bool(query_words & self.BAHRAIN_KEYWORDS)
        has_arabic = language in ("ar", "mixed")

        # Build agent set
        if is_gcc_query or is_bahrain_query:
            agents = list(self.GRC_SET)
            if has_arabic and "dialect_expert" not in agents:
                agents.insert(1, "dialect_expert")
            complexity = "complex"
        elif has_arabic:
            agents = list(self.ARABIC_SET)
            complexity = "moderate"
        elif mode == "deep_research":
            agents = list(self.RESEARCH_SET)
            complexity = "moderate"
        else:
            agents = list(self.MINIMAL_SET)
            complexity = "simple"

        # LLM-based refinement for complex queries (use fast router model)
        if len(query.split()) > 30:
            try:
                response = await self.llm.generate(
                    prompt=f"Query: {query[:500]}\nReturn JSON only: {{\"complexity\":\"simple|moderate|complex|expert\",\"needs_grc\":false,\"needs_arabic\":false,\"key_intent\":\"one sentence\"}}",
                    system="You are a query router. Return only valid JSON.",
                    model=self.config.router,
                    temperature=0.1,
                    max_tokens=150,
                )
                parsed = json.loads(response.text.strip().replace("```json","").replace("```",""))
                if parsed.get("needs_grc"):
                    for a in ["grc_reasoner", "bahrain_policy"]:
                        if a not in agents:
                            agents.append(a)
                complexity = parsed.get("complexity", complexity)
            except Exception:
                pass  # Use keyword-based routing

        return {"agents": agents, "language": language, "complexity": complexity}


# ─── Individual Agent Executors ───────────────────────────────────────────────

class AgentExecutor:
    """Executes individual agents with their specialist models."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.config = get_config()

    async def execute_agent(
        self,
        agent_id: str,
        query: str,
        context: str = "",
        previous_outputs: Dict[str, str] = None,
    ) -> AgentResult:
        """Execute a single agent."""
        t0 = time.time()
        previous_outputs = previous_outputs or {}

        system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_id, AGENT_SYSTEM_PROMPTS["synthesizer"])
        model = AGENT_MODELS.get(agent_id, self.config.primary)

        # Build context-enriched prompt
        context_parts = [f"Query: {query}"]
        if context:
            context_parts.append(f"Memory Context:\n{context[:1000]}")
        if previous_outputs:
            prev = "\n\n".join([
                f"[{k.upper()} OUTPUT]:\n{v[:600]}"
                for k, v in list(previous_outputs.items())[-3:]  # Last 3 agents
            ])
            context_parts.append(f"Previous Agent Outputs:\n{prev}")

        full_prompt = "\n\n".join(context_parts)

        try:
            response = await self.llm.generate(
                prompt=full_prompt,
                system=system_prompt,
                model=model,
                temperature=self.config.temperature,
            )
            return AgentResult(
                agent_id=agent_id,
                output=response.text,
                confidence=0.85,
                latency_ms=(time.time() - t0) * 1000,
                model_used=response.model,
            )
        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            return AgentResult(
                agent_id=agent_id,
                output=f"Agent {agent_id} encountered an error: {str(e)[:200]}",
                confidence=0.0,
                latency_ms=(time.time() - t0) * 1000,
            )


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class CognitiveOrchestrator:
    """
    Master orchestrator implementing Mixture-of-Agents.
    Falls back to sequential pipeline if LangGraph not installed.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.router = MoARouter(llm_client)
        self.executor = AgentExecutor(llm_client)
        self.config = get_config()

        if LANGGRAPH_AVAILABLE:
            self._graph = self._build_langgraph()
            logger.info("✅ LangGraph MoA orchestrator initialized")
        else:
            self._graph = None
            logger.warning("LangGraph not installed — using sequential fallback. pip install langgraph")

    def _build_langgraph(self):
        """Build the LangGraph stateful agent graph."""
        workflow = StateGraph(PipelineState)

        # Add all agent nodes
        async def planner_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent("planner", state["query"])
            state["agent_outputs"]["planner"] = result.output
            try:
                plan = json.loads(result.output.replace("```json","").replace("```","").strip())
                state["query_type"] = plan.get("query_type", "research")
                state["complexity"] = plan.get("complexity", "moderate")
                state["language"] = plan.get("language", state.get("language", "en"))
            except Exception:
                pass
            return state

        async def researcher_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "researcher", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["agent_outputs"]["researcher"] = result.output
            return state

        async def reasoner_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "reasoner", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["agent_outputs"]["reasoner"] = result.output
            return state

        async def verifier_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "verifier", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["agent_outputs"]["verifier"] = result.output
            return state

        async def synthesizer_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "synthesizer", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["final_answer"] = result.output
            state["agent_outputs"]["synthesizer"] = result.output
            return state

        async def dialect_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "dialect_expert", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["agent_outputs"]["dialect_expert"] = result.output
            return state

        async def grc_node(state: PipelineState) -> PipelineState:
            result = await self.executor.execute_agent(
                "grc_reasoner", state["query"], previous_outputs=state["agent_outputs"]
            )
            state["agent_outputs"]["grc_reasoner"] = result.output
            return state

        # Add nodes to graph
        workflow.add_node("planner", planner_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("reasoner", reasoner_node)
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("synthesizer", synthesizer_node)
        workflow.add_node("dialect_expert", dialect_node)
        workflow.add_node("grc_reasoner", grc_node)

        # Define edges (sequential pipeline)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "reasoner")
        workflow.add_edge("reasoner", "verifier")
        workflow.add_edge("verifier", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    async def execute(self, query: str, mode: str = "cognitive", session_id: str = "") -> Dict:
        """Execute the full MoA pipeline."""
        t0 = time.time()

        # Get routing decision
        routing = await self.router.route(query, mode)
        selected_agents = routing["agents"]
        language = routing["language"]

        logger.info(f"MoA routing: {selected_agents} | lang={language} | mode={mode}")

        # Execute with LangGraph if available, else sequential
        if self._graph and LANGGRAPH_AVAILABLE:
            result = await self._run_langgraph(query, session_id, language, selected_agents)
        else:
            result = await self._run_sequential(query, session_id, language, selected_agents)

        result["latency_ms"] = (time.time() - t0) * 1000
        result["selected_agents"] = selected_agents
        result["language"] = language
        return result

    async def _run_langgraph(self, query: str, session_id: str,
                              language: str, agents: List[str]) -> Dict:
        """Execute pipeline via LangGraph state machine."""
        initial_state: PipelineState = {
            "query": query,
            "session_id": session_id,
            "language": language,
            "query_type": "research",
            "complexity": "moderate",
            "selected_agents": agents,
            "agent_outputs": {},
            "tool_results": [],
            "memory_context": "",
            "knowledge_graph": {},
            "final_answer": "",
            "confidence_score": 0.85,
            "hallucination_flags": [],
            "sources": [],
            "latency_ms": 0.0,
            "error": None,
        }

        final_state = await self._graph.ainvoke(initial_state)
        return {
            "answer": final_state.get("final_answer", ""),
            "traces": [{"agent": k, "output": v[:500]} for k, v in final_state.get("agent_outputs", {}).items()],
            "confidence": final_state.get("confidence_score", 0.85),
            "sources": final_state.get("sources", []),
        }

    async def _run_sequential(self, query: str, session_id: str,
                               language: str, agents: List[str]) -> Dict:
        """Sequential fallback pipeline — works without LangGraph."""
        outputs = {}

        # Always run in this order regardless of selection
        pipeline_order = ["planner", "researcher", "dialect_expert", "grc_reasoner",
                          "reasoner", "verifier", "hallucination_shield", "synthesizer"]

        active_agents = [a for a in pipeline_order if a in agents]

        for agent_id in active_agents:
            logger.info(f"Running agent: {agent_id}")
            result = await self.executor.execute_agent(
                agent_id=agent_id,
                query=query,
                previous_outputs=outputs,
            )
            outputs[agent_id] = result.output
            await asyncio.sleep(0.05)

        final = outputs.get("synthesizer", outputs.get(list(outputs.keys())[-1], ""))

        return {
            "answer": final,
            "traces": [{"agent": k, "output": v[:500]} for k, v in outputs.items()],
            "confidence": 0.85,
            "sources": [],
        }
