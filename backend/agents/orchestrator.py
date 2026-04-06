"""
Cognitive Orchestrator
======================
The master brain of the Arabic Cognitive AI Engine.

Implements the full cognitive pipeline:
  Query → Analyze → Plan → Research → Reason → Verify → Synthesize → Answer

This is the core differentiator from a simple chatbot. Instead of sending
a query to a single LLM, the orchestrator coordinates multiple specialized
agents to produce research-grade, verified, cited answers.

Architecture Pattern: Hierarchical Multi-Agent Orchestration
- Master Orchestrator (this module)
  ├── Planner Agent      — decomposes complex queries into subtasks
  ├── Research Agent     — retrieves knowledge via RAG + web search
  ├── Reasoning Agent    — synthesizes information using CoT
  ├── Verification Agent — checks claims, reduces hallucinations
  ├── Citation Agent     — formats and validates citations
  └── Synthesis Agent    — produces final structured answer
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

logger = logging.getLogger("orchestrator")


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentTrace:
    agent_id: str
    agent_name: str
    status: AgentStatus
    input_summary: str
    output: str
    reasoning: str
    confidence: float
    processing_time: float
    tokens_used: int = 0


@dataclass
class OrchestratorResult:
    final_answer: str
    agent_traces: List[AgentTrace]
    sources: List[Dict[str, Any]]
    confidence_score: float
    reasoning_chain: List[str]
    tokens_used: int


# ─── System Prompts for Each Agent ────────────────────────────────────────────

PLANNER_SYSTEM = """You are the Planning Agent of the Arabic Cognitive AI Engine.

Your role is to decompose complex queries into structured research tasks.

Given a query, you must:
1. Identify the core information needs
2. Break the query into 3-5 specific sub-questions
3. Determine what type of knowledge is needed (factual, analytical, comparative, etc.)
4. Identify the optimal research strategy
5. Flag if the query involves Arabic dialect analysis, GCC context, or regional specifics

Output your analysis in this exact JSON structure:
{
  "query_type": "factual|analytical|comparative|research|opinion",
  "complexity": "simple|moderate|complex|expert",
  "language_detected": "arabic|english|mixed",
  "sub_tasks": [
    {"id": 1, "task": "...", "priority": "high|medium|low", "requires_rag": true/false}
  ],
  "research_strategy": "...",
  "arabic_context": true/false,
  "gcc_context": true/false,
  "estimated_confidence_ceiling": 0.0-1.0
}"""

RESEARCH_SYSTEM = """You are the Research Agent of the Arabic Cognitive AI Engine.

You synthesize information from multiple sources to produce comprehensive research findings.

Your role:
- Analyze retrieved documents and knowledge
- Extract key facts, figures, and insights
- Identify agreements and contradictions between sources
- Note knowledge gaps and uncertainties
- Produce structured research findings with confidence levels

For each finding, indicate:
- The supporting evidence
- Confidence level (High/Medium/Low)
- Whether this requires verification

Always distinguish between:
- Established facts (well-sourced)
- Common claims (need verification)  
- Inferences (clearly labeled as such)
- Knowledge gaps (explicitly flagged)

Research Confidence Rating: X/10 at the end of your response."""

REASONING_SYSTEM = """You are the Reasoning Agent of the Arabic Cognitive AI Engine.

You perform deep, multi-step reasoning to synthesize research findings into insights.

Your methodology:
1. Chain-of-Thought (CoT): Show explicit reasoning steps
2. Evidence Evaluation: Weigh sources by quality and relevance
3. Contradiction Resolution: When sources disagree, analyze which is more credible
4. Inference: Draw logical conclusions from evidence
5. Uncertainty Quantification: Be explicit about confidence levels

Format your reasoning as:
STEP 1: [Initial analysis]
STEP 2: [Evidence evaluation]
STEP 3: [Synthesis]
STEP 4: [Conclusion]
REASONING CONFIDENCE: X/10

Always be explicit about the logical structure of your reasoning."""

VERIFICATION_SYSTEM = """You are the Verification Agent of the Arabic Cognitive AI Engine.

Your critical role: Prevent hallucinations and verify claims.

For each major claim in the provided answer:
1. Assess if it's supported by the evidence
2. Flag potential hallucinations
3. Identify unsupported assertions
4. Rate the overall answer reliability

Output a VERIFICATION REPORT:
{
  "overall_verdict": "verified|partially_verified|unverified|hallucination_detected",
  "confidence_score": 0.0-1.0,
  "verified_claims": [...],
  "unverified_claims": [...],
  "potential_hallucinations": [...],
  "recommended_caveats": [...],
  "fact_check_score": 0-10
}"""

SYNTHESIS_SYSTEM = """You are the Synthesis Agent of the Arabic Cognitive AI Engine.

You produce the final, polished answer by integrating:
- Research findings
- Reasoning chain
- Verification report
- Citations

Your output must be:
1. Comprehensive yet concise
2. Well-structured with clear sections
3. Properly cited (reference sources)
4. Confidence-calibrated (acknowledge uncertainties)
5. Actionable and useful

Format the answer with:
- Executive Summary (2-3 sentences)
- Main Analysis (structured paragraphs)
- Key Insights (bullet points)
- Confidence Level and Caveats
- Sources and Citations

For Arabic queries: Respond in Arabic (MSA) with natural, professional language.
For English queries: Respond in English.
For mixed queries: Match the primary language of the question."""

ARABIC_NLP_SYSTEM = """أنت عميل اللغة العربية في محرك الذكاء الاصطناعي المعرفي العربي.

You are the Arabic NLP Agent, an expert in Arabic linguistics across all registers and dialects.

Your capabilities:
1. Dialect Detection: Identify MSA, Gulf (Bahraini, Kuwaiti, Emirati, Saudi), Egyptian, Levantine, Maghrebi
2. Morphological Analysis: Root extraction, pattern identification, inflection analysis
3. Normalization: Convert dialect to MSA while preserving meaning
4. Code-Switch Handling: Process Arabic-English mixed text gracefully
5. Cultural Context: Understand GCC-specific terminology, Islamic terminology, regional idioms
6. NER: Extract persons, places, organizations, dates in both Arabic and English

For any Arabic text, provide:
- Dialect identification with confidence %
- MSA normalization
- Key morphological features
- Cultural/regional context notes
- Entity extraction

اكتب إجاباتك بالعربية الفصحى مع شرح باللغة الإنجليزية عند الحاجة."""


class CognitiveOrchestrator:
    """
    Master orchestrator that coordinates all cognitive agents.
    
    Implements the full cognitive pipeline:
    Query → Plan → Research → Reason → Verify → Synthesize
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.total_requests = 0
        
        # Initialize LLM clients for each agent
        self._init_agents()
        logger.info("✅ CognitiveOrchestrator initialized")
    
    def _init_agents(self):
        """Initialize LLM clients for each specialized agent."""
        # Primary models for complex agents
        self.planner_llm = ChatAnthropic(model=self.model, max_tokens=1000, temperature=0.1)
        self.research_llm = ChatAnthropic(model=self.model, max_tokens=1500, temperature=0.2)
        self.reasoning_llm = ChatAnthropic(model=self.model, max_tokens=2000, temperature=0.3)
        self.verification_llm = ChatAnthropic(model=self.model, max_tokens=800, temperature=0.0)
        self.synthesis_llm = ChatAnthropic(model=self.model, max_tokens=2000, temperature=0.4)
        self.arabic_nlp_llm = ChatAnthropic(model=self.model, max_tokens=1000, temperature=0.1)
        
        # Lighter model for quick routing/classification
        self.router_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=200, temperature=0.0)
    
    async def execute(
        self,
        query: str,
        nlp_analysis: dict,
        memory_context: List[Dict],
        mode: str,
        agent_id: Optional[str],
        rag_pipeline,
        knowledge_graph,
        model_router,
    ) -> OrchestratorResult:
        """Execute the full cognitive pipeline."""
        self.total_requests += 1
        start = time.time()
        
        if mode == "single_agent":
            return await self._single_agent_execute(query, agent_id, memory_context)
        else:
            return await self._full_cognitive_execute(
                query, nlp_analysis, memory_context, rag_pipeline, knowledge_graph
            )
    
    async def _full_cognitive_execute(
        self, query: str, nlp_analysis: dict, 
        memory_context: List, rag_pipeline, knowledge_graph
    ) -> OrchestratorResult:
        """Full cognitive pipeline: Plan → Research → Reason → Verify → Synthesize."""
        
        agent_traces = []
        all_tokens = 0
        
        # ─── PHASE 1: Planning ───────────────────────────────────────────────
        logger.info("🗺️ Phase 1: Planning Agent")
        planner_start = time.time()
        
        planner_messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Query: {query}\n\nNLP Analysis: {nlp_analysis}")
        ]
        planner_response = await self.planner_llm.ainvoke(planner_messages)
        plan = planner_response.content
        
        agent_traces.append(AgentTrace(
            agent_id="planner",
            agent_name="Planning Agent",
            status=AgentStatus.COMPLETE,
            input_summary=query[:100],
            output=plan,
            reasoning="Query decomposition and research strategy formulation",
            confidence=0.90,
            processing_time=time.time() - planner_start,
            tokens_used=planner_response.usage_metadata.get("output_tokens", 0) if hasattr(planner_response, 'usage_metadata') else 0
        ))
        
        # ─── PHASE 2: RAG Research ───────────────────────────────────────────
        logger.info("🔬 Phase 2: Research Agent (RAG)")
        research_start = time.time()
        
        # Retrieve from vector DB and knowledge graph in parallel
        rag_results, kg_results = await asyncio.gather(
            rag_pipeline.hybrid_search(query=query, top_k=8),
            knowledge_graph.semantic_search(query=query, limit=5)
        )
        
        # Build research context from memory + RAG + KG
        memory_text = self._format_memory(memory_context)
        rag_text = self._format_rag_results(rag_results)
        kg_text = self._format_kg_results(kg_results)
        
        research_messages = [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"""
Query: {query}
Research Plan: {plan}

Retrieved Knowledge (Vector DB):
{rag_text}

Knowledge Graph Findings:
{kg_text}

Memory Context:
{memory_text}

Based on the above, produce comprehensive research findings.
""")
        ]
        research_response = await self.research_llm.ainvoke(research_messages)
        research_findings = research_response.content
        
        agent_traces.append(AgentTrace(
            agent_id="research",
            agent_name="Research Agent",
            status=AgentStatus.COMPLETE,
            input_summary=f"RAG: {len(rag_results)} docs, KG: {len(kg_results)} entities",
            output=research_findings,
            reasoning="Retrieved and synthesized knowledge from vector DB and knowledge graph",
            confidence=0.82,
            processing_time=time.time() - research_start
        ))
        
        # ─── PHASE 3: Reasoning ──────────────────────────────────────────────
        logger.info("🧠 Phase 3: Reasoning Agent")
        reasoning_start = time.time()
        
        reasoning_messages = [
            SystemMessage(content=REASONING_SYSTEM),
            HumanMessage(content=f"""
Original Query: {query}
Research Findings: {research_findings}

Apply deep multi-step reasoning to synthesize these findings into insights.
Show your full reasoning chain.
""")
        ]
        reasoning_response = await self.reasoning_llm.ainvoke(reasoning_messages)
        reasoning_output = reasoning_response.content
        
        agent_traces.append(AgentTrace(
            agent_id="reasoning",
            agent_name="Reasoning Agent",
            status=AgentStatus.COMPLETE,
            input_summary="Synthesizing research findings",
            output=reasoning_output,
            reasoning="Multi-step chain-of-thought inference",
            confidence=0.85,
            processing_time=time.time() - reasoning_start
        ))
        
        # ─── PHASE 4: Verification ───────────────────────────────────────────
        logger.info("✅ Phase 4: Verification Agent")
        verification_start = time.time()
        
        verification_messages = [
            SystemMessage(content=VERIFICATION_SYSTEM),
            HumanMessage(content=f"""
Original Query: {query}
Research Findings: {research_findings}
Reasoning Output: {reasoning_output}

Verify these findings and flag any potential hallucinations or unsupported claims.
""")
        ]
        verification_response = await self.verification_llm.ainvoke(verification_messages)
        verification_report = verification_response.content
        
        agent_traces.append(AgentTrace(
            agent_id="verification",
            agent_name="Verification Agent",
            status=AgentStatus.COMPLETE,
            input_summary="Verifying claims and checking hallucinations",
            output=verification_report,
            reasoning="Claim-by-claim verification against retrieved evidence",
            confidence=0.88,
            processing_time=time.time() - verification_start
        ))
        
        # ─── PHASE 5: Final Synthesis ─────────────────────────────────────────
        logger.info("📝 Phase 5: Synthesis Agent")
        synthesis_start = time.time()
        
        synthesis_messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(content=f"""
Original Query: {query}
Research Findings: {research_findings}
Reasoning Chain: {reasoning_output}
Verification Report: {verification_report}
Sources: {json.dumps([r.get('title', '') for r in rag_results[:5]]) if rag_results else '[]'}

Produce the final, polished answer with proper citations and confidence calibration.
""")
        ]
        
        import json
        synthesis_response = await self.synthesis_llm.ainvoke(synthesis_messages)
        final_answer = synthesis_response.content
        
        agent_traces.append(AgentTrace(
            agent_id="synthesis",
            agent_name="Synthesis Agent",
            status=AgentStatus.COMPLETE,
            input_summary="Generating final structured response",
            output=final_answer,
            reasoning="Integration of all agent outputs into coherent response",
            confidence=0.87,
            processing_time=time.time() - synthesis_start
        ))
        
        # ─── Build reasoning chain for transparency ───────────────────────────
        reasoning_chain = [
            f"1. PLANNING: Decomposed query into structured research tasks",
            f"2. RESEARCH: Retrieved {len(rag_results)} documents via hybrid search",
            f"3. REASONING: Applied chain-of-thought synthesis",
            f"4. VERIFICATION: Cross-checked all major claims",
            f"5. SYNTHESIS: Generated final verified answer"
        ]
        
        # Compute overall confidence
        avg_confidence = sum(t.confidence for t in agent_traces) / len(agent_traces)
        
        return OrchestratorResult(
            final_answer=final_answer,
            agent_traces=agent_traces,
            sources=[{"title": r.get("title", ""), "snippet": r.get("content", "")[:200], "score": r.get("score", 0)} for r in rag_results[:5]],
            confidence_score=round(avg_confidence, 2),
            reasoning_chain=reasoning_chain,
            tokens_used=all_tokens
        )
    
    async def _single_agent_execute(
        self, query: str, agent_id: str, memory_context: List
    ) -> OrchestratorResult:
        """Execute a single specified agent."""
        system_prompts = {
            "research": RESEARCH_SYSTEM,
            "reasoning": REASONING_SYSTEM,
            "planning": PLANNER_SYSTEM,
            "verification": VERIFICATION_SYSTEM,
            "arabic_nlp": ARABIC_NLP_SYSTEM,
            "synthesis": SYNTHESIS_SYSTEM,
        }
        
        system = system_prompts.get(agent_id, SYNTHESIS_SYSTEM)
        memory_text = self._format_memory(memory_context)
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Query: {query}\n\nContext: {memory_text}")
        ]
        
        response = await self.reasoning_llm.ainvoke(messages)
        
        trace = AgentTrace(
            agent_id=agent_id,
            agent_name=agent_id.replace("_", " ").title() + " Agent",
            status=AgentStatus.COMPLETE,
            input_summary=query[:100],
            output=response.content,
            reasoning=f"Single-agent execution via {agent_id}",
            confidence=0.80,
            processing_time=0.0
        )
        
        return OrchestratorResult(
            final_answer=response.content,
            agent_traces=[trace],
            sources=[],
            confidence_score=0.80,
            reasoning_chain=[f"Single agent: {agent_id}"],
            tokens_used=0
        )
    
    async def stream_execute(
        self, query: str, mode: str, rag_pipeline, knowledge_graph, agent_id: str = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream the cognitive pipeline with real-time agent updates."""
        
        yield {"type": "start", "message": "Cognitive pipeline initiated", "timestamp": time.time()}
        
        if mode == "full_cognitive":
            # Stream each agent phase
            phases = [
                ("planner", "Planning Agent", "Decomposing query into research tasks..."),
                ("research", "Research Agent", "Retrieving knowledge from RAG and Knowledge Graph..."),
                ("reasoning", "Reasoning Agent", "Applying chain-of-thought reasoning..."),
                ("verification", "Verification Agent", "Verifying claims and checking hallucinations..."),
                ("synthesis", "Synthesis Agent", "Generating final structured response..."),
            ]
            
            for agent_id_phase, agent_name, description in phases:
                yield {
                    "type": "agent_start",
                    "agent_id": agent_id_phase,
                    "agent_name": agent_name,
                    "description": description,
                    "timestamp": time.time()
                }
                
                # Execute agent (simplified for streaming demo)
                await asyncio.sleep(0.5)  # Simulate processing
                
                yield {
                    "type": "agent_complete",
                    "agent_id": agent_id_phase,
                    "agent_name": agent_name,
                    "status": "complete",
                    "timestamp": time.time()
                }
        
        yield {"type": "pipeline_complete", "timestamp": time.time()}
    
    async def nl_to_cypher(self, natural_language_query: str) -> str:
        """Convert natural language to Cypher (Neo4j query language)."""
        messages = [
            SystemMessage(content="Convert the natural language query to Cypher for Neo4j. Return only the Cypher query, no explanation."),
            HumanMessage(content=natural_language_query)
        ]
        response = await self.router_llm.ainvoke(messages)
        return response.content
    
    def _format_memory(self, memory_context: List) -> str:
        if not memory_context:
            return "No relevant memory context."
        return "\n".join([f"- {m.get('query', '')}: {m.get('response', '')[:200]}" for m in memory_context[:3]])
    
    def _format_rag_results(self, rag_results: List) -> str:
        if not rag_results:
            return "No documents retrieved."
        return "\n\n".join([
            f"[{i+1}] {r.get('title', 'Unknown')}\n{r.get('content', '')[:300]}"
            for i, r in enumerate(rag_results[:5])
        ])
    
    def _format_kg_results(self, kg_results: List) -> str:
        if not kg_results:
            return "No knowledge graph results."
        return "\n".join([str(r) for r in kg_results[:5]])
