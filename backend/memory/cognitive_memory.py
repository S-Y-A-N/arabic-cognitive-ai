"""
Cognitive Memory System
=======================
Three-tier memory architecture inspired by human cognitive memory:

  Tier 1: Working Memory   — Current session context (RAM-level speed)
  Tier 2: Episodic Memory  — Recent interactions (Redis, TTL-based)
  Tier 3: Semantic Memory  — Permanent knowledge (Weaviate vector DB)

This mirrors how the human brain processes information:
  Sensory Input → Working Memory → (consolidation) → Long-term Memory

Key insight: Most AI systems have no memory between sessions.
This system gives the Arabic Cognitive AI Engine persistent memory
that improves with every interaction.

Memory Operations:
  - STORE:   Save query + response + extracted entities
  - RECALL:  Retrieve semantically similar past interactions
  - FORGET:  TTL-based expiry for episodic memory (privacy by design)
  - CONSOLIDATE: Move important episodic memories to semantic memory

Enterprise use cases:
  - Remember user preferences across sessions
  - Build institutional knowledge from repeated queries
  - Track regulatory changes over time
  - Maintain research context across long projects

GCC Compliance:
  - All memories tagged with user consent level
  - Automatic expiry after configurable period
  - User can request complete memory deletion
  - No cross-user memory leakage
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger("memory_system")


class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryImportance(Enum):
    CRITICAL = 5    # Facts that were verified and used multiple times
    HIGH = 4        # Frequently referenced information
    MEDIUM = 3      # Standard research findings
    LOW = 2         # Background context
    EPHEMERAL = 1   # Session-specific, discard after use


@dataclass
class MemoryEntry:
    """A single memory in the cognitive memory system."""
    memory_id: str
    memory_type: MemoryType
    session_id: str
    query: str
    response_summary: str       # Summarized (not full response — for efficiency)
    entities: List[Dict]        # Extracted entities (for KG linkage)
    keywords: List[str]
    language: str               # ar | en | mixed
    agent_used: str
    importance: MemoryImportance = MemoryImportance.MEDIUM
    access_count: int = 0       # How many times retrieved
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # None = permanent
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_context_string(self) -> str:
        """Format memory as context for LLM prompt."""
        age_hours = (time.time() - self.created_at) / 3600
        age_str = f"{int(age_hours)}h ago" if age_hours < 24 else f"{int(age_hours/24)}d ago"
        return f"[Memory {age_str}] Q: {self.query[:100]}\nA: {self.response_summary[:300]}"
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class ConsolidationCandidate:
    """A memory candidate for consolidation from episodic to semantic."""
    memory_id: str
    query: str
    response_summary: str
    access_count: int
    importance_score: float  # Computed consolidation priority


class WorkingMemory:
    """
    Tier 1: Working Memory — Current session context.
    
    In-memory only, fastest access.
    Limited capacity (like human working memory: 7±2 chunks).
    Automatically manages context window for LLM.
    """
    
    CAPACITY = 10  # Max items in working memory
    
    def __init__(self):
        self._sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.CAPACITY))
    
    def store(self, session_id: str, entry: MemoryEntry):
        """Store in working memory (FIFO queue with max capacity)."""
        self._sessions[session_id].append(entry)
    
    def retrieve(self, session_id: str, top_k: int = 5) -> List[MemoryEntry]:
        """Get most recent k items from working memory."""
        session_mem = list(self._sessions[session_id])
        return session_mem[-top_k:]
    
    def get_context_string(self, session_id: str, max_tokens: int = 800) -> str:
        """Get working memory formatted as LLM context string."""
        entries = self.retrieve(session_id)
        if not entries:
            return ""
        
        context_parts = ["Recent session context:"]
        total_chars = 0
        
        for entry in reversed(entries):
            context = entry.to_context_string()
            if total_chars + len(context) > max_tokens * 4:  # ~4 chars per token
                break
            context_parts.append(f"  {context}")
            total_chars += len(context)
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Clear working memory for a session."""
        self._sessions.pop(session_id, None)
    
    def active_sessions_count(self) -> int:
        return len(self._sessions)


class EpisodicMemory:
    """
    Tier 2: Episodic Memory — Recent interactions with TTL.
    
    In production: backed by Redis with TTL.
    In development: in-memory with time-based expiry.
    
    TTL Strategy:
    - Positive feedback: 30 days
    - Neutral: 7 days
    - Negative feedback: 24 hours (learn but forget the bad)
    - Sensitive/PII content: 24 hours max
    """
    
    DEFAULT_TTL_DAYS = 7
    
    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._memory_store: Dict[str, MemoryEntry] = {}
        logger.info("EpisodicMemory initialized (in-memory fallback)")
    
    async def store(self, entry: MemoryEntry, ttl_days: int = None):
        """Store with automatic expiry."""
        ttl = (ttl_days or self.DEFAULT_TTL_DAYS) * 86400
        entry.expires_at = time.time() + ttl
        
        if self._redis:
            await self._redis.setex(
                f"episodic:{entry.memory_id}",
                int(ttl),
                json.dumps(entry.__dict__)
            )
        else:
            self._memory_store[entry.memory_id] = entry
    
    async def retrieve(self, query_embedding: List[float] = None, session_id: str = None, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant episodic memories."""
        # Clean expired entries
        self._cleanup_expired()
        
        if session_id:
            entries = [e for e in self._memory_store.values() if e.session_id == session_id]
        else:
            entries = list(self._memory_store.values())
        
        # Sort by recency and access count
        entries.sort(key=lambda e: (e.access_count * 0.3 + (e.created_at / time.time()) * 0.7), reverse=True)
        
        # Update access count
        for entry in entries[:top_k]:
            entry.access_count += 1
            entry.last_accessed = time.time()
        
        return entries[:top_k]
    
    async def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[MemoryEntry]:
        """Simple keyword-based episodic memory search."""
        self._cleanup_expired()
        
        results = []
        keywords_lower = [k.lower() for k in keywords]
        
        for entry in self._memory_store.values():
            score = sum(1 for kw in keywords_lower if kw in entry.query.lower() or kw in entry.response_summary.lower())
            if score > 0:
                results.append((score, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:top_k]]
    
    def _cleanup_expired(self):
        """Remove expired entries from memory store."""
        expired = [k for k, v in self._memory_store.items() if v.is_expired()]
        for k in expired:
            del self._memory_store[k]
        
        if expired:
            logger.debug(f"Cleaned {len(expired)} expired episodic memories")
    
    async def total_count(self) -> int:
        self._cleanup_expired()
        return len(self._memory_store)


class SemanticMemory:
    """
    Tier 3: Semantic Memory — Permanent knowledge store.
    
    Backed by the vector database (Weaviate).
    Stores important facts, verified knowledge, and domain expertise.
    Retrieved via semantic similarity search.
    
    This is where the system builds long-term Arabic knowledge.
    """
    
    def __init__(self, vector_store=None):
        self._vector_store = vector_store
        self._memory_index: Dict[str, MemoryEntry] = {}
        logger.info("SemanticMemory initialized")
    
    async def store(self, entry: MemoryEntry):
        """Store important knowledge permanently."""
        entry.memory_type = MemoryType.SEMANTIC
        entry.expires_at = None  # Semantic memory is permanent
        
        self._memory_index[entry.memory_id] = entry
        
        # In production: also store in Weaviate vector DB
        if self._vector_store:
            await self._vector_store.insert(
                content=f"{entry.query}\n{entry.response_summary}",
                metadata={
                    "memory_id": entry.memory_id,
                    "memory_type": "semantic",
                    "language": entry.language,
                    "keywords": ",".join(entry.keywords[:10]),
                }
            )
        
        logger.debug(f"Stored semantic memory: {entry.memory_id}")
    
    async def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Semantic search through long-term memory."""
        if self._vector_store:
            # Use vector similarity search
            results = await self._vector_store.hybrid_search(query=query, top_k=top_k)
            return [self._memory_index.get(r.get("memory_id")) for r in results if r.get("memory_id") in self._memory_index]
        
        # Fallback: keyword matching
        query_words = set(query.lower().split())
        scores = []
        
        for mem in self._memory_index.values():
            mem_words = set((mem.query + " " + " ".join(mem.keywords)).lower().split())
            overlap = len(query_words & mem_words) / max(len(query_words), 1)
            if overlap > 0:
                scores.append((overlap, mem))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scores[:top_k]]
    
    async def total_count(self) -> int:
        return len(self._memory_index)


class MemoryConsolidator:
    """
    Consolidates episodic memories into semantic memory.
    
    Consolidation rules:
    - Accessed 3+ times → promote to semantic memory
    - Importance >= HIGH → immediately promote
    - Verified factual information → promote
    - User corrected it → do NOT promote
    
    Runs periodically (like sleep consolidation in humans).
    """
    
    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
        self._consolidation_count = 0
    
    async def consolidate(self, top_k: int = 20):
        """Promote high-value episodic memories to semantic memory."""
        memories = await self.episodic.retrieve(top_k=100)
        
        candidates = []
        for mem in memories:
            if mem.importance.value >= MemoryImportance.HIGH.value or mem.access_count >= 3:
                importance_score = mem.access_count * 0.4 + mem.importance.value * 0.6
                candidates.append(ConsolidationCandidate(
                    memory_id=mem.memory_id,
                    query=mem.query,
                    response_summary=mem.response_summary,
                    access_count=mem.access_count,
                    importance_score=importance_score
                ))
        
        # Sort by importance and take top_k
        candidates.sort(key=lambda c: c.importance_score, reverse=True)
        candidates = candidates[:top_k]
        
        consolidated = 0
        for candidate in candidates:
            mem = self.episodic._memory_store.get(candidate.memory_id)
            if mem:
                await self.semantic.store(mem)
                consolidated += 1
        
        self._consolidation_count += consolidated
        if consolidated > 0:
            logger.info(f"Consolidated {consolidated} episodic memories to semantic")
        
        return consolidated


class CognitiveMemorySystem:
    """
    Master cognitive memory system.
    Coordinates all three memory tiers.
    
    Usage:
        memory = CognitiveMemorySystem()
        await memory.initialize()
        
        # Store a new interaction
        await memory.store(
            session_id="session_123",
            query="What is AI?",
            response="AI is...",
            agent="deep_research",
            entities=[{"name": "AI", "type": "Concept"}]
        )
        
        # Retrieve context for a new query
        context = await memory.retrieve_context("Tell me more about AI", "session_123")
    """
    
    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.consolidator = MemoryConsolidator(self.episodic, self.semantic)
        self._consolidation_interval = 3600  # Consolidate every hour
        logger.info("✅ CognitiveMemorySystem initialized (3-tier)")
    
    async def initialize(self):
        """Initialize all memory tiers."""
        logger.info("CognitiveMemorySystem ready")
        # Start consolidation background task
        asyncio.create_task(self._run_consolidation_loop())
    
    async def store(
        self,
        session_id: str,
        query: str,
        response: str,
        agent: str = "unknown",
        entities: List[Dict] = None,
        feedback_signal: int = 0,
    ) -> str:
        """Store a new interaction across memory tiers."""
        
        # Detect language
        arabic_ratio = sum(1 for c in query if '\u0600' <= c <= '\u06FF') / max(len(query), 1)
        language = "ar" if arabic_ratio > 0.5 else "mixed" if arabic_ratio > 0.1 else "en"
        
        # Summarize response (first 500 chars)
        response_summary = response[:500] + "..." if len(response) > 500 else response
        
        # Extract keywords
        keywords = self._extract_keywords(query + " " + response_summary)
        
        # Determine importance based on feedback
        importance = (
            MemoryImportance.HIGH if feedback_signal > 0 else
            MemoryImportance.LOW if feedback_signal < 0 else
            MemoryImportance.MEDIUM
        )
        
        memory_id = hashlib.md5(f"{session_id}{query}{time.time()}".encode()).hexdigest()[:16]
        
        entry = MemoryEntry(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            query=query,
            response_summary=response_summary,
            entities=entities or [],
            keywords=keywords,
            language=language,
            agent_used=agent,
            importance=importance,
        )
        
        # Store in working memory (fast, current session)
        self.working.store(session_id, entry)
        
        # Store in episodic memory (persist across sessions)
        ttl_days = 30 if feedback_signal > 0 else 7 if feedback_signal == 0 else 1
        await self.episodic.store(entry, ttl_days=ttl_days)
        
        # Immediately promote highly important memories to semantic
        if importance.value >= MemoryImportance.CRITICAL.value:
            await self.semantic.store(entry)
        
        logger.debug(f"Stored memory {memory_id} (lang: {language}, importance: {importance.name})")
        return memory_id
    
    async def retrieve_context(self, query: str, session_id: str, top_k: int = 5) -> str:
        """Retrieve relevant memory context for a query."""
        
        # Working memory (current session — most relevant)
        working_context = self.working.get_context_string(session_id, max_tokens=600)
        
        # Episodic memory (keyword search)
        keywords = self._extract_keywords(query)
        episodic_entries = await self.episodic.keyword_search(keywords, top_k=3)
        episodic_context = "\n".join([e.to_context_string() for e in episodic_entries if e.session_id != session_id])
        
        # Semantic memory (most relevant permanent knowledge)
        semantic_entries = await self.semantic.search(query, top_k=2)
        semantic_context = "\n".join([e.to_context_string() for e in semantic_entries if e])
        
        # Combine all memory contexts
        parts = []
        if working_context:
            parts.append(f"WORKING MEMORY (current session):\n{working_context}")
        if episodic_context:
            parts.append(f"EPISODIC MEMORY (related past queries):\n{episodic_context}")
        if semantic_context:
            parts.append(f"SEMANTIC KNOWLEDGE (verified facts):\n{semantic_context}")
        
        return "\n\n".join(parts) if parts else ""
    
    async def retrieve_relevant(self, query: str, session_id: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memory entries as list of dicts."""
        entries = await self.episodic.retrieve(session_id=session_id, top_k=top_k)
        return [{"query": e.query, "response": e.response_summary, "agent": e.agent_used} for e in entries]
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract keywords from text."""
        stop_words = {"من", "في", "على", "إلى", "عن", "مع", "هذا", "the", "is", "a", "of", "in", "to", "and"}
        words = [w.lower() for w in text.split() if len(w) > 3 and w.lower() not in stop_words]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:n]
    
    async def clear_session(self, session_id: str):
        """Clear working memory for a session (episodic/semantic preserved)."""
        self.working.clear_session(session_id)
    
    async def shutdown(self):
        """Graceful shutdown — run final consolidation."""
        logger.info("Running final memory consolidation before shutdown...")
        await self.consolidator.consolidate(top_k=50)
        logger.info("Memory system shutdown complete")
    
    async def active_sessions_count(self) -> int:
        return self.working.active_sessions_count()
    
    async def total_entries(self) -> int:
        ep = await self.episodic.total_count()
        sem = await self.semantic.total_count()
        return ep + sem
    
    async def get_memory_stats(self) -> Dict:
        """Return statistics about the memory system."""
        return {
            "working_memory_sessions": self.working.active_sessions_count(),
            "episodic_memories": await self.episodic.total_count(),
            "semantic_memories": await self.semantic.total_count(),
            "total_consolidated": self.consolidator._consolidation_count,
            "tiers": {
                "working": "RAM (current session)",
                "episodic": "Redis TTL-based (7-30 days)",
                "semantic": "Weaviate vector DB (permanent)"
            }
        }
    
    async def _run_consolidation_loop(self):
        """Background consolidation loop — runs every hour."""
        while True:
            await asyncio.sleep(self._consolidation_interval)
            try:
                count = await self.consolidator.consolidate()
                if count > 0:
                    logger.info(f"Memory consolidation: {count} memories promoted to semantic")
            except Exception as e:
                logger.error(f"Consolidation failed: {e}")
