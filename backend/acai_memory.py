"""
ACAI — Persistent Memory System (Hermes-inspired)
===================================================
Implements cross-session memory using SQLite FTS5.
Agents remember past conversations, learned facts, and auto-generated skills.

Architecture inspired by:
- Hermes Agent (Nous Research, 2026) — FTS5 + LLM summarization
- Rowboat — Obsidian-style knowledge vault
- 3-tier: Working → Episodic → Semantic

Usage:
    from acai_memory import ACAIMemory
    mem = ACAIMemory()
    mem.save_conversation("lughawi", "شلونك؟", "الحمد لله بخير", tags=["bahraini","greeting"])
    results = mem.search("شلونك")
    skills = mem.get_skills("dialect_analysis")

Install: pip install sqlite-fts4  (FTS5 is built into Python's sqlite3)
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ─── Database path (persistent across sessions) ───────────────────────────────
DB_PATH = Path("acai_memory.db")
SKILLS_DIR = Path("acai_skills")
SKILLS_DIR.mkdir(exist_ok=True)

class ACAIMemory:
    """
    Persistent memory system for ACAI.
    Survives server restarts. Every agent can read/write.
    
    Three tiers:
    1. Working memory     — current session (in RAM)
    2. Episodic memory    — past conversations (SQLite FTS5)  
    3. Semantic memory    — learned facts + skills (SQLite + Markdown files)
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._working: Dict[str, List] = {}  # session_id → messages
        self._init_db()
    
    def _init_db(self):
        """Create tables with FTS5 full-text search (like Hermes)."""
        with sqlite3.connect(self.db_path) as db:
            db.executescript("""
                -- Conversations: full-text searchable
                CREATE TABLE IF NOT EXISTS conversations (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id  TEXT NOT NULL,
                    query     TEXT NOT NULL,
                    response  TEXT NOT NULL,
                    tags      TEXT DEFAULT '[]',
                    quality   INTEGER DEFAULT 3,
                    created   TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                );
                
                -- FTS5 index over conversations (like Hermes FTS5)
                CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
                    query, response, tags,
                    content='conversations',
                    content_rowid='id'
                );
                
                -- Trigger to keep FTS5 in sync
                CREATE TRIGGER IF NOT EXISTS conv_ai AFTER INSERT ON conversations BEGIN
                    INSERT INTO conv_fts(rowid, query, response, tags)
                    VALUES (new.id, new.query, new.response, new.tags);
                END;
                
                -- Facts: long-term semantic knowledge
                CREATE TABLE IF NOT EXISTS facts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact      TEXT NOT NULL,
                    source    TEXT,
                    domain    TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    created   TEXT DEFAULT (datetime('now')),
                    expires   TEXT
                );
                
                CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                    fact, domain,
                    content='facts',
                    content_rowid='id'
                );
                
                CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                    INSERT INTO facts_fts(rowid, fact, domain)
                    VALUES (new.id, new.fact, new.domain);
                END;
                
                -- Skills: auto-generated procedural memory (agentskills.io format)
                CREATE TABLE IF NOT EXISTS skills (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    agent_id   TEXT,
                    trigger    TEXT,
                    steps      TEXT,        -- JSON array of steps
                    use_count  INTEGER DEFAULT 0,
                    rating     REAL DEFAULT 0.0,
                    created    TEXT DEFAULT (datetime('now')),
                    last_used  TEXT
                );
                
                -- User model: who is this user, what do they care about
                CREATE TABLE IF NOT EXISTS user_model (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    key     TEXT NOT NULL UNIQUE,
                    value   TEXT NOT NULL,
                    updated TEXT DEFAULT (datetime('now'))
                );
            """)
    
    # ─── Working Memory (session-scoped, in RAM) ──────────────────────────────
    
    def working_add(self, session_id: str, role: str, content: str):
        """Add message to working memory (current session)."""
        if session_id not in self._working:
            self._working[session_id] = []
        self._working[session_id].append({
            "role": role, "content": content,
            "time": datetime.now().isoformat()
        })
        # Keep last 20 messages only
        self._working[session_id] = self._working[session_id][-20:]
    
    def working_get(self, session_id: str, last_n: int = 10) -> List[Dict]:
        """Get recent messages from working memory."""
        return self._working.get(session_id, [])[-last_n:]
    
    def working_clear(self, session_id: str):
        """Clear working memory for a session."""
        self._working.pop(session_id, None)
    
    # ─── Episodic Memory (SQLite FTS5) ────────────────────────────────────────
    
    def save_conversation(
        self, agent_id: str, query: str, response: str,
        tags: List[str] = None, quality: int = 3, session_id: str = None
    ) -> int:
        """Save a conversation to episodic memory."""
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        with sqlite3.connect(self.db_path) as db:
            cur = db.execute(
                """INSERT INTO conversations (agent_id, query, response, tags, quality, session_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent_id, query, response, tags_json, quality, session_id)
            )
            return cur.lastrowid
    
    def search(self, query: str, limit: int = 5, agent_id: str = None) -> List[Dict]:
        """
        Full-text search across all past conversations.
        Uses SQLite FTS5 — same as Hermes Agent.
        """
        with sqlite3.connect(self.db_path) as db:
            if agent_id:
                rows = db.execute("""
                    SELECT c.id, c.agent_id, c.query, c.response, c.tags, c.created, c.quality
                    FROM conv_fts f
                    JOIN conversations c ON f.rowid = c.id
                    WHERE conv_fts MATCH ? AND c.agent_id = ?
                    ORDER BY rank LIMIT ?
                """, (self._fts_escape(query), agent_id, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT c.id, c.agent_id, c.query, c.response, c.tags, c.created, c.quality
                    FROM conv_fts f
                    JOIN conversations c ON f.rowid = c.id
                    WHERE conv_fts MATCH ?
                    ORDER BY rank LIMIT ?
                """, (self._fts_escape(query), limit)).fetchall()
        
        return [
            {"id":r[0],"agent":r[1],"query":r[2],"response":r[3],
             "tags":json.loads(r[4]),"created":r[5],"quality":r[6]}
            for r in rows
        ]
    
    def get_context(self, query: str, agent_id: str = None, limit: int = 3) -> str:
        """
        Get relevant past context for a new query.
        Returns formatted string to prepend to LLM prompt.
        """
        results = self.search(query, limit=limit, agent_id=agent_id)
        if not results:
            return ""
        
        lines = ["[ذاكرة ذات صلة من المحادثات السابقة]"]
        for r in results:
            lines.append(f"• سؤال سابق: {r['query'][:100]}")
            lines.append(f"  الإجابة: {r['response'][:200]}")
        return "\n".join(lines)
    
    def _fts_escape(self, query: str) -> str:
        """Escape special FTS5 characters."""
        # Simple approach: quote the query
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
    
    # ─── Semantic Memory / Facts ──────────────────────────────────────────────
    
    def save_fact(self, fact: str, source: str = None, domain: str = "general",
                  confidence: float = 1.0, expires_days: int = None) -> int:
        """Save a long-term fact to semantic memory."""
        expires = None
        if expires_days:
            expires = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as db:
            cur = db.execute(
                """INSERT INTO facts (fact, source, domain, confidence, expires)
                   VALUES (?, ?, ?, ?, ?)""",
                (fact, source, domain, confidence, expires)
            )
            return cur.lastrowid
    
    def search_facts(self, query: str, domain: str = None, limit: int = 5) -> List[Dict]:
        """Search semantic memory for relevant facts."""
        with sqlite3.connect(self.db_path) as db:
            if domain:
                rows = db.execute("""
                    SELECT f.id, f.fact, f.source, f.domain, f.confidence
                    FROM facts_fts ff JOIN facts f ON ff.rowid = f.id
                    WHERE facts_fts MATCH ? AND f.domain = ?
                    AND (f.expires IS NULL OR f.expires > datetime('now'))
                    ORDER BY rank LIMIT ?
                """, (self._fts_escape(query), domain, limit)).fetchall()
            else:
                rows = db.execute("""
                    SELECT f.id, f.fact, f.source, f.domain, f.confidence
                    FROM facts_fts ff JOIN facts f ON ff.rowid = f.id
                    WHERE facts_fts MATCH ?
                    AND (f.expires IS NULL OR f.expires > datetime('now'))
                    ORDER BY rank LIMIT ?
                """, (self._fts_escape(query), limit)).fetchall()
        
        return [{"id":r[0],"fact":r[1],"source":r[2],"domain":r[3],"confidence":r[4]}
                for r in rows]
    
    # ─── Skill Memory (agentskills.io compatible) ─────────────────────────────
    
    def create_skill(self, name: str, description: str, agent_id: str,
                     trigger: str, steps: List[str]) -> str:
        """
        Auto-generate a reusable skill from a solved problem.
        Saves as both SQLite record AND Markdown file (agentskills.io format).
        """
        steps_json = json.dumps(steps, ensure_ascii=False)
        
        # Save to DB
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT OR REPLACE INTO skills (name, description, agent_id, trigger, steps)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, agent_id, trigger, steps_json))
        
        # Save as Markdown file (agentskills.io format)
        skill_content = f"""---
name: {name}
agent: {agent_id}
trigger: {trigger}
created: {datetime.now().isoformat()}
---

# {name}

**Description:** {description}

**Trigger:** When the user asks about: {trigger}

## Steps

""" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        
        skill_file = SKILLS_DIR / f"{name.lower().replace(' ','_')}.md"
        skill_file.write_text(skill_content, encoding="utf-8")
        
        return str(skill_file)
    
    def get_skills(self, trigger_keyword: str = None, agent_id: str = None) -> List[Dict]:
        """Retrieve relevant skills."""
        with sqlite3.connect(self.db_path) as db:
            if trigger_keyword:
                rows = db.execute("""
                    SELECT name, description, agent_id, trigger, steps, use_count, rating
                    FROM skills
                    WHERE trigger LIKE ? OR description LIKE ? OR name LIKE ?
                    ORDER BY use_count DESC, rating DESC LIMIT 5
                """, (f"%{trigger_keyword}%",)*3).fetchall()
            elif agent_id:
                rows = db.execute("""
                    SELECT name, description, agent_id, trigger, steps, use_count, rating
                    FROM skills WHERE agent_id = ?
                    ORDER BY use_count DESC LIMIT 10
                """, (agent_id,)).fetchall()
            else:
                rows = db.execute("""
                    SELECT name, description, agent_id, trigger, steps, use_count, rating
                    FROM skills ORDER BY use_count DESC LIMIT 10
                """).fetchall()
        
        return [{"name":r[0],"description":r[1],"agent":r[2],"trigger":r[3],
                 "steps":json.loads(r[4]),"use_count":r[5],"rating":r[6]} for r in rows]
    
    def skill_used(self, name: str, rating: float = None):
        """Mark a skill as used and optionally rate it."""
        with sqlite3.connect(self.db_path) as db:
            if rating:
                db.execute("""
                    UPDATE skills SET use_count = use_count+1, last_used=datetime('now'),
                    rating = (rating * use_count + ?)/(use_count+1)
                    WHERE name = ?
                """, (rating, name))
            else:
                db.execute("""
                    UPDATE skills SET use_count=use_count+1, last_used=datetime('now')
                    WHERE name = ?
                """, (name,))
    
    # ─── User Model (Honcho-inspired) ─────────────────────────────────────────
    
    def update_user_model(self, key: str, value: str):
        """Update what we know about the user."""
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT INTO user_model (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=datetime('now')
            """, (key, value))
    
    def get_user_model(self) -> Dict:
        """Get full user model."""
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT key, value FROM user_model").fetchall()
        return {r[0]: r[1] for r in rows}
    
    # ─── Stats ────────────────────────────────────────────────────────────────
    
    def stats(self) -> Dict:
        """Memory system statistics."""
        with sqlite3.connect(self.db_path) as db:
            convs = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            facts = db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            skills = db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            agents = db.execute("SELECT agent_id, COUNT(*) FROM conversations GROUP BY agent_id").fetchall()
        return {
            "total_conversations": convs,
            "total_facts": facts,
            "total_skills": skills,
            "working_sessions": len(self._working),
            "by_agent": {r[0]: r[1] for r in agents}
        }


# ─── Skill Auto-Generator ─────────────────────────────────────────────────────

class SkillGenerator:
    """
    Analyzes conversations and auto-generates skills (like Hermes).
    When an agent successfully handles a complex task, extract it as a reusable skill.
    """
    
    def __init__(self, memory: ACAIMemory):
        self.memory = memory
    
    def should_create_skill(self, query: str, response: str, quality: int) -> bool:
        """Decide if this interaction warrants a skill."""
        # Create skill if: response is detailed + quality is high + response has steps
        has_steps = any(s in response for s in ["١.", "٢.", "1.", "2.", "أولاً", "أول شيء", "Step"])
        is_detailed = len(response) > 300
        is_good = quality >= 4
        return has_steps and is_detailed and is_good
    
    def extract_skill(self, agent_id: str, query: str, response: str) -> Optional[str]:
        """
        Auto-extract a skill from a successful interaction.
        Returns the skill file path if created, None otherwise.
        """
        # Simple extraction: use the query as trigger, response structure as steps
        trigger = query[:80]
        
        # Extract numbered steps from response
        import re
        steps = re.findall(r'[١٢٣٤٥٦٧٨٩1-9][.)] (.+)', response)
        if not steps:
            # Just use first 3 sentences
            sentences = [s.strip() for s in response.split('\n') if s.strip() and len(s) > 20]
            steps = sentences[:3]
        
        if not steps:
            return None
        
        # Generate skill name from query
        name_words = query.replace("?","").replace("؟","").strip().split()[:4]
        name = "_".join(name_words[:3]) if name_words else f"skill_{agent_id}"
        name = f"{agent_id}_{name}"[:50]
        
        # Create description
        description = f"يُجيب على: {query[:100]}"
        
        try:
            skill_path = self.memory.create_skill(
                name=name, description=description,
                agent_id=agent_id, trigger=trigger, steps=steps[:5]
            )
            return skill_path
        except Exception:
            return None


# ─── Singleton instance ───────────────────────────────────────────────────────
_memory_instance = None

def get_memory() -> ACAIMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ACAIMemory()
    return _memory_instance


# ─── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ACAI Persistent Memory...")
    mem = ACAIMemory(DB_PATH)
    
    # Test episodic memory
    mem.save_conversation("lughawi", "شلونك؟", "الحمد لله بخير",
                          tags=["bahraini","greeting"], quality=5)
    mem.save_conversation("musheer", "ما دور CBB؟", "مصرف البحرين المركزي يُنظّم القطاع المصرفي",
                          tags=["cbb","bahrain","banking"], quality=5)
    
    # Test search
    results = mem.search("بخير")
    print(f"  Search 'بخير': {len(results)} results")
    
    # Test skill creation
    skill_path = mem.create_skill(
        name="bahrain_banking_card_lost",
        description="Handles lost card queries in Bahraini dialect",
        agent_id="lughawi",
        trigger="بطاقتي ضاعت",
        steps=["أوقف البطاقة فوراً من التطبيق", "اتصل بالبنك", "اطلب بطاقة بديلة"]
    )
    print(f"  Skill created: {skill_path}")
    
    # Test facts
    mem.save_fact("مصرف البحرين المركزي أُسِّس عام 2006", source="CBB Law", domain="gcc_policy")
    facts = mem.search_facts("CBB")
    print(f"  Facts search 'CBB': {len(facts)} results")
    
    # Stats
    stats = mem.stats()
    print(f"  Stats: {stats}")
    print("\n✅ Memory system working!")
