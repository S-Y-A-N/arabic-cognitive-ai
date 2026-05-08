class MemoryStore:
    def __init__(self, db: Path = MEMORY_DB):
        self.db = str(db)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    query    TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality  INTEGER DEFAULT 3,
                    tags     TEXT DEFAULT '[]',
                    created  TEXT DEFAULT (datetime('now'))
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
                    query, response, tags,
                    content='conversations', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS conv_fts_ai
                    AFTER INSERT ON conversations BEGIN
                    INSERT INTO conv_fts(rowid, query, response, tags)
                    VALUES (new.id, new.query, new.response, new.tags);
                END;
                CREATE TABLE IF NOT EXISTS experiment_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    query    TEXT,
                    mode     TEXT,
                    pipeline TEXT,
                    latency  INTEGER,
                    created  TEXT DEFAULT (datetime('now'))
                );
            """)

    def get_context(self, query: str, limit: int = 3) -> str:
        """FTS5 search — returns formatted context string for prompt injection."""
        try:
            q_esc = '"' + query.replace('"', '""') + '"'
            with sqlite3.connect(self.db) as c:
                rows = c.execute("""
                    SELECT c.query, c.response
                    FROM conv_fts f JOIN conversations c ON f.rowid = c.id
                    WHERE conv_fts MATCH ? AND c.quality >= 3
                    ORDER BY rank LIMIT ?
                """, (q_esc, limit)).fetchall()
            if not rows: return ""
            lines = ["[ذاكرة ذات صلة من محادثات سابقة]"]
            for q, r in rows:
                lines.append(f"• سؤال: {q[:80]}")
                lines.append(f"  جواب: {r[:160]}")
            return "\n".join(lines)
        except Exception as e:
            log.debug(f"Memory search failed: {e}")
            return ""

    def save(self, agent_id: str, query: str, response: str,
             quality: int = 3, tags: list = None):
        try:
            with sqlite3.connect(self.db) as c:
                c.execute(
                    "INSERT INTO conversations(agent_id,query,response,quality,tags) VALUES(?,?,?,?,?)",
                    (agent_id, query[:1000], response[:3000], quality,
                     json.dumps(tags or [], ensure_ascii=False))
                )
        except Exception as e:
            log.error(f"Memory save error: {e}")

    def log_experiment(self, query: str, mode: str, pipeline: list, latency: int):
        try:
            with sqlite3.connect(self.db) as c:
                c.execute(
                    "INSERT INTO experiment_log(query,mode,pipeline,latency) VALUES(?,?,?,?)",
                    (query[:300], mode, json.dumps(pipeline), latency)
                )
        except Exception as e:
            log.error(f"Experiment log error: {e}")

    def experiment_summary(self) -> dict:
        try:
            with sqlite3.connect(self.db) as c:
                rows = c.execute(
                    "SELECT mode, COUNT(*), AVG(latency) FROM experiment_log GROUP BY mode"
                ).fetchall()
            return {r[0]: {"count": r[1], "avg_latency_ms": int(r[2] or 0)} for r in rows}
        except: return {}

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self.db) as c:
                n_conv  = c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
                by_ag   = c.execute(
                    "SELECT agent_id, COUNT(*) FROM conversations GROUP BY agent_id"
                ).fetchall()
            return {"total": n_conv, "by_agent": {r[0]: r[1] for r in by_ag}}
        except: return {"total": 0, "by_agent": {}}


memory = MemoryStore()