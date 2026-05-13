# TODO Will be changed

class MinimalRAG:
    """
    Zero-dependency RAG using SQLite FTS5.
    No Weaviate, no sentence-transformers needed for demo.
    Ingest → chunk → store → retrieve → inject → cite.
    """

    def __init__(self, db: Path = RAG_DB):
        self.db = str(db)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_name TEXT NOT NULL,
                    chunk_no INTEGER,
                    content  TEXT NOT NULL,
                    created  TEXT DEFAULT (datetime('now'))
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, doc_name,
                    content='chunks', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai
                    AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content, doc_name)
                    VALUES (new.id, new.content, new.doc_name);
                END;
            """)

    def ingest(self, text: str, doc_name: str, chunk_size: int = 400) -> int:
        """Split document and store as searchable chunks."""
        # Sentence-aware chunking
        sents = re.split(r'(?<=[.!?،؟])\s+', text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur) + len(s) < chunk_size:
                cur += s + " "
            else:
                if cur.strip(): chunks.append(cur.strip())
                cur = s + " "
        if cur.strip(): chunks.append(cur.strip())
        if not chunks:   chunks = [text[:chunk_size]]

        with sqlite3.connect(self.db) as c:
            for i, ch in enumerate(chunks):
                c.execute(
                    "INSERT INTO chunks(doc_name, chunk_no, content) VALUES(?,?,?)",
                    (doc_name, i, ch)
                )
        log.info(f"RAG: ingested '{doc_name}' → {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list:
        """FTS5 search over chunks."""
        try:
            q_esc = '"' + query.replace('"', '""') + '"'
            with sqlite3.connect(self.db) as c:
                rows = c.execute("""
                    SELECT ch.doc_name, ch.chunk_no, ch.content
                    FROM chunks_fts cf JOIN chunks ch ON cf.rowid = ch.id
                    WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
                """, (q_esc, k)).fetchall()
            return [{"doc": r[0], "chunk": r[1], "content": r[2]} for r in rows]
        except Exception as e:
            log.debug(f"RAG retrieve error: {e}")
            return []

    def get_rag_context(self, query: str, k: int = 3) -> str:
        chunks = self.retrieve(query, k)
        if not chunks: return ""
        lines = ["[مقتطفات من الوثائق المرجعية]"]
        for c in chunks:
            lines.append(f"📄 {c['doc']} — القطعة {c['chunk']+1}")
            lines.append(f"   {c['content'][:300]}")
        return "\n".join(lines)

    def list_docs(self) -> list:
        try:
            with sqlite3.connect(self.db) as c:
                rows = c.execute(
                    "SELECT doc_name, COUNT(*) FROM chunks GROUP BY doc_name"
                ).fetchall()
            return [{"doc": r[0], "chunks": r[1]} for r in rows]
        except: return []


rag = MinimalRAG()

# ── Auto-ingest sample CBB document on first run ──────────────────────────────
CBB_SAMPLE = """مصرف البحرين المركزي — ملخص تنظيمي
الهدف: المحافظة على الاستقرار النقدي والمالي في مملكة البحرين.

الترخيص المصرفي:
رأس المال الأدنى للبنوك التجارية: 100 مليون دينار بحريني.
يشترط تقديم طلب مكتمل مع خطة عمل خمسية ونظام حوكمة معتمد.
يستغرق قرار الترخيص عادةً 6-12 شهراً.

حماية المستهلك (CBB Rulebook — المجلد الخامس):
يلتزم البنك بالإفصاح الكامل عن الرسوم والفوائد.
يجب توفير قناة شكاوى رسمية.
الرد على الشكاوى خلال 15 يوم عمل.

مكافحة غسل الأموال:
تطبيق إجراءات KYC (اعرف عميلك) إلزامي.
الإبلاغ عن المعاملات المشبوهة لوحدة الاستخبارات المالية.

رؤية البحرين 2030:
تنويع الاقتصاد وتقليل الاعتماد على النفط.
تطوير قطاع الخدمات المالية والتكنولوجيا المالية (Fintech).
تمكين الكوادر البحرينية في القطاع المالي.

SAMA — البنك المركزي السعودي:
ينظم القطاع المالي في المملكة العربية السعودية.
متطلبات الترخيص مشابهة لـ CBB مع اشتراطات إضافية للبنوك الإسلامية.
يشترط الالتزام بنظام ساما للمدفوعات الفورية (SADAD/SARIE).

UAECB — مصرف الإمارات المركزي:
ينظم البنوك في الإمارات العربية المتحدة.
رأس المال الأدنى: 150 مليون درهم إماراتي.
نظام AECB لتقارير الائتمان."""