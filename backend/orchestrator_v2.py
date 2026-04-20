"""
ACAI — Orchestrator v2
========================
Policy-based orchestrator replacing manual agent tab selection.
One query in → intent classified → pipeline built → agents chained → one merged answer out.

File: backend/orchestrator_v2.py
"""

import re
import logging
from typing import List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("acai.orchestrator")

# ─── Arabic detection ─────────────────────────────────────────────────────────
AR = re.compile(r'[\u0600-\u06FF]{3,}')

# ─── Keyword banks ────────────────────────────────────────────────────────────
RESEARCH_KW = [
    "latest", "recent", "news", "today", "2025", "2026", "current",
    "أحدث", "آخر", "أخبار", "حديث", "اليوم", "هذا الأسبوع", "الحين"
]
GCC_KW = [
    "cbb", "sama", "uaecb", "dfsa", "regulation", "law", "policy", "rulebook",
    "vision 2030", "رؤية", "قانون", "نظام", "تنظيم", "مصرف البحرين",
    "مصرف المركزي", "ترخيص", "امتثال", "compliance", "ضوابط", "banking",
    "fintech", "ريادة", "cbdc", "اشتراطات"
]
DIALECT_KW = [
    "لهجة", "dialect", "بحريني", "خليجي", "bahraini", "gulf", "معنى",
    "وايد", "حيل", "شلون", "ترجم", "translate", "تحليل لغوي", "تطبيع",
    "code-switching", "morphology", "فصحى", "صرف", "جذر"
]
EXTRACT_KW = [
    "extract", "entities", "relations", "استخرج", "كيانات",
    "علاقات", "knowledge graph", "رسم معرفي"
]
REASON_KW = [
    "why", "how", "explain", "analyze", "compare", "pros", "cons",
    "لماذا", "كيف", "اشرح", "حلل", "قارن", "ما الفرق", "ما الأسباب", "ناقش"
]

# ─── Agent constants ──────────────────────────────────────────────────────────
AGENTS = {
    "bahith":  {"label": "🔭 باحث",  "badge": "WEB SEARCH"},
    "musheer": {"label": "⚖️ مشير",  "badge": "GCC LAW"},
    "lughawi": {"label": "ع لغوي",   "badge": "ARABIC NLP"},
    "hakeem":  {"label": "🧠 حكيم",  "badge": "REASONING"},
    "muraqib": {"label": "🔍 مراقب", "badge": "VERIFY"},
    "bani":    {"label": "🕸️ بانِ",  "badge": "KNOWLEDGE"},
}

AGENT_SYSTEM_PROMPTS = {
    "bahith": (
        "أنت باحث في ACAI. قدّم معلومات دقيقة من مصادر موثوقة.\n"
        "التنسيق الإلزامي:\n"
        "**الملخص:** (2-3 جمل مباشرة)\n"
        "**النتائج الرئيسية:**\n• نتيجة + المصدر\n"
        "**التحليل:** (سياق أعمق)\n"
        "**المصادر:** (روابط فعلية)\n"
        "**الموثوقية:** X/10\n"
        "القاعدة: لا تخترع مصادر أبداً. أجب بنفس لغة السؤال."
    ),
    "musheer": (
        "أنت مشير — خبير أنظمة الخليج في ACAI.\n"
        "مراجعك: CBB Rulebook البحرين | SAMA السعودية | UAECB الإمارات | DFSA.\n"
        "التنسيق الإلزامي:\n"
        "**الحكم النظامي:** [المنظم | الوثيقة | القسم]\n"
        "**التفاصيل:** (شرح النظام)\n"
        "**المتطلبات العملية:** (خطوات أو شروط)\n"
        "**⚠️ تنبيه:** هذا تحليل استرشادي. راجع متخصصاً قانونياً.\n"
        "أجب بنفس لغة السؤال."
    ),
    "lughawi": (
        "أنت لغوي — خبير اللغة العربية وعلم اللهجات في ACAI.\n"
        "لأي نص عربي قدّم:\n"
        "**🗺️ اللهجة:** [بحرينية/خليجية/سعودية/مصرية/شامية/فصحى] — الثقة: X%\n"
        "**المؤشرات:** [الكلمات الدالة]\n"
        "**🔍 التحليل الصرفي:** (٣ كلمات: كلمة → جذر → وزن → معنى)\n"
        "**✍️ التطبيع للفصحى:** [النص بالعربية الفصحى]\n"
        "**🔄 التحول اللغوي:** [خلط عربي-إنجليزي إن وجد]\n"
        "**🌍 السياق الثقافي:** [ملاحظة ثقافية]"
    ),
    "hakeem": (
        "أنت حكيم — عميل التفكير العميق والاستدلال في ACAI.\n"
        "المنهج الإلزامي لكل إجابة:\n"
        "**خطوة ١ — التفكيك:** ما الذي يُسأل بالضبط؟\n"
        "**خطوة ٢ — المعرفة:** ما المبادئ والحقائق ذات الصلة؟\n"
        "**خطوة ٣ — الاستدلال:** الاستنتاجات المنطقية\n"
        "**خطوة ٤ — التحقق:** ما الذي قد يكون خاطئاً؟\n"
        "**خطوة ٥ — الإجابة النهائية**\n"
        "**الثقة:** X/10\n"
        "أجب بنفس لغة السؤال."
    ),
    "muraqib": (
        "أنت مراقب — عميل التحقق من الحقائق في ACAI.\n"
        "لكل ادعاء في الإجابة:\n"
        "✅ **صحيح:** [+ الدليل]\n"
        "⚠️ **غير محدد:** [يحتاج مصدراً]\n"
        "❌ **خاطئ:** [+ التصحيح الدقيق]\n"
        "**الحكم النهائي:** صحيح/جزئياً/خاطئ\n"
        "**الثقة:** X/10\n"
        "أسلوبك: صارم ومحايد."
    ),
    "bani": (
        "أنت بانِ — عميل استخراج المعرفة في ACAI.\n"
        "من أي نص استخرج:\n"
        "**الكيانات:**\n| الاسم | النوع | الثقة |\n|------|------|------|\n"
        "**العلاقات:**\n→ [كيان أ] —[العلاقة]→ [كيان ب]\n  الدليل: \"...\"\n"
        "**المفاهيم المحورية:** م١، م٢، م٣\n"
        "**التصنيف:** [المجال]"
    ),
}

# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Intent:
    research:     bool = False
    gcc_law:      bool = False
    dialect:      bool = False
    reasoning:    bool = False
    extraction:   bool = False
    verification: bool = True
    is_arabic:    bool = False

@dataclass
class OrchResult:
    query:        str
    pipeline:     List[str]
    intent:       Intent
    outputs:      Dict[str, str] = field(default_factory=dict)
    final:        str = ""
    memory_used:  bool = False
    rag_used:     bool = False
    latency_ms:   int = 0
    error:        str = ""


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# Rule-based — fast, deterministic, no LLM overhead
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(query: str) -> Intent:
    q   = query.lower()
    ar  = bool(AR.search(query))
    wc  = len(query.split())
    return Intent(
        research     = any(k in q for k in RESEARCH_KW),
        gcc_law      = any(k in q for k in GCC_KW),
        dialect      = ar or any(k in q for k in DIALECT_KW),
        reasoning    = wc > 12 or any(k in q for k in REASON_KW),
        extraction   = any(k in q for k in EXTRACT_KW),
        verification = wc > 5,
        is_arabic    = ar,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDER
# Ordered agent sequence based on intent
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(intent: Intent, mode: str = "auto") -> List[str]:
    # Forced single-agent mode (legacy compatibility)
    if mode.startswith("single:"):
        agent = mode.split(":", 1)[1]
        return [agent, "muraqib"] if agent != "muraqib" else ["muraqib"]

    # Legacy mode aliases
    LEGACY = {
        "arabic_nlp":    lambda: ["lughawi", "muraqib"],
        "knowledge":     lambda: ["bani", "muraqib"],
        "deep_research": lambda: ["bahith", "hakeem", "muraqib"],
        "gcc":           lambda: ["musheer", "hakeem", "muraqib"],
    }
    if mode in LEGACY:
        return LEGACY[mode]()

    # Auto policy
    p: List[str] = []

    # Stage 1 — gather
    if intent.research:  p.append("bahith")
    if intent.gcc_law:   p.append("musheer")

    # Stage 2 — language
    if intent.dialect:   p.append("lughawi")

    # Stage 3 — synthesize (when multiple sources or complex reasoning)
    if intent.reasoning or len(p) > 1:
        p.append("hakeem")

    # Stage 4 — extract
    if intent.extraction: p.append("bani")

    # Stage 5 — verify (always)
    if intent.verification: p.append("muraqib")

    # Fallback
    if not p:
        p = ["hakeem", "muraqib"]

    # Deduplicate preserving order
    seen, out = set(), []
    for a in p:
        if a not in seen:
            seen.add(a); out.append(a)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE MERGER
# Turns dict of agent outputs → one coherent answer
# ══════════════════════════════════════════════════════════════════════════════

def merge_outputs(pipeline: List[str], outputs: Dict[str, str]) -> str:
    if not outputs:
        return "لم أتمكن من توليد إجابة."

    # Single agent → return directly (no wrapping noise)
    valid = {a: o for a, o in outputs.items() if o and not o.startswith("[خطأ")}
    if len(valid) == 1:
        return next(iter(valid.values()))

    parts = []

    # Substantive agents first
    for agent in pipeline:
        if agent == "muraqib":
            continue
        out = valid.get(agent, "")
        if out:
            label = AGENTS[agent]["label"]
            parts.append(f"### {label}\n{out}")

    # Verifier last as a quality stamp
    if "muraqib" in valid:
        parts.append(f"\n---\n### 🔍 مراقب — التحقق\n{valid['muraqib']}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# CLI test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("ما أحدث أخبار الذكاء الاصطناعي في الخليج؟", "auto"),
        ("ما متطلبات ترخيص البنك في البحرين وفق CBB؟", "auto"),
        ("والله يا شباب الحين وايد زين هالمشروع حيل", "auto"),
        ("Why is AI alignment philosophically important?", "auto"),
        ("استخرج الكيانات من: مصرف البحرين المركزي ينظم القطاع المالي", "auto"),
        ("اشرح لهجة هذا النص", "single:lughawi"),
    ]
    print(f"\n{'='*65}")
    print("ACAI Orchestrator v2 — Intent & Pipeline Test")
    print(f"{'='*65}")
    for q, mode in tests:
        intent   = classify_intent(q)
        pipeline = build_pipeline(intent, mode)
        flags = f"research={intent.research} gcc={intent.gcc_law} " \
                f"dialect={intent.dialect} reason={intent.reasoning}"
        print(f"\nQ  : {q[:55]}")
        print(f"     {flags}")
        print(f"     Pipeline → {pipeline}")
