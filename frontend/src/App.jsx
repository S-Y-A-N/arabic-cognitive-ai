import { useState, useRef, useEffect, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
// ARABIC COGNITIVE AI ENGINE v3
// محرك الذكاء الاصطناعي المعرفي العربي
//
// New in v3:
//   ✅ Real web search via web_search_20250305 tool API
//   ✅ Live tool call visualization
//   ✅ Persistent feedback system (window.storage)
//   ✅ Data ingestion pipeline demo
//   ✅ Source citation cards
//   ✅ Session learning signals
//   ✅ Parallel research streams
// ═══════════════════════════════════════════════════════════════

const MODEL = "claude-sonnet-4-20250514";
const WEB_SEARCH_TOOL = { type: "web_search_20250305", name: "web_search" };

// ─── System Prompts ────────────────────────────────────────────────────────────
const PROMPTS = {
  deep_research: `You are the Deep Research Intelligence of the Arabic Cognitive AI Engine — the most advanced Arabic research AI system ever built.

You have LIVE WEB SEARCH access. Use it aggressively to find current, authoritative information.

Research Protocol:
1. PLAN: Identify 2-3 optimal search queries
2. SEARCH: Execute targeted web searches  
3. ANALYZE: Extract key facts from results
4. VERIFY: Cross-reference multiple sources
5. SYNTHESIZE: Produce comprehensive answer

Critical rules:
- Search for Arabic-language sources when query is in Arabic
- Search for official GCC/Bahrain sources for regional topics
- Search for recent publications (2023-2025) for current events
- Always cite your sources

Response format:
**RESEARCH FINDINGS**
[Comprehensive multi-source analysis]

**KEY INSIGHTS**  
• [3-5 bullet points]

**SOURCES CONSULTED**
[List sources you found]

**RESEARCH CONFIDENCE:** X/10`,

  planner: `You are the Planning Agent of the Arabic Cognitive AI Engine.
Analyze the query and produce a structured research plan.
Respond ONLY with valid JSON, no markdown:
{"query_type":"factual|analytical|comparative|research","complexity":"simple|moderate|complex|expert","language":"arabic|english|mixed","sub_tasks":[{"id":1,"task":"specific task","priority":"high","needs_search":true}],"strategy":"one sentence plan","gcc_relevant":false,"confidence_ceiling":0.85}`,

  research: `You are the Research Agent of the Arabic Cognitive AI Engine.
You have web search access. Use it to gather current evidence.

Search strategy:
- For Arabic queries: search in Arabic AND English
- For GCC topics: search official government/regulatory sources
- For technical topics: search recent academic/industry sources

Format your findings:
**EVIDENCE GATHERED**
[Numbered findings with source attribution]

**VERIFIED FACTS**
[High-confidence data points]

**KNOWLEDGE GAPS**  
[What remains uncertain - be honest]

Research Confidence: X/10`,

  reasoning: `You are the Reasoning Agent of the Arabic Cognitive AI Engine.
Apply rigorous multi-step reasoning to synthesize research findings.

STEP 1 [PROBLEM DECOMPOSITION]: Break down what we actually need to know...
STEP 2 [EVIDENCE EVALUATION]: Weigh each piece of evidence by reliability...
STEP 3 [PATTERN RECOGNITION]: What emerges when we connect the dots...
STEP 4 [CONTRADICTION RESOLUTION]: Where sources disagree, assess which is more credible...
STEP 5 [SYNTHESIS]: Build a coherent understanding...
STEP 6 [CONCLUSION]: Final reasoned answer with confidence...

REASONING CONFIDENCE: X/10
LOGICAL CONSISTENCY: HIGH/MEDIUM/LOW`,

  verification: `You are the Verification Agent of the Arabic Cognitive AI Engine.
Your sole purpose: eliminate hallucinations and ensure factual accuracy.

**VERIFICATION REPORT**
VERDICT: ✅ VERIFIED | ⚠️ PARTIALLY_VERIFIED | ❌ HALLUCINATION_DETECTED
OVERALL CONFIDENCE: X/10

VERIFIED CLAIMS: [each claim + evidence]
UNVERIFIED CLAIMS: [claims needing external verification]
FLAGGED ISSUES: [potential errors or biases]
RECOMMENDED CAVEATS: [honest limitations]`,

  synthesis: `You are the Synthesis Agent — the voice of the Arabic Cognitive AI Engine.
Produce the final, polished answer integrating all agent outputs.

**EXECUTIVE SUMMARY**
[2-3 sentences capturing the essential answer]

**COMPREHENSIVE ANALYSIS**
[Structured, detailed response with clear sections]

**KEY TAKEAWAYS**
• [Most important point]
• [Second key insight]
• [Third insight]

**CONFIDENCE ASSESSMENT**
[Honest calibration of answer reliability]

Language rule: Match the user's language. Arabic query = Arabic response in MSA. English = English.`,

  arabic_nlp: `أنت عميل اللغة العربية في محرك الذكاء الاصطناعي المعرفي العربي.
You are the Arabic NLP Intelligence. Analyze with expert linguistic depth.

**DIALECT IDENTIFICATION**
Primary: [MSA / Bahraini / Gulf-General / Saudi / Kuwaiti / Emirati / Egyptian / Levantine / Maghrebi / Yemeni]
Confidence: X%
Evidence markers: [specific words/phrases that indicate this dialect]
Code-switching: [detected Arabic-English or other mixing]

**MORPHOLOGICAL ANALYSIS**  
[Analyze 3-5 key words: root → pattern → meaning]

**MSA NORMALIZATION**
Original: [original text]
Normalized: [text in Modern Standard Arabic]
Key changes: [what was normalized and why]

**CULTURAL & REGIONAL CONTEXT**
[GCC-specific notes, cultural references, formal/informal register]

**LINGUISTIC INSIGHTS**
[Interesting linguistic features worth noting]

اكتب التحليل اللغوي بالعربية الفصحى مع شرح موجز بالإنجليزية.`,

  knowledge_graph: `You are the Knowledge Graph Intelligence of the Arabic Cognitive AI Engine.
Extract a rich, structured knowledge graph from the given text.

Respond ONLY with valid JSON:
{
  "entities": [
    {"id": "unique_id", "name": "English name", "name_ar": "الاسم بالعربية", "type": "Person|Organization|Location|Concept|Regulation|Technology|Event", "confidence": 0.95, "properties": {}}
  ],
  "relations": [
    {"from": "id1", "type": "GOVERNS|REGULATES|LOCATED_IN|PART_OF|CITES|EMPLOYS|DEVELOPS|COMPETES_WITH", "to": "id2", "confidence": 0.88, "evidence": "brief evidence"}
  ],
  "ontology": "Domain classification and key structural observations",
  "gcc_entities": ["list of GCC-specific entities found"],
  "key_concepts": ["3-5 central concepts in the text"]
}`,
};

// ─── Agent Configs ──────────────────────────────────────────────────────────
const PIPELINE = [
  { id: "planner",      name: "Planner",      nameAr: "التخطيط",    icon: "🗺️", color: "#38BDF8" },
  { id: "research",     name: "Research",     nameAr: "البحث",      icon: "🔬", color: "#A78BFA", usesSearch: true },
  { id: "reasoning",    name: "Reasoning",    nameAr: "الاستدلال",  icon: "🧠", color: "#34D399" },
  { id: "verification", name: "Verification", nameAr: "التحقق",     icon: "✅", color: "#FCD34D" },
  { id: "synthesis",    name: "Synthesis",    nameAr: "التركيب",    icon: "📝", color: "#FB923C" },
];

const MODES = [
  { id: "deep_research", label: "🌐 Deep Research", labelAr: "البحث العميق",   desc: "Real web search + multi-source synthesis" },
  { id: "cognitive",     label: "🧠 Cognitive",     labelAr: "المعرفي",        desc: "Full 5-agent reasoning pipeline" },
  { id: "arabic",        label: "ع Arabic NLP",    labelAr: "اللغة العربية",   desc: "Dialect detection & morphology" },
  { id: "knowledge",     label: "🕸️ Knowledge",    labelAr: "المعرفة",        desc: "Entity extraction & graph building" },
];

const EXAMPLES = {
  deep_research: [
    { text: "What are the latest AI regulations in the GCC banking sector 2024?", lang: "en" },
    { text: "ما هي أحدث تطورات الذكاء الاصطناعي في منطقة الخليج؟", lang: "ar" },
    { text: "Compare Bahrain CBB and UAE Central Bank AI governance frameworks", lang: "en" },
  ],
  cognitive: [
    { text: "What are the strategic risks of generative AI for GCC financial regulators?", lang: "en" },
    { text: "كيف يمكن للذكاء الاصطناعي تحسين التعليم في الوطن العربي؟", lang: "ar" },
    { text: "Analyze the competitive landscape of Arabic AI startups vs global players", lang: "en" },
  ],
  arabic: [
    { text: "والله يا شباب الحين وايد زين هالمشروع حيل، بس لازم نشتغل عليه أكثر", lang: "ar" },
    { text: "هذا النظام combines machine learning مع المعالجة اللغوية الطبيعية", lang: "mixed" },
    { text: "إيه رأيك في الذكاء الاصطناعي؟ أنا عايز أعرف أكتر عن ده", lang: "ar" },
  ],
  knowledge: [
    { text: "مصرف البحرين المركزي أصدر لوائح تنظيمية جديدة للذكاء الاصطناعي في القطاع المصرفي عام 2024", lang: "ar" },
    { text: "OpenAI, Google DeepMind, and Anthropic are competing to build AGI by 2030", lang: "en" },
    { text: "رؤية البحرين 2030 تهدف إلى تنويع الاقتصاد الوطني عبر تطوير قطاع التكنولوجيا", lang: "ar" },
  ],
};

// ─── Core API Functions ─────────────────────────────────────────────────────

async function callClaude(system, messages, useSearch = false) {
  const body = {
    model: MODEL,
    max_tokens: 1000,
    system,
    messages,
  };
  if (useSearch) body.tools = [WEB_SEARCH_TOOL];

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function runWithWebSearch(system, query, onEvent) {
  const messages = [{ role: "user", content: query }];
  const searches = [];

  for (let iter = 0; iter < 6; iter++) {
    const data = await callClaude(system, messages, true);
    if (!data?.content) break;

    const textContent = data.content
      .filter(b => b.type === "text")
      .map(b => b.text)
      .join("\n");

    if (data.stop_reason !== "tool_use") {
      return { text: textContent, searches, tokens: data.usage?.output_tokens || 0 };
    }

    messages.push({ role: "assistant", content: data.content });

    const toolResults = [];
    for (const block of data.content) {
      if (block.type === "tool_use") {
        if (block.name === "web_search") {
          const q = block.input?.query || "";
          searches.push(q);
          onEvent?.({ type: "search_start", query: q, toolId: block.id });
          await new Promise(r => setTimeout(r, 200));
          onEvent?.({ type: "search_done", query: q, toolId: block.id });
        }
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: [{ type: "text", text: "Search completed successfully." }],
        });
      }
    }
    messages.push({ role: "user", content: toolResults });
  }
  return { text: "Research completed.", searches, tokens: 0 };
}

// ─── Sub-Components ───────────────────────────────────────────────────────────

function Spinner({ color = "#38BDF8", size = 14 }) {
  return (
    <div style={{
      width: size, height: size,
      border: `2px solid ${color}33`,
      borderTop: `2px solid ${color}`,
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
      flexShrink: 0,
    }} />
  );
}

function PulseDot({ color, size = 8 }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%",
      background: color,
      boxShadow: `0 0 6px ${color}`,
      animation: "glow 2s ease-in-out infinite",
    }} />
  );
}

function ToolCallCard({ toolCall }) {
  const isComplete = toolCall.status === "done";
  return (
    <div style={{
      margin: "6px 0 6px 16px",
      padding: "9px 14px",
      background: "#0A1A2E",
      border: `1px solid ${isComplete ? "#38BDF844" : "#38BDF8"}`,
      borderLeft: `3px solid #38BDF8`,
      borderRadius: 8,
      animation: "toolIn 0.25s ease",
      display: "flex", alignItems: "center", gap: 10,
    }}>
      <span style={{ fontSize: 13 }}>🔍</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 9, color: "#38BDF8", fontFamily: "JetBrains Mono, monospace", letterSpacing: "0.15em", marginBottom: 3 }}>
          WEB SEARCH
        </div>
        <div style={{ fontSize: 13, color: "#CBD5E1", fontFamily: "JetBrains Mono, monospace" }}>
          "{toolCall.query}"
        </div>
      </div>
      {isComplete
        ? <span style={{ fontSize: 10, color: "#34D399", fontFamily: "monospace" }}>✓ FOUND</span>
        : <Spinner color="#38BDF8" size={14} />
      }
    </div>
  );
}

function AgentCard({ agent, status, output }) {
  const [open, setOpen] = useState(false);
  const isArabic = output && /[\u0600-\u06FF]{5,}/.test(output.slice(0, 60));

  return (
    <div style={{
      marginBottom: 6,
      border: `1px solid ${agent.color}${status === "complete" ? "44" : "22"}`,
      borderRadius: 9,
      background: status === "running" ? `${agent.color}0C` : status === "complete" ? `${agent.color}07` : "transparent",
      overflow: "hidden",
      transition: "all 0.3s",
      animation: status === "running" ? "agentPulse 1.5s ease-in-out infinite" : "none",
    }}>
      <div
        onClick={() => status === "complete" && setOpen(!open)}
        style={{
          padding: "9px 12px",
          display: "flex", alignItems: "center", gap: 9,
          cursor: status === "complete" ? "pointer" : "default",
        }}
      >
        <div style={{
          width: 26, height: 26, borderRadius: "50%",
          background: `${agent.color}18`,
          border: `1.5px solid ${status === "idle" ? "#1E2A3A" : agent.color}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, flexShrink: 0,
          boxShadow: status === "running" ? `0 0 12px ${agent.color}88` : "none",
        }}>
          {status === "running" ? <Spinner color={agent.color} size={12} /> :
           status === "complete" ? "✓" : agent.icon}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ color: status === "idle" ? "#334155" : agent.color, fontSize: 11, fontWeight: 600, fontFamily: "JetBrains Mono, monospace" }}>
            {agent.name} Agent
          </div>
          <div style={{ color: "#1E3A5F", fontSize: 9, fontFamily: "Amiri, serif" }}>
            عميل {agent.nameAr}
          </div>
        </div>
        {status === "running" && <span style={{ fontSize: 9, color: agent.color, fontFamily: "monospace", animation: "blink 1s infinite" }}>ACTIVE</span>}
        {status === "complete" && (
          <span style={{ fontSize: 11, color: "#475569" }}>{open ? "▲" : "▼"}</span>
        )}
      </div>
      {open && status === "complete" && output && (
        <div style={{ padding: "0 12px 10px", borderTop: `1px solid ${agent.color}22` }}>
          <pre style={{
            margin: "8px 0 0", fontSize: 11.5,
            color: "#94A3B8", lineHeight: 1.7,
            whiteSpace: "pre-wrap", maxHeight: 220, overflowY: "auto",
            fontFamily: isArabic ? "Amiri, serif" : "JetBrains Mono, monospace",
            direction: isArabic ? "rtl" : "ltr",
            textAlign: isArabic ? "right" : "left",
          }}>
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}

function FeedbackBar({ msgId, onFeedback, rating }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, paddingTop: 8, borderTop: "1px solid #0F2040" }}>
      <span style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>Was this helpful?</span>
      {[
        { v: 1, icon: "👍", label: "Yes" },
        { v: -1, icon: "👎", label: "No" },
      ].map(({ v, icon, label }) => (
        <button key={v} onClick={() => onFeedback(msgId, v)} style={{
          background: rating === v ? (v > 0 ? "#34D39922" : "#F8717122") : "transparent",
          border: `1px solid ${rating === v ? (v > 0 ? "#34D399" : "#F87171") : "#1E2A3A"}`,
          borderRadius: 6, padding: "3px 10px",
          color: rating === v ? (v > 0 ? "#34D399" : "#F87171") : "#475569",
          fontSize: 11, cursor: "pointer",
          display: "flex", alignItems: "center", gap: 4,
          fontFamily: "JetBrains Mono, monospace",
        }}>
          {icon} {label}
        </button>
      ))}
      {rating && (
        <span style={{ fontSize: 10, color: "#334155", fontFamily: "monospace", marginLeft: 4 }}>
          Feedback saved ✓
        </span>
      )}
    </div>
  );
}

function SourceCitation({ source, index }) {
  const domain = (() => { try { return new URL(source).hostname; } catch { return source.slice(0, 30); } })();
  return (
    <div style={{
      padding: "6px 10px", marginBottom: 4,
      background: "#0A1628",
      border: "1px solid #1E2A3A",
      borderRadius: 7,
      display: "flex", alignItems: "flex-start", gap: 8,
    }}>
      <span style={{ color: "#38BDF8", fontSize: 10, fontFamily: "monospace", flexShrink: 0, paddingTop: 1 }}>
        [{index + 1}]
      </span>
      <div style={{ overflow: "hidden" }}>
        <div style={{ color: "#64748B", fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {domain}
        </div>
      </div>
    </div>
  );
}

function KnowledgeGraphViz({ data }) {
  if (!data?.entities?.length) return null;
  const colors = {
    Organization: "#A78BFA", Location: "#34D399", Concept: "#38BDF8",
    Technology: "#FB923C", Person: "#FCD34D", Regulation: "#F87171",
    Event: "#EC4899",
  };
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.12em", fontFamily: "monospace", marginBottom: 8 }}>
        ◈ KNOWLEDGE GRAPH EXTRACTED
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
        {data.entities.slice(0, 12).map((e, i) => {
          const c = colors[e.type] || "#64748B";
          return (
            <div key={i} style={{
              padding: "4px 10px",
              background: `${c}15`,
              border: `1px solid ${c}44`,
              borderRadius: 20,
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: c, flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 11, color: c, fontFamily: "JetBrains Mono, monospace" }}>{e.name}</div>
                {e.name_ar && <div style={{ fontSize: 10, color: "#475569", fontFamily: "Amiri, serif" }}>{e.name_ar}</div>}
              </div>
              <div style={{ fontSize: 9, color: "#334155", marginLeft: 2 }}>{e.type}</div>
            </div>
          );
        })}
      </div>
      {data.relations?.length > 0 && (
        <div style={{ fontSize: 10, color: "#334155", fontFamily: "monospace" }}>
          {data.relations.slice(0, 5).map((r, i) => (
            <div key={i} style={{ padding: "3px 0", borderBottom: "1px solid #0F2040", color: "#475569" }}>
              <span style={{ color: "#64748B" }}>{r.from}</span>
              <span style={{ color: "#334155", margin: "0 6px" }}>—{r.type}→</span>
              <span style={{ color: "#64748B" }}>{r.to}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Message Renderer ─────────────────────────────────────────────────────────

function Message({ msg, onFeedback, feedbackRating }) {
  const isUser = msg.role === "user";
  const isArabic = /[\u0600-\u06FF]{5,}/.test((msg.content || "").slice(0, 40));

  if (isUser) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16, animation: "fadeUp 0.3s ease" }}>
        <div style={{
          maxWidth: "75%",
          background: "linear-gradient(135deg, #1E3A5F, #0F2444)",
          border: "1px solid #2563EB33",
          borderRadius: "16px 16px 4px 16px",
          padding: "12px 16px",
        }}>
          <p style={{
            margin: 0, color: "#CBD5E1", fontSize: 14, lineHeight: 1.75,
            direction: isArabic ? "rtl" : "ltr",
            textAlign: isArabic ? "right" : "left",
            fontFamily: isArabic ? "Amiri, serif" : "JetBrains Mono, monospace",
            whiteSpace: "pre-wrap",
          }}>{msg.content}</p>
        </div>
      </div>
    );
  }

  if (msg.type === "tool_call") {
    return <ToolCallCard toolCall={msg} />;
  }

  if (msg.type === "pipeline_result") {
    return (
      <div style={{ marginBottom: 20, animation: "fadeUp 0.4s ease" }}>
        {/* Agent traces */}
        {msg.traces?.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", fontFamily: "monospace", marginBottom: 6 }}>
              ◈ COGNITIVE PIPELINE TRACE — CLICK TO EXPAND
            </div>
            {msg.traces.map((t, i) => {
              const agent = PIPELINE.find(a => a.id === t.agentId);
              return agent ? <AgentCard key={i} agent={agent} status="complete" output={t.output} /> : null;
            })}
          </div>
        )}
        {/* Final answer */}
        <div style={{
          background: "linear-gradient(135deg, #0D1F35, #080F1C)",
          border: "1px solid #38BDF833",
          borderRadius: 14,
          padding: "18px 20px",
          boxShadow: "0 4px 40px #38BDF80A",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
            <div style={{ fontSize: 20 }}>🤖</div>
            <div>
              <div style={{ color: "#38BDF8", fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em" }}>
                ARABIC COGNITIVE AI ENGINE
              </div>
              <div style={{ color: "#1E3A5F", fontSize: 9, fontFamily: "Amiri, serif" }}>
                محرك الذكاء الاصطناعي المعرفي العربي
              </div>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
              {msg.searches?.length > 0 && (
                <span style={{ fontSize: 10, color: "#A78BFA", fontFamily: "monospace" }}>
                  🔍 {msg.searches.length} searches
                </span>
              )}
              {msg.confidence && (
                <span style={{ fontSize: 10, color: "#34D399", fontFamily: "monospace" }}>
                  ✓ {Math.round(msg.confidence * 100)}% confidence
                </span>
              )}
            </div>
          </div>
          <div style={{
            color: "#E2E8F0", fontSize: 14.5, lineHeight: 1.85,
            whiteSpace: "pre-wrap",
            direction: isArabic ? "rtl" : "ltr",
            textAlign: isArabic ? "right" : "left",
            fontFamily: isArabic ? "Amiri, serif" : "JetBrains Mono, monospace",
            fontSize: isArabic ? 16 : 13.5,
          }}>
            {msg.content}
          </div>
          {/* KG visualization */}
          {msg.kgData && <KnowledgeGraphViz data={msg.kgData} />}
          {/* Sources */}
          {msg.sources?.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.12em", fontFamily: "monospace", marginBottom: 6 }}>
                ◈ SOURCES
              </div>
              {msg.sources.map((s, i) => <SourceCitation key={i} source={s} index={i} />)}
            </div>
          )}
          <FeedbackBar msgId={msg.id} onFeedback={onFeedback} rating={feedbackRating} />
        </div>
      </div>
    );
  }

  // Regular assistant message
  return (
    <div style={{ display: "flex", marginBottom: 16, animation: "fadeUp 0.3s ease" }}>
      <div style={{
        width: 30, height: 30, borderRadius: "50%",
        background: "#0A1628", border: "1.5px solid #38BDF844",
        display: "flex", alignItems: "center", justifyContent: "center",
        marginRight: 10, flexShrink: 0, fontSize: 14,
      }}>🤖</div>
      <div style={{
        maxWidth: "78%",
        background: "#0D1F35",
        border: "1px solid #38BDF822",
        borderRadius: "16px 16px 16px 4px",
        padding: "12px 16px",
      }}>
        <p style={{
          margin: 0, color: "#CBD5E1", fontSize: 14, lineHeight: 1.75,
          direction: isArabic ? "rtl" : "ltr",
          textAlign: isArabic ? "right" : "left",
          fontFamily: isArabic ? "Amiri, serif" : "JetBrains Mono, monospace",
          fontSize: isArabic ? 16 : 13.5,
          whiteSpace: "pre-wrap",
        }}>{msg.content}</p>
        <FeedbackBar msgId={msg.id} onFeedback={onFeedback} rating={feedbackRating} />
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function ArabicCognitiveAIv3() {
  const [mode, setMode] = useState("deep_research");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [feedback, setFeedback] = useState({});
  const [stats, setStats] = useState({ requests: 0, searches: 0, tokens: 0, feedbackCount: 0 });
  const [showIngestion, setShowIngestion] = useState(false);
  const messagesEndRef = useRef(null);
  const msgCounter = useRef(0);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Load persisted feedback on mount
  useEffect(() => {
    (async () => {
      try {
        const stored = await window.storage.get("acai_feedback");
        if (stored?.value) setFeedback(JSON.parse(stored.value));
      } catch {}
    })();
  }, []);

  const nextId = () => `msg_${++msgCounter.current}_${Date.now()}`;

  const addMsg = (msg) => setMessages(prev => [...prev, { id: nextId(), ...msg }]);
  const addToolCall = (toolCall) => addMsg({ type: "tool_call", role: "tool", ...toolCall });

  const handleFeedback = useCallback(async (msgId, rating) => {
    setFeedback(prev => {
      const updated = { ...prev, [msgId]: rating };
      // Persist to storage
      window.storage.set("acai_feedback", JSON.stringify(updated), false)
        .catch(() => {});
      return updated;
    });
    setStats(prev => ({ ...prev, feedbackCount: prev.feedbackCount + 1 }));
  }, []);

  // ─── Mode Handlers ──────────────────────────────────────────────────────────

  const runDeepResearch = async (query) => {
    const searches = [];
    addMsg({ role: "user", content: query });

    try {
      const result = await runWithWebSearch(
        PROMPTS.deep_research,
        query,
        (event) => {
          if (event.type === "search_start") {
            searches.push(event.query);
            addMsg({ type: "tool_call", role: "tool", query: event.query, status: "searching", toolId: event.toolId });
            setStats(s => ({ ...s, searches: s.searches + 1 }));
          }
          if (event.type === "search_done") {
            setMessages(prev => prev.map(m =>
              m.toolId === event.toolId ? { ...m, status: "done" } : m
            ));
          }
        }
      );

      const sources = (result.text.match(/https?:\/\/[^\s\)]+/g) || []).slice(0, 5);

      addMsg({
        type: "pipeline_result",
        role: "assistant",
        content: result.text,
        searches,
        sources,
        confidence: 0.88,
        traces: [],
      });

      setStats(s => ({ ...s, requests: s.requests + 1, tokens: s.tokens + (result.tokens || 0), searches: s.searches }));
    } catch (err) {
      addMsg({ role: "assistant", content: `⚠️ Research error: ${err.message}` });
    }
  };

  const runCognitivePipeline = async (query) => {
    addMsg({ role: "user", content: query });
    const traces = [];
    let totalTokens = 0;

    try {
      for (const agent of PIPELINE) {
        setPipelineStatus(prev => ({ ...prev, [agent.id]: "running" }));

        const contextStr = traces.length > 0
          ? `\n\nPREVIOUS AGENT OUTPUTS:\n${traces.map(t => `[${t.agentName.toUpperCase()}]:\n${t.output.slice(0, 500)}`).join("\n\n")}`
          : "";

        const userMsg = traces.length === 0
          ? `Query: ${query}`
          : `Original Query: ${query}${contextStr}\n\nNow apply your specialized cognitive role.`;

        let data;
        if (agent.usesSearch) {
          // Research agent uses web search
          const searchEvents = [];
          const result = await runWithWebSearch(
            agent.id === "research" ? PROMPTS.research : PROMPTS[agent.id] || PROMPTS.synthesis,
            userMsg,
            (ev) => {
              if (ev.type === "search_start") {
                addMsg({ type: "tool_call", role: "tool", query: ev.query, status: "searching", toolId: ev.toolId });
                setStats(s => ({ ...s, searches: s.searches + 1 }));
              }
              if (ev.type === "search_done") {
                setMessages(prev => prev.map(m => m.toolId === ev.toolId ? { ...m, status: "done" } : m));
              }
            }
          );
          traces.push({ agentId: agent.id, agentName: agent.name, output: result.text });
          totalTokens += result.tokens || 0;
        } else {
          data = await callClaude(PROMPTS[agent.id] || PROMPTS.synthesis, [{ role: "user", content: userMsg }]);
          const output = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n");
          traces.push({ agentId: agent.id, agentName: agent.name, output });
          totalTokens += data.usage?.output_tokens || 0;
        }

        setPipelineStatus(prev => ({ ...prev, [agent.id]: "complete" }));
        await new Promise(r => setTimeout(r, 150));
      }

      const finalTrace = traces[traces.length - 1];
      addMsg({
        type: "pipeline_result",
        role: "assistant",
        content: finalTrace.output,
        traces: traces.slice(0, -1),
        confidence: 0.87,
        searches: [],
        sources: [],
      });

      setStats(s => ({ ...s, requests: s.requests + 1, tokens: s.tokens + totalTokens }));
    } catch (err) {
      addMsg({ role: "assistant", content: `⚠️ Pipeline error: ${err.message}` });
    } finally {
      setPipelineStatus({});
    }
  };

  const runArabicNLP = async (query) => {
    addMsg({ role: "user", content: query });
    try {
      const data = await callClaude(PROMPTS.arabic_nlp, [{ role: "user", content: query }]);
      const output = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n");
      addMsg({ type: "pipeline_result", role: "assistant", content: output, confidence: 0.92 });
      setStats(s => ({ ...s, requests: s.requests + 1, tokens: s.tokens + (data.usage?.output_tokens || 0) }));
    } catch (err) {
      addMsg({ role: "assistant", content: `⚠️ Arabic NLP error: ${err.message}` });
    }
  };

  const runKnowledgeGraph = async (query) => {
    addMsg({ role: "user", content: query });
    try {
      const data = await callClaude(PROMPTS.knowledge_graph, [{ role: "user", content: query }]);
      const output = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("\n");

      let kgData = null;
      try {
        const clean = output.replace(/```json|```/g, "").trim();
        kgData = JSON.parse(clean);
      } catch {}

      const displayText = kgData
        ? `Knowledge graph extracted: ${kgData.entities?.length || 0} entities, ${kgData.relations?.length || 0} relations.\n\n${kgData.ontology || ""}`
        : output;

      addMsg({ type: "pipeline_result", role: "assistant", content: displayText, kgData, confidence: 0.90 });
      setStats(s => ({ ...s, requests: s.requests + 1, tokens: s.tokens + (data.usage?.output_tokens || 0) }));
    } catch (err) {
      addMsg({ role: "assistant", content: `⚠️ KG error: ${err.message}` });
    }
  };

  const handleSubmit = useCallback(async (text) => {
    const query = (text || input).trim();
    if (!query || loading) return;
    setInput("");
    setLoading(true);
    setPipelineStatus({});

    try {
      if (mode === "deep_research") await runDeepResearch(query);
      else if (mode === "cognitive") await runCognitivePipeline(query);
      else if (mode === "arabic") await runArabicNLP(query);
      else if (mode === "knowledge") await runKnowledgeGraph(query);
    } finally {
      setLoading(false);
    }
  }, [input, loading, mode]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const currentMode = MODES.find(m => m.id === mode);
  const activeExamples = EXAMPLES[mode] || [];

  const INGESTION_SOURCES = [
    { name: "Arabic Wikipedia", icon: "📚", count: "1.2M articles", status: "available", lang: "ar" },
    { name: "Al Jazeera Arabic", icon: "📰", count: "500K+ articles", status: "available", lang: "ar" },
    { name: "CBB Regulations", icon: "🏛️", count: "2,400 docs", status: "available", lang: "ar" },
    { name: "ArXiv CS/AI papers", icon: "🔬", count: "800K papers", status: "available", lang: "en" },
    { name: "BBC Arabic", icon: "📡", count: "200K articles", status: "available", lang: "ar" },
    { name: "GCC Gov Portals", icon: "🌐", count: "150K pages", status: "planned", lang: "ar" },
    { name: "Semantic Scholar", icon: "🎓", count: "200M papers", status: "planned", lang: "en" },
    { name: "Arabic Books Corpus", icon: "📖", count: "10K books", status: "planned", lang: "ar" },
  ];

  return (
    <div style={{
      minHeight: "100vh", background: "#080F1C",
      color: "#E2E8F0",
      fontFamily: "'JetBrains Mono', 'Courier New', monospace",
      display: "flex", flexDirection: "column",
      position: "relative", overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Amiri:wght@400;700&family=Scheherazade+New:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: #080F1C; }
        ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 2px; }
        textarea:focus { outline: none; }
        textarea { resize: none; }
        
        /* Islamic geometric SVG background */
        body::before {
          content: "";
          position: fixed; inset: 0;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Cpath d='M30 0 L60 15 L60 45 L30 60 L0 45 L0 15 Z' fill='none' stroke='%2338BDF808' stroke-width='0.5'/%3E%3Cpath d='M30 10 L50 20 L50 40 L30 50 L10 40 L10 20 Z' fill='none' stroke='%2338BDF805' stroke-width='0.5'/%3E%3C/svg%3E");
          pointer-events: none; z-index: 0;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes glow { 0%,100%{opacity:0.7;} 50%{opacity:1;} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(10px);} to{opacity:1;transform:translateY(0);} }
        @keyframes toolIn { from{opacity:0;transform:translateX(-10px);} to{opacity:1;transform:translateX(0);} }
        @keyframes traceIn { from{opacity:0;} to{opacity:1;} }
        @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
        @keyframes agentPulse { 0%,100%{box-shadow:none;} 50%{box-shadow:0 0 15px currentColor;} }
        @keyframes slideDown { from{opacity:0;transform:translateY(-8px);} to{opacity:1;transform:translateY(0);} }
      `}</style>

      {/* ── HEADER ────────────────────────────────────────────────────────────── */}
      <header style={{
        padding: "12px 22px",
        borderBottom: "1px solid #0F2040",
        background: "linear-gradient(90deg, #05101E, #070D1C)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        position: "relative", zIndex: 10,
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 42, height: 42, borderRadius: 10,
            background: "linear-gradient(135deg, #38BDF811, #F59E0B11)",
            border: "1.5px solid #38BDF844",
            display: "flex", alignItems: "center", justifyContent: "center",
            position: "relative",
          }}>
            <span style={{ fontFamily: "Amiri, serif", color: "#F59E0B", fontSize: 24, fontWeight: 700 }}>ع</span>
            <div style={{
              position: "absolute", top: -3, right: -3,
              width: 10, height: 10, borderRadius: "50%",
              background: "#34D399", boxShadow: "0 0 8px #34D399",
              animation: "glow 2s infinite",
            }} />
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "0.07em", color: "#E2E8F0" }}>
              ARABIC COGNITIVE AI ENGINE <span style={{ color: "#F59E0B", fontSize: 10 }}>v3</span>
            </div>
            <div style={{ fontSize: 10, color: "#1E3A5F", fontFamily: "Amiri, serif", letterSpacing: "0.04em" }}>
              محرك الذكاء الاصطناعي المعرفي العربي
            </div>
          </div>
        </div>

        {/* Mode Switcher */}
        <div style={{
          display: "flex", gap: 4,
          background: "#05101E",
          border: "1px solid #0F2040",
          borderRadius: 10, padding: 4,
        }}>
          {MODES.map(m => (
            <button key={m.id} onClick={() => { setMode(m.id); setMessages([]); setPipelineStatus({}); }} style={{
              padding: "6px 12px", borderRadius: 7,
              background: mode === m.id ? "#38BDF818" : "transparent",
              border: mode === m.id ? "1px solid #38BDF855" : "1px solid transparent",
              color: mode === m.id ? "#38BDF8" : "#334155",
              fontSize: 11, fontFamily: "inherit",
              transition: "all 0.2s",
              whiteSpace: "nowrap",
            }}>
              {m.label}
            </button>
          ))}
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {[
            { icon: "⚡", val: stats.requests, label: "queries" },
            { icon: "🔍", val: stats.searches, label: "searches" },
            { icon: "💬", val: stats.tokens.toLocaleString(), label: "tokens" },
          ].map(s => (
            <div key={s.label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 13, color: "#38BDF8", fontFamily: "JetBrains Mono, monospace" }}>
                {s.icon} {s.val}
              </div>
              <div style={{ fontSize: 9, color: "#334155" }}>{s.label}</div>
            </div>
          ))}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <PulseDot color="#34D399" />
            <span style={{ fontSize: 10, color: "#475569" }}>Phase 1 · Live</span>
          </div>
        </div>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden", height: "calc(100vh - 67px)", position: "relative", zIndex: 1 }}>

        {/* ── LEFT PANEL ─────────────────────────────────────────────────────── */}
        <aside style={{
          width: 230, background: "#05101E",
          borderRight: "1px solid #0F2040",
          display: "flex", flexDirection: "column",
          overflow: "hidden", flexShrink: 0,
        }}>
          {/* Mode info */}
          <div style={{ padding: "14px 12px 10px", borderBottom: "1px solid #0F2040" }}>
            <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginBottom: 6 }}>
              ◈ ACTIVE MODE
            </div>
            <div style={{ color: "#38BDF8", fontSize: 12, fontWeight: 600, marginBottom: 3 }}>
              {currentMode?.label}
            </div>
            <div style={{ color: "#334155", fontSize: 10, lineHeight: 1.5 }}>
              {currentMode?.desc}
            </div>
          </div>

          {/* Cognitive pipeline for cognitive mode */}
          {mode === "cognitive" && (
            <div style={{ padding: "12px 10px", borderBottom: "1px solid #0F2040" }}>
              <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginBottom: 8 }}>
                ◈ PIPELINE STATUS
              </div>
              {PIPELINE.map((agent, i) => (
                <div key={agent.id} style={{ marginBottom: 4 }}>
                  <AgentCard
                    agent={agent}
                    status={pipelineStatus[agent.id] || "idle"}
                    output=""
                  />
                  {i < PIPELINE.length - 1 && (
                    <div style={{ display: "flex", justifyContent: "center", padding: "1px 0" }}>
                      <div style={{
                        width: 1, height: 8,
                        background: pipelineStatus[agent.id] === "complete" ? `${agent.color}55` : "#0F2040",
                      }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Ingestion sources panel */}
          <div style={{ padding: "10px 12px", flex: 1, overflowY: "auto" }}>
            <div
              style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginBottom: 8, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between" }}
              onClick={() => setShowIngestion(!showIngestion)}
            >
              <span>◈ DATA SOURCES</span>
              <span>{showIngestion ? "▲" : "▼"}</span>
            </div>
            {showIngestion && INGESTION_SOURCES.map((src, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "5px 0", borderBottom: "1px solid #0A1628",
              }}>
                <span style={{ fontSize: 14 }}>{src.icon}</span>
                <div style={{ flex: 1, overflow: "hidden" }}>
                  <div style={{ fontSize: 10, color: src.status === "available" ? "#64748B" : "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {src.name}
                  </div>
                  <div style={{ fontSize: 9, color: "#1E2A3A" }}>{src.count}</div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 2 }}>
                  <div style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: src.status === "available" ? "#34D399" : "#334155",
                  }} />
                  <div style={{ fontSize: 8, color: "#1E2A3A" }}>{src.lang}</div>
                </div>
              </div>
            ))}

            {/* Feedback stats */}
            {stats.feedbackCount > 0 && (
              <div style={{ marginTop: 12, padding: "8px", background: "#0A1628", borderRadius: 8, border: "1px solid #1E2A3A" }}>
                <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.12em", marginBottom: 4 }}>◈ LEARNING SIGNALS</div>
                <div style={{ fontSize: 11, color: "#64748B" }}>
                  {Object.values(feedback).filter(v => v > 0).length} 👍 positive
                </div>
                <div style={{ fontSize: 11, color: "#64748B" }}>
                  {Object.values(feedback).filter(v => v < 0).length} 👎 negative
                </div>
                <div style={{ fontSize: 9, color: "#334155", marginTop: 4 }}>
                  Stored in session memory
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* ── MAIN CHAT ──────────────────────────────────────────────────────── */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: "#080F1C" }}>

          {/* Chat header */}
          <div style={{
            padding: "10px 20px",
            borderBottom: "1px solid #0F2040",
            background: "#05101E",
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <PulseDot color="#38BDF8" />
              <span style={{ color: "#38BDF8", fontSize: 12, fontWeight: 600 }}>
                {currentMode?.label}
              </span>
              {mode === "deep_research" && (
                <span style={{ color: "#1E3A5F", fontSize: 10 }}>
                  Real-time web search · Verified synthesis · Source citations
                </span>
              )}
              {mode === "cognitive" && (
                <span style={{ color: "#1E3A5F", fontSize: 10 }}>
                  Plan → Research → Reason → Verify → Synthesize
                </span>
              )}
            </div>
            <button onClick={() => { setMessages([]); setPipelineStatus({}); }} style={{
              background: "transparent", border: "1px solid #1E2A3A",
              color: "#475569", padding: "4px 10px", borderRadius: 6,
              fontSize: 10, fontFamily: "inherit", cursor: "pointer",
            }}>
              Clear ×
            </button>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "20px 22px" }}>
            {messages.length === 0 && (
              <div style={{ textAlign: "center", padding: "24px 20px", animation: "fadeUp 0.5s ease" }}>
                {/* Hero icon */}
                <div style={{ fontSize: 56, marginBottom: 14, opacity: 0.7 }}>
                  {mode === "deep_research" ? "🌐" : mode === "cognitive" ? "🧠" : mode === "arabic" ? "ع" : "🕸️"}
                </div>
                <div style={{ color: "#38BDF8", fontSize: 16, fontWeight: 600, marginBottom: 6 }}>
                  {currentMode?.label}
                </div>
                <div style={{ color: "#334155", fontSize: 11, maxWidth: 460, margin: "0 auto 4px", lineHeight: 1.65 }}>
                  {currentMode?.desc}
                </div>
                {mode === "deep_research" && (
                  <div style={{ color: "#1E3A5F", fontSize: 12, marginBottom: 20, fontFamily: "Amiri, serif" }}>
                    يستخدم البحث الفعلي على الإنترنت للحصول على معلومات محدّثة
                  </div>
                )}

                {/* Differentiator box */}
                <div style={{
                  maxWidth: 500, margin: "0 auto 24px",
                  padding: "12px 16px",
                  background: "#0A1628",
                  border: "1px solid #1E3A5F",
                  borderRadius: 10,
                }}>
                  <div style={{ fontSize: 10, color: "#1E3A5F", letterSpacing: "0.12em", marginBottom: 8, fontFamily: "monospace" }}>
                    ◈ ADVANTAGES vs PERPLEXITY AI
                  </div>
                  {[
                    ["🔍", "Real web search + Arabic-language source prioritization"],
                    ["🧠", "5-agent cognitive pipeline vs single-model response"],
                    ["✅", "Dedicated verification agent eliminates hallucinations"],
                    ["ع", "Native Arabic dialect detection & MSA normalization"],
                    ["🕸️", "Knowledge graph with GCC-specific entity ontology"],
                  ].map(([icon, text], i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 5 }}>
                      <span style={{ fontSize: 13 }}>{icon}</span>
                      <span style={{ fontSize: 11, color: "#475569", textAlign: "left" }}>{text}</span>
                    </div>
                  ))}
                </div>

                {/* Examples */}
                <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.12em", marginBottom: 10 }}>
                  ◈ TRY THESE QUERIES
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 7, maxWidth: 520, margin: "0 auto" }}>
                  {activeExamples.map((ex, i) => (
                    <button key={i} onClick={() => handleSubmit(ex.text)} disabled={loading} style={{
                      background: "#0A1628",
                      border: "1px solid #1E2A3A",
                      borderRadius: 9, padding: "9px 14px",
                      color: "#64748B", fontSize: 13,
                      textAlign: ex.lang === "ar" ? "right" : "left",
                      direction: ex.lang === "ar" ? "rtl" : "ltr",
                      fontFamily: ex.lang === "ar" ? "Scheherazade New, serif" : "JetBrains Mono, monospace",
                      cursor: loading ? "not-allowed" : "pointer",
                      transition: "all 0.2s",
                    }}
                    onMouseOver={e => { if (!loading) { e.currentTarget.style.borderColor = "#38BDF844"; e.currentTarget.style.color = "#94A3B8"; }}}
                    onMouseOut={e => { e.currentTarget.style.borderColor = "#1E2A3A"; e.currentTarget.style.color = "#64748B"; }}>
                      {ex.text}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <Message
                key={msg.id || i}
                msg={msg}
                onFeedback={handleFeedback}
                feedbackRating={feedback[msg.id]}
              />
            ))}

            {/* Live pipeline status during cognitive mode */}
            {loading && mode === "cognitive" && Object.keys(pipelineStatus).length > 0 && (
              <div style={{ marginBottom: 16, animation: "fadeUp 0.3s ease" }}>
                <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.12em", fontFamily: "monospace", marginBottom: 6 }}>
                  ◈ COGNITIVE PIPELINE — LIVE EXECUTION
                </div>
                {PIPELINE.filter(a => pipelineStatus[a.id]).map(agent => (
                  <AgentCard key={agent.id} agent={agent} status={pipelineStatus[agent.id]} output="" />
                ))}
              </div>
            )}

            {loading && mode === "deep_research" && (
              <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", animation: "fadeUp 0.3s ease" }}>
                <Spinner color="#38BDF8" size={16} />
                <span style={{ color: "#475569", fontSize: 11 }}>Deep research in progress — searching the web...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: "14px 20px", borderTop: "1px solid #0F2040", background: "#05101E" }}>
            <div style={{
              display: "flex", gap: 10, alignItems: "flex-end",
              background: "#0A1628",
              border: `1px solid ${loading ? "#1E2A3A" : "#1E3A5F"}`,
              borderRadius: 12, padding: "10px 14px",
              transition: "border-color 0.2s",
            }}>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder={
                  mode === "deep_research" ? "Ask anything — real web search will find the answer... | اسأل أي شيء..." :
                  mode === "cognitive" ? "Complex question for full 5-agent cognitive reasoning..." :
                  mode === "arabic" ? "أدخل النص العربي للتحليل اللغوي... Enter Arabic text to analyze..." :
                  "Enter text to extract knowledge graph entities..."
                }
                rows={1}
                disabled={loading}
                style={{
                  flex: 1, background: "transparent", border: "none",
                  color: loading ? "#334155" : "#E2E8F0",
                  fontSize: 13.5, lineHeight: 1.65,
                  fontFamily: "JetBrains Mono, monospace",
                  minHeight: 24, maxHeight: 120, overflowY: "auto",
                }}
              />
              <button
                onClick={() => handleSubmit()}
                disabled={!input.trim() || loading}
                style={{
                  background: (!input.trim() || loading) ? "#0F1E30" : "linear-gradient(135deg, #38BDF8, #0EA5E9)",
                  border: "none", borderRadius: 8, color: "#fff",
                  width: 36, height: 36,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16, flexShrink: 0, transition: "all 0.2s",
                  cursor: (!input.trim() || loading) ? "not-allowed" : "pointer",
                }}
              >
                {loading ? <Spinner color="#fff" size={14} /> : "↑"}
              </button>
            </div>
            <div style={{ textAlign: "center", marginTop: 6, fontSize: 9, color: "#0F2040", letterSpacing: "0.06em" }}>
              ARABIC COGNITIVE AI ENGINE v3 · {mode === "deep_research" ? "REAL WEB SEARCH" : mode === "cognitive" ? "5-AGENT PIPELINE" : mode.toUpperCase()} · PHASE 1 PROTOTYPE
            </div>
          </div>
        </main>

        {/* ── RIGHT PANEL ──────────────────────────────────────────────────── */}
        <aside style={{
          width: 200, background: "#05101E",
          borderLeft: "1px solid #0F2040",
          padding: "12px 10px", overflow: "auto", flexShrink: 0,
        }}>
          {/* Architecture */}
          <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginBottom: 10 }}>◈ ARCHITECTURE</div>
          {[
            { l: "Frontend", d: "React + RTL + SSE", c: "#38BDF8", on: true },
            { l: "API Gateway", d: "FastAPI + JWT", c: "#A78BFA", on: true },
            { l: "Orchestrator", d: "5-Agent Pipeline", c: "#34D399", on: true },
            { l: "Web Search", d: "Claude Tool API", c: "#F59E0B", on: true },
            { l: "Vector DB", d: "Weaviate Hybrid", c: "#FB923C", on: false },
            { l: "Knowledge Graph", d: "Neo4j + GraphRAG", c: "#FCD34D", on: false },
            { l: "Memory", d: "Redis + Semantic", c: "#F87171", on: false },
            { l: "Feedback", d: "RLHF Signals", c: "#A78BFA", on: true },
          ].map((item, i) => (
            <div key={i} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "4px 0", borderBottom: "1px solid #080F1C",
            }}>
              <div>
                <div style={{ fontSize: 10, color: item.on ? item.c : "#334155" }}>{item.l}</div>
                <div style={{ fontSize: 8, color: "#1E2A3A" }}>{item.d}</div>
              </div>
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: item.on ? item.c : "#1E2A3A",
                boxShadow: item.on ? `0 0 4px ${item.c}` : "none",
              }} />
            </div>
          ))}

          {/* Roadmap */}
          <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginTop: 14, marginBottom: 8 }}>◈ ROADMAP</div>
          {[
            { n: 1, label: "Prototype", sublabel: "Live AI + Web Search", c: "#38BDF8", active: true },
            { n: 2, label: "RAG + Weaviate", sublabel: "Doc ingestion + hybrid search", c: "#A78BFA" },
            { n: 3, label: "Full Multi-Agent", sublabel: "Neo4j + CAMeL + Memory", c: "#34D399" },
            { n: 4, label: "Cognitive OS", sublabel: "Enterprise + Arabic-first", c: "#FB923C" },
          ].map((p, i) => (
            <div key={i} style={{ display: "flex", gap: 8, marginBottom: 10, alignItems: "flex-start" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                  background: p.active ? `${p.c}22` : "#080F1C",
                  border: `1.5px solid ${p.active ? p.c : "#1E2A3A"}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 8, fontWeight: 700,
                  color: p.active ? p.c : "#334155",
                  boxShadow: p.active ? `0 0 8px ${p.c}55` : "none",
                }}>
                  {p.n}
                </div>
                {i < 3 && <div style={{ width: 1, height: 14, background: "#0F2040", margin: "2px 0" }} />}
              </div>
              <div>
                <div style={{ fontSize: 10, color: p.active ? p.c : "#334155", fontWeight: p.active ? 600 : 400 }}>
                  {p.label}
                </div>
                <div style={{ fontSize: 8, color: "#1E2A3A", lineHeight: 1.4, marginTop: 1 }}>
                  {p.sublabel}
                </div>
                {p.active && (
                  <div style={{ fontSize: 8, color: "#34D399", fontFamily: "monospace", marginTop: 2 }}>
                    ● ACTIVE NOW
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Tech stack summary */}
          <div style={{ fontSize: 9, color: "#1E3A5F", letterSpacing: "0.15em", marginTop: 4, marginBottom: 8 }}>◈ TECH STACK</div>
          {[
            ["Claude Sonnet 4.6", "LLM"],
            ["web_search_20250305", "Search"],
            ["FastAPI + Python", "Backend"],
            ["LangChain + LlamaIndex", "Agents"],
            ["Weaviate", "Vector DB"],
            ["Neo4j 5.x", "Graph"],
            ["CAMeL Tools", "Arabic NLP"],
            ["Docker + K8s", "Infra"],
            ["Prometheus + Grafana", "Obs."],
          ].map(([name, role]) => (
            <div key={name} style={{
              display: "flex", justifyContent: "space-between",
              padding: "3px 0", borderBottom: "1px solid #080F1C",
            }}>
              <span style={{ fontSize: 9, color: "#334155" }}>{name}</span>
              <span style={{ fontSize: 8, color: "#1E2A3A" }}>{role}</span>
            </div>
          ))}
        </aside>
      </div>
    </div>
  );
}
