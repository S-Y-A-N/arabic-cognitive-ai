import { useState, useRef, useEffect, useCallback } from "react";
import { Bubble } from "./message-bubble";
import { DARK, LIGHT } from "../../data/theme";
import { isAr } from "../../utils/arabic";
import { callBackend } from "../../features/chats/api/get-response";

export const ChatPanel = ({ ag, theme }) => {
  const t    = theme === "dark" ? DARK : LIGHT;
  const [msgs, setMsgs] = useState([]);
  const [inp,  setInp]  = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);
  const taRef  = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);

  const send = useCallback(async (text) => {
    const q = (text || inp).trim();
    if (!q || busy) return;
    setInp(""); if (taRef.current) taRef.current.style.height = "auto";
    setBusy(true);

    const uid = Date.now(), aid = uid + 1;
    setMsgs(p => [...p,
      { id:uid, role:"user", content:q },
      { id:aid, role:"assistant", content:"", streaming:true, searches:[] }
    ]);
    const t0 = Date.now();
    const upd = fn => setMsgs(p => p.map(m => m.id === aid ? fn(m) : m));

    await callBackend(
      q, ag.mode, ag.id,
      chunk  => upd(m => ({ ...m, content: m.content + chunk })),
      q2     => upd(m => ({ ...m, searches: [...m.searches, { q:q2, done:false }] })),
      () => upd(m => ({
        ...m, streaming:false,
        latency: Date.now() - t0,
        searches: m.searches.map(s => ({ ...s, done:true }))
      })),
      err    => upd(m => ({
        ...m,
        content: `خطأ في الاتصال:\n${err}\n\nتأكد من تشغيل:\ncd backend && uvicorn main_v5:app --port 8000`,
        streaming:false, error:true
      })),
    );
    setBusy(false);
  }, [inp, busy, ag]);

  const onKey  = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  const resize = e => {
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
    setInp(e.target.value);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column",
      height:"100%", overflow:"hidden" }}>

      {/* Agent header */}
      <div style={{ padding:"0.7rem 2rem 1rem", borderBottom:`1px solid ${t.border}`,
        background:t.panel }}>
        <div className="h-flex m-gap">
          <div className="agent-icon medium" style={{
            background:`${ag.hex}12`, border:`2px solid ${ag.hex}44`,
            fontSize: ag.id ==="lughawi" ? 26 : ag.id === "auto" ? 24 : 21,
            color:ag.hex,
            boxShadow:`0 0 24px ${ag.glow}` }}>
            <span>{ag.icon}</span>
          </div>
          <div>
            <div className="h-flex s-gap baseline">
              <h2 style={{
                color:ag.hex, fontFamily:"'Scheherazade New',serif" }}>{ag.ar}</h2>
              <span style={{ fontSize:13, color:t.muted }}>{ag.en}</span>
              {/* {ag.id === "auto" && (
                <span style={{ fontSize:9, padding:"2px 8px", borderRadius:20,
                  background:"#f59e0b18", color:"#f59e0b",
                  border:"1px solid #f59e0b33", fontWeight:700,
                  letterSpacing:".1em", animation:"acai-pulse 2s infinite" }}>
                  ⚡ SMART ROUTE
                </span>
              )} */}
            </div>
            <p style={{ margin:0, fontSize:12, color:t.muted,}}>{ag.title}</p>
          </div>
          {/* <div style={{ marginLeft:"auto", padding:"5px 14px", borderRadius:20,
            background:`${ag.hex}12`, border:`1px solid ${ag.hex}33`,
            fontSize:10, color:ag.hex, fontWeight:700,
            letterSpacing:".12em" }}>{ag.badge}</div> */}
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex:1, overflowY:"auto",
        padding:"24px 28px", background:t.bg }}>
        {msgs.length === 0 && (
          <div style={{ textAlign:"center", padding:"40px 16px",
            animation:"acai-up .5s ease" }}>
            <div className="agent-icon large" style={{
              margin:"0 auto 20px",
              background:`${ag.hex}0e`, border:`2px solid ${ag.hex}2a`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:40, color:ag.hex, fontWeight:900,
              fontFamily:ag.id==="lughawi"?"'Scheherazade New',serif":"inherit",
              boxShadow:`0 0 36px ${ag.glow}`,
              animation:"acai-float 5s ease-in-out infinite" }}>
              <span>{ag.icon}</span>
            </div>
            <h3 style={{ margin:"0 0 10px", fontSize:21, fontWeight:700,
              color:t.text, fontFamily:"'Scheherazade New',serif" }}>
              {ag.title}
            </h3>
            <p style={{ margin:"0 0 30px", color:t.muted, fontSize:13,
              maxWidth:420, marginLeft:"auto", marginRight:"auto",
              fontFamily:"'Scheherazade New',serif",
              direction:"rtl", lineHeight:1.9 }}>
              {ag.id === "auto"
                ? "يحلل سؤالك ويختار أفضل وكلاء تلقائياً — Researcher + GCC Advisor + Reasoner + Verifier"
                : ag.badge + " · " + ag.en}
            </p>
            <div style={{ display:"flex", flexDirection:"column",
              gap:8, maxWidth:520, margin:"0 auto" }}>
              {ag.tips.map((s, i) => (
                <button key={i} onClick={() => send(s)}
                  style={{ background:t.sub,
                    border:`1px solid ${t.border}`,
                    borderLeft:`3px solid ${ag.hex}`,
                    borderRadius:12, padding:"12px 16px",
                    color:t.muted, cursor:"pointer",
                    transition:"all .2s",
                    fontSize:isAr(s)?15:13.5,
                    fontFamily:isAr(s)?"'Scheherazade New',serif":"inherit",
                    direction:isAr(s)?"rtl":"ltr",
                    textAlign:isAr(s)?"right":"left" }}
                  onMouseOver={e=>{e.currentTarget.style.background=`${ag.hex}0a`;
                    e.currentTarget.style.color=t.text;}}
                  onMouseOut={e=>{e.currentTarget.style.background=t.sub;
                    e.currentTarget.style.color=t.muted;}}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {msgs.map(m => <Bubble key={m.id} msg={m} ag={ag} t={t}/>)}
        <div ref={endRef}/>
      </div>

      {/* Input */}
      <div style={{ padding:"14px 28px 20px",
        borderTop:`1px solid ${t.border}`,
        background:t.panel, flexShrink:0 }}>
        <div style={{ display:"flex", gap:10, alignItems:"flex-end",
          background:t.input,
          border:`1.5px solid ${busy?t.border:ag.hex+"55"}`,
          borderRadius:16, padding:"12px 14px",
          transition:"all .25s",
          boxShadow:busy?"none":`0 0 20px ${ag.glow}` }}>
          <textarea ref={taRef} value={inp} onChange={resize}
            onKeyDown={onKey} placeholder={ag.hint}
            rows={1} disabled={busy}
            style={{ flex:1, background:"transparent",
              border:"none", outline:"none", color:t.text,
              resize:"none", fontSize:isAr(inp)?16:14.5,
              lineHeight:1.65,
              maxHeight:160, overflowY:"auto" }}/>
          <button onClick={() => send()} disabled={!inp.trim() || busy}
            style={{ width:40, height:40, borderRadius:12, border:"none",
              flexShrink:0,
              background: (!inp.trim()||busy) ? t.border
                : `linear-gradient(135deg,${ag.hex},${ag.hex}bb)`,
              color: (!inp.trim()||busy) ? t.muted : "#fff",
              fontSize:18, display:"flex", alignItems:"center",
              justifyContent:"center",
              cursor:(!inp.trim()||busy)?"not-allowed":"pointer",
              transition:"all .2s",
              boxShadow:(!inp.trim()||busy)?"none":`0 4px 14px ${ag.glow}` }}>
            {busy
              ? <div style={{ width:15, height:15, borderRadius:"50%",
                  border:`2px solid ${t.muted}44`,
                  borderTopColor:t.muted,
                  animation:"acai-spin .65s linear infinite" }}/>
              : "↑"}
          </button>
        </div>
        <p style={{ margin:"8px 0 0", textAlign:"center",
          fontSize:10, color:t.faint }}>
          <b>{ag.ar}</b> نموذج ذكاء اصطناعي معرّض للخطأ. تأكد من صحة المعلومات المهمة.
        </p>
      </div>
    </div>
  );
};