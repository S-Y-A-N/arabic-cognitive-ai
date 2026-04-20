import { memo } from "react";
import { Chip } from "./chip";
import { isAr } from "../../utils/arabic";

export const Bubble = memo(({ msg, ag, t }) => {
  const ar = isAr(msg.content);
  if (msg.role === "user") return (
    <div style={{ display:"flex", justifyContent:"flex-end",
      marginBottom:20, animation:"acai-up .25s ease" }}>
      <div style={{ maxWidth:"74%", background:t.userBg,
        border:"1px solid #2563eb22", borderRadius:"18px 18px 4px 18px",
        padding:"13px 18px" }}>
        <p style={{ margin:0, color:"#e2e8f0",
          fontSize:ar?17:14.5, lineHeight:1.85,
          direction:ar?"rtl":"ltr", textAlign:ar?"right":"left",
          fontFamily:ar?"'Scheherazade New',serif":"inherit",
          whiteSpace:"pre-wrap" }}>{msg.content}</p>
      </div>
    </div>
  );

  const ar2 = isAr(msg.content);
  return (
    <div style={{ display:"flex", gap:12, marginBottom:24, animation:"acai-up .3s ease" }}>
      <div style={{ width:36, height:36, borderRadius:10, flexShrink:0, marginTop:3,
        background:`${ag.hex}14`, border:`1.5px solid ${ag.hex}44`,
        display:"flex", alignItems:"center", justifyContent:"center",
        color:ag.hex, fontWeight:900,
        fontSize:ag.id==="lughawi"||ag.id==="auto"?19:14,
        fontFamily:ag.id==="lughawi"?"'Scheherazade New',serif":"inherit",
        boxShadow:`0 0 14px ${ag.glow}` }}>
        {ag.icon}
      </div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:7 }}>
          <span style={{ fontSize:15, fontWeight:700, color:ag.hex,
            fontFamily:"'Scheherazade New',serif" }}>{ag.ar}</span>
          <span style={{ fontSize:8, padding:"2px 7px", borderRadius:20,
            background:`${ag.hex}18`, color:ag.hex,
            fontWeight:700, letterSpacing:".12em" }}>{ag.badge}</span>
          {msg.streaming && <div style={{ width:11, height:11, borderRadius:"50%",
            border:`1.5px solid ${ag.hex}33`, borderTopColor:ag.hex,
            animation:"acai-spin .65s linear infinite", marginLeft:2 }}/>}
          {msg.latency && !msg.streaming &&
            <span style={{ fontSize:10, color:t.muted, marginLeft:"auto" }}>
              {(msg.latency/1000).toFixed(1)}s
            </span>}
        </div>

        {msg.searches?.length > 0 && (
          <div style={{ marginBottom:8 }}>
            {msg.searches.map((s, i) => <Chip key={i} q={s.q} done={s.done}/>)}
          </div>
        )}

        <div style={{ background:t.card,
          border:`1px solid ${msg.streaming ? ag.hex+"44" : t.border}`,
          borderRadius:"4px 18px 18px 18px", padding:"16px 20px",
          boxShadow:msg.streaming?`0 0 18px ${ag.glow}`:"none",
          transition:"border-color .3s,box-shadow .3s" }}>
          {msg.content
            ? <p style={{ margin:0,
                color:msg.error?"#f87171":t.text,
                fontSize:ar2?17:14.5, lineHeight:1.95,
                whiteSpace:"pre-wrap", wordBreak:"break-word",
                direction:ar2?"rtl":"ltr", textAlign:ar2?"right":"left",
                fontFamily:ar2?"'Scheherazade New',serif"
                              :"'JetBrains Mono','Fira Code',monospace" }}>
                {msg.content}
                {msg.streaming &&
                  <span style={{ opacity:.5, animation:"acai-blink 1s infinite" }}>▌</span>}
              </p>
            : <div style={{ display:"flex", alignItems:"center", gap:9 }}>
                <div style={{ width:15, height:15, borderRadius:"50%",
                  border:`2px solid ${ag.hex}33`, borderTopColor:ag.hex,
                  animation:"acai-spin .65s linear infinite" }}/>
                <span style={{ fontSize:12, color:t.muted,
                  fontFamily:"'Scheherazade New',serif" }}>يعالج...</span>
              </div>}
        </div>

        {!msg.streaming && msg.content && !msg.error && (
          <div style={{ marginTop:6 }}>
            <button onClick={() => navigator.clipboard.writeText(msg.content)}
              style={{ background:"none", border:`1px solid ${t.border}`,
                borderRadius:6, padding:"3px 10px", cursor:"pointer",
                fontSize:11, color:t.muted, transition:"all .2s" }}
              onMouseOver={e=>{e.currentTarget.style.borderColor=ag.hex;
                e.currentTarget.style.color=ag.hex;}}
              onMouseOut={e=>{e.currentTarget.style.borderColor=t.border;
                e.currentTarget.style.color=t.muted;}}>
              ⎘ نسخ
            </button>
          </div>
        )}
      </div>
    </div>
  );
});