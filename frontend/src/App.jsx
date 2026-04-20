import { useState, useEffect } from "react";
import { AGENTS } from "./data/agents";
import { ChatPanel } from "./components/ui/chat-panel";
import { DARK, LIGHT } from "./data/theme";
import { BACKEND } from "./data/backend";

export default function App() {
  const [active, setActive] = useState("auto");
  const [theme, setTheme] = useState("dark");
  const [sidebar, setSidebar] = useState(true);
  const [health, setHealth] = useState(null);
  const t = theme === "dark" ? DARK : LIGHT;
  const ag = AGENTS.find(a => a.id === active);

  useEffect(() => {
    const check = () =>
      fetch(`${BACKEND}/api/health`)
        .then(r => r.json()).then(() => setHealth(true))
        .catch(() => setHealth(false));
    check();
    const iv = setInterval(check, 30000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="container" style={{ background: t.bg, color: t.text }}>
      {/* HEADER */}
      <header style={{ background: t.panel, borderBottom: `1px solid ${t.border}` }}>

        {/* Sidebar toggle button */}
        <button className="sidebar-toggle" onClick={() => setSidebar(o => !o)}
          style={{ color: t.muted }}
          onMouseOver={e => e.currentTarget.style.color = t.text}
          onMouseOut={e => e.currentTarget.style.color = t.muted}>
          {sidebar ? "▶ أغلق" : "◀ افتح"}
        </button>

        <div className="logo">
          <div className="logo-image">
            <svg fill={theme === "dark" ? "white" : "black"} height="40px" width="40px" version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 325 325" xml:space="preserve">
              <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
              <g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g>
              <g id="SVGRepo_iconCarrier"> <g>
                <path d="M324.268,160.718l-46.131-46.13V49.349c0-1.381-1.119-2.5-2.5-2.5h-65.24L164.268,0.718c-0.938-0.938-2.598-0.938-3.535,0 l-46.129,46.131H49.361c-1.381,0-2.5,1.119-2.5,2.5v65.239l-46.129,46.13c-0.976,0.977-0.976,2.559,0,3.535l46.129,46.13v65.239 c0,1.381,1.119,2.5,2.5,2.5h65.242l46.129,46.131c0.469,0.469,1.104,0.732,1.768,0.732c0.663,0,1.299-0.264,1.768-0.732 l46.129-46.131h65.24c1.381,0,2.5-1.119,2.5-2.5v-65.239l46.131-46.13c0.469-0.469,0.732-1.104,0.732-1.768 C325,161.822,324.737,161.186,324.268,160.718z M242.17,195.485v46.668h-46.672l-32.998,33l-32.998-33H82.828v-46.668l-32.996-33 l32.996-33V82.817h46.674l32.998-33l32.998,33h46.672v46.668l32.998,33L242.17,195.485z"></path>
                <path d="M228.893,96.097h-38.896L162.5,68.596l-27.498,27.501H96.107v38.889l-27.498,27.5l27.498,27.5v38.889h38.895l27.498,27.501 l27.496-27.501h38.896v-38.889l27.498-27.5l-27.498-27.5V96.097z M210.228,174.692h-18.262l12.912,12.91l0.664,17.926 l-17.926-0.664l-12.912-12.911v18.259L162.5,223.356l-12.205-13.145v-18.261l-12.912,12.913l-17.926,0.664l0.664-17.926 l12.912-12.912l-18.26,0.002l-13.144-12.207l13.144-12.205l18.26-0.002l-12.912-12.91l-0.664-17.926l17.926,0.664l12.912,12.912 v-18.26l12.205-13.145l12.205,13.145v18.26l12.912-12.912l17.926-0.664l-0.664,17.926l-12.912,12.912h18.262l13.141,12.205 L210.228,174.692z"></path>
                <polygon points="150.768,134.162 134.174,150.755 134.174,174.217 150.766,190.809 174.232,190.809 190.822,174.219 190.822,150.753 174.232,134.162 "></polygon> </g> </g>
            </svg>
          </div>
          <div className="logo-text">
            <div className="logo-top-text" style={{ color: t.muted }}>
              Arabic Cognitive AI
            </div>
            <div className="logo-btm-text" style={{ color: t.faint }}>
              الذكاء المعرفي العربي الاصطناعي
            </div>
          </div>
        </div>

        <div className="header-right">
          {/* Dark/Light toggle */}
          <button className="theme-toggle" onClick={() => setTheme(th => th === "dark" ? "light" : "dark")}
            style={{
              background: theme === "dark" ? "#1e3a5f" : "#e2e8f0",
              border: `1px solid ${t.border}`
            }}>
            <div className="theme-icon" style={{
              left: theme === "dark" ? 3 : 19,
              background: theme === "dark" ? "#4575cf" : "#ce871e",
            }}>
              {theme === "dark" ? "🌙" : "☀️"}
            </div>
          </button>
        </div>
      </header>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* SIDEBAR */}
        <aside className="sidebar" style={{
          width: sidebar ? 300 : 0,
          background: t.panel,
          borderLeft: sidebar ? `1px solid ${t.border}` : "none"
        }}>
          <div className="sidebar-content">
            <p className="sidebar-title">
              عملاء المعرفة
            </p>
            {AGENTS.map(a => {
              const on = a.id === active;
              return (
                <button className="agent-btn" key={a.id} onClick={() => setActive(a.id)}
                  style={{
                    background: on ? `${a.hex}0f` : "transparent",
                    border: on ? `1px solid ${a.hex}33` : "1px solid transparent",
                    borderLeft: `3px solid ${on ? a.hex : "transparent"}`,
                    boxShadow: on ? `inset 0 0 20px ${a.glow}` : "none"
                  }}
                  onMouseOver={e => {
                    if (!on) {
                      e.currentTarget.style.background = `${a.hex}07`;
                      e.currentTarget.style.borderLeftColor = `${a.hex}55`;
                    }
                  }}
                  onMouseOut={e => {
                    if (!on) {
                      e.currentTarget.style.background = "transparent";
                      e.currentTarget.style.borderLeftColor = "transparent";
                    }
                  }}>
                  {/* Agent Icon */}
                  <div className="agent-icon" style={{
                    background: `${a.hex}${on ? "16" : "0a"}`,
                    border: `1px solid ${a.hex}${on ? "44" : "22"}`,
                    color: on ? a.hex : `${a.hex}88`,
                    fontSize: a.id === "lughawi" ? 17 : 14,
                    fontFamily: a.id === "lughawi" ? "'Scheherazade New',serif" : "inherit",
                    boxShadow: on ? `0 0 9px ${a.glow}` : "none",
                  }}>
                    <span>{a.icon}</span>
                  </div>
                  {/* Agent Titles: AR, EN, DESC */}
                  <div className="agent-title-box">
                    <div className="h-flex s-gap baseline">
                      <span className="agent-title-ar" style={{ fontWeight: on ? 700 : 500, color: on ? a.hex : t.muted, }}>
                        {a.ar}
                      </span>
                      <span className="agent-title-en" style={{ color: on ? `${a.hex}99` : t.faint }}>
                        {a.en}
                      </span>
                    </div>
                    <div className="agent-title-desc" style={{ color: t.faint }}>
                      {a.title}
                    </div>
                  </div>
                </button>
              );
            })}

            <div className="sidebar-section" style={{
              borderTop: `1px solid ${t.border}`
            }}>
              {/* Health */}
              <div className="h-flex s-gap">
                <div className="glow-circle" style={{
                  background: health === null ? "#f59e0b" : health ? "#22c55e" : "#ef4444",
                  animation: health === null ? "acai-pulse 1.5s infinite" : "none",
                  boxShadow: `0 0 5px ${health ? "#22c55e" : "#ef4444"}`
                }} />
                <span style={{ color: t.muted }}>
                  الخادم: {health === null ? "..." : health ? "متوفر" : "غير متوفر"}
                </span>
              </div>
            </div>
          </div>
        </aside>

        {/* MAIN: Chat Panel */}
        <main>
          <ChatPanel key={active} ag={ag} theme={theme} />
        </main>
      </div>
    </div>
  );
}
