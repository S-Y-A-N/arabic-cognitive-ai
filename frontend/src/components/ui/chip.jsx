export const Chip = ({ q, done }) => (
  <span style={{
    display:"inline-flex", alignItems:"center", gap:5,
    padding:"3px 10px", borderRadius:20, marginRight:6, marginBottom:4,
    background: done ? "#06b6d418" : "#06b6d40e",
    border: `1px solid ${done ? "#06b6d455" : "#06b6d422"}`,
    animation: "acai-up .2s ease",
  }}>
    {!done
      ? <div style={{ width:8, height:8, borderRadius:"50%",
          border:"1.5px solid #06b6d430", borderTopColor:"#06b6d4",
          animation:"acai-spin .65s linear infinite" }}/>
      : <span style={{ fontSize:9, color:"#06b6d4" }}>✓</span>}
    <span style={{ fontSize:11, color:"#06b6d4", maxWidth:230,
      overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{q}</span>
  </span>
);