import { BACKEND } from "../../../data/backend";

const API_KEY = "dev-key-12345";
const headers = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

export async function callBackend(query, mode, agentId, onChunk, onSearch, onDone, onError) {

  try {
    // Announce which agents will run (via health check intent)
    onSearch("⚙️ " + mode.replace("single:", ""));

    const r = await fetch(`${BACKEND}/api/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, mode, session_id: agentId }),
    });

    console.log(r)

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error);
    }

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    console.log(reader)
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.type === "chunk" && d.text) onChunk(d.text);
          if (d.type === "done") {
            if (d.pipeline?.length > 1) {
              d.pipeline.forEach(a => onSearch(`✓ ${a}`));
            }
            onDone(d);
            return;
          }
          if (d.type === "error") { onError(d.error); return; }
        } catch (e) {
          onError(e.message)
        }
      }
    }
    onDone({});
  } catch (e) {
    onError(e.message);
  }
}