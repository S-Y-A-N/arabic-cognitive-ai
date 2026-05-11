import { BACKEND } from "../../../data/backend";

const headers = {
  "Content-Type": "application/json",
};

export async function callBackend(query, mode, agentId, onChunk, onSearch, onDone, onError) {

  try {
    // Announce which agents will run (via health check intent)
    onSearch("⚙️ " + mode.replace("single:", ""));

    const response = await fetch(`${BACKEND}/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, mode, session_id: agentId }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error);
    }
    
    while (true) {
      const { value, done } = await reader.read();
      // console.log(value)
      if (done) break;

      // add chunks to buffer
      const chunk = decoder.decode(value, { stream: true });
      onChunk(chunk)

      // seperate by new line and remove `\n`
      // const lines = buffer.split("\n");
      // buffer = lines.pop();
      // show stream by new lines
      // for (const line of lines) {
        // console.log(line)
        // if (!line.startsWith("data: ")) continue;
        // try {
        //   const d = JSON.parse(line.slice(6));
        //   if (d.type === "chunk" && d.text) onChunk(d.text);
        //   if (d.type === "done") {
        //     if (d.pipeline?.length > 1) {
        //       d.pipeline.forEach(a => onSearch(`✓ ${a}`));
        //     }
        //     onDone(d);
        //     return;
        //   }
        //   if (d.type === "error") { onError(d.error); return; }
        // } catch (e) {
        //   onError(e.message)
        // }
      // }
    }
    onDone({});
  } catch (e) {
    onError(e.message);
  }
}