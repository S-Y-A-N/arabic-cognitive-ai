import { BACKEND } from "../../../data/backend";

const headers = {
  "Content-Type": "application/json",
};

// TODO agentId is session id becuase one chat per agent for now
// Later add feature to create many chats with session ids
export async function callBackend(query, mode, agentId, onChunk, onSearch, onDone, onError) {
  try {
    // Announce which agents will run (via health check intent)
    onSearch("⚙️ " + mode.replace("single:", ""));

    const endpoint = mode === "auto" ? "query" : "query/stream"
    const response = await fetch(`${BACKEND}/${endpoint}`, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, mode, session_id: agentId }),
    });
    console.log(response.body)
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error);
    }

    const META_PREFIX = "__METADATA__";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      // add chunks to buffer
      const chunk = decoder.decode(value, { stream: true });

      if (chunk.includes(META_PREFIX)) {
        const [text, metaStr] = chunk.split(META_PREFIX);
        if (text) onChunk(text);
        const meta = JSON.parse(metaStr);
        if (meta.pipeline) {
          meta.pipeline.forEach(agent => onSearch(agent)); // one chip per agent
        }
        onDone(meta);
        return
      }

      onChunk(chunk)
    }
    onDone({});
  } catch (e) {
    onError(e.message);
  }
}