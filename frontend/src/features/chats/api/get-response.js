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

    }
    onDone({});
  } catch (e) {
    onError(e.message);
  }
}