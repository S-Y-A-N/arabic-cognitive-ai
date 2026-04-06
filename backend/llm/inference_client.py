"""
ACAI v4 — Unified LLM Inference Client
========================================
Single interface for Ollama (local), vLLM (GPU), and Anthropic (cloud).
Handles fallback chain automatically — if Ollama is down, tries Anthropic.
Supports streaming, tool-use, and async generation.

Usage:
    client = LLMClient()
    response = await client.generate("What is AI?", model="qwen2.5:72b")
    
    # With streaming
    async for chunk in client.stream("Explain GCC banking regulations"):
        print(chunk, end="")

    # With tools (web search)
    result = await client.generate_with_tools(prompt, tools=[WEB_SEARCH_TOOL])
"""

import asyncio
import logging
import time
import json
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx

from model_config import get_config, ANTHROPIC_API_KEY, ANTHROPIC_MODEL

logger = logging.getLogger("llm_client")
config = get_config()


class LLMResponse:
    def __init__(self, text: str, model: str, tokens: int = 0,
                 tool_calls: List = None, latency_ms: float = 0):
        self.text = text
        self.model = model
        self.tokens = tokens
        self.tool_calls = tool_calls or []
        self.latency_ms = latency_ms
        self.has_tool_calls = len(self.tool_calls) > 0


class LLMClient:
    """
    Unified client for all LLM backends.
    Automatically falls back: Ollama → vLLM → Anthropic
    """

    def __init__(self):
        self.config = get_config()
        self._http = httpx.AsyncClient(timeout=120.0)
        logger.info(f"LLMClient initialized | backend={self.config.backend} | "
                    f"primary={self.config.primary}")

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        messages: List[Dict] = None,
    ) -> LLMResponse:
        """Generate a response. Auto-selects backend and falls back on failure."""
        model = model or self.config.primary
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        t0 = time.time()

        # Build message list
        msgs = messages or []
        if not messages:
            if system:
                msgs = [{"role": "system", "content": system},
                        {"role": "user",   "content": prompt}]
            else:
                msgs = [{"role": "user", "content": prompt}]

        for backend in self._get_fallback_chain():
            try:
                result = await self._dispatch(backend, model, msgs, temperature, max_tokens)
                result.latency_ms = (time.time() - t0) * 1000
                logger.info(f"LLM response | backend={backend} model={model} "
                            f"tokens={result.tokens} latency={result.latency_ms:.0f}ms")
                return result
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {e} — trying next")

        raise RuntimeError("All LLM backends failed")

    async def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = None,
        temperature: float = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they are generated."""
        model = model or self.config.primary
        temperature = temperature if temperature is not None else self.config.temperature

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        backend = self.config.backend
        if backend == "ollama":
            async for chunk in self._stream_ollama(model, msgs, temperature):
                yield chunk
        elif backend == "anthropic" or (backend != "ollama" and ANTHROPIC_API_KEY):
            async for chunk in self._stream_anthropic(system, prompt, model):
                yield chunk
        else:
            # Fallback to non-streaming
            result = await self.generate(prompt, system, model, temperature)
            yield result.text

    async def generate_with_tools(
        self,
        prompt: str,
        system: str,
        tools: List[Dict],
        model: str = None,
        max_turns: int = 6,
    ) -> LLMResponse:
        """
        Multi-turn tool-use generation.
        Continues until stop_reason is 'end_turn' (not 'tool_use').
        Works with Claude API tool format.
        """
        model = model or self.config.primary
        messages = [{"role": "user", "content": prompt}]
        all_tool_calls = []
        final_text = ""

        for turn in range(max_turns):
            # For tool-use, prefer Anthropic if available (best tool-use support)
            if ANTHROPIC_API_KEY:
                response = await self._anthropic_with_tools(system, messages, tools)
            else:
                response = await self.generate(prompt=prompt, system=system, model=model)
                return response

            if response.has_tool_calls:
                all_tool_calls.extend(response.tool_calls)
                # Add assistant turn
                messages.append({"role": "assistant", "content": response.text})
                # Tool results would be added by orchestrator
                final_text = response.text
            else:
                final_text = response.text
                break

        return LLMResponse(text=final_text, model=model, tool_calls=all_tool_calls)

    # ─── Backend Dispatch ─────────────────────────────────────────────────────

    async def _dispatch(self, backend: str, model: str, messages: List[Dict],
                        temperature: float, max_tokens: int) -> LLMResponse:
        if backend == "ollama":
            return await self._call_ollama(model, messages, temperature, max_tokens)
        elif backend == "vllm":
            return await self._call_vllm(model, messages, temperature, max_tokens)
        elif backend == "anthropic":
            return await self._call_anthropic(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ─── Ollama ───────────────────────────────────────────────────────────────

    async def _call_ollama(self, model: str, messages: List[Dict],
                           temperature: float, max_tokens: int) -> LLMResponse:
        """
        Call Ollama local server.
        Install Ollama: https://ollama.ai
        Pull model:     ollama pull qwen2.5:72b
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": min(self.config.context_window, 32768),
            }
        }
        response = await self._http.post(
            f"{self.config.ollama_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0)
        return LLMResponse(text=text, model=model, tokens=tokens)

    async def _stream_ollama(self, model: str, messages: List[Dict],
                             temperature: float) -> AsyncGenerator[str, None]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature}
        }
        async with self._http.stream(
            "POST", f"{self.config.ollama_url}/api/chat", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        pass

    # ─── vLLM ─────────────────────────────────────────────────────────────────

    async def _call_vllm(self, model: str, messages: List[Dict],
                         temperature: float, max_tokens: int) -> LLMResponse:
        """
        Call vLLM OpenAI-compatible server.
        Start:  vllm serve Qwen/Qwen2.5-72B-Instruct --port 8080 --gpu-memory-utilization 0.9
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = await self._http.post(
            f"{self.config.vllm_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer dummy"}  # vLLM doesn't need key
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return LLMResponse(text=text, model=model, tokens=tokens)

    # ─── Anthropic ────────────────────────────────────────────────────────────

    async def _call_anthropic(self, messages: List[Dict],
                              temperature: float, max_tokens: int) -> LLMResponse:
        """Anthropic API fallback — only used if local models fail."""
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("No ANTHROPIC_API_KEY set")

        # Separate system message
        system = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                api_messages.append(m)

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system:
            payload["system"] = system

        response = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        tokens = data.get("usage", {}).get("output_tokens", 0)
        return LLMResponse(text=text, model=ANTHROPIC_MODEL, tokens=tokens)

    async def _anthropic_with_tools(self, system: str, messages: List[Dict],
                                     tools: List[Dict]) -> LLMResponse:
        """Anthropic with tool_use — best tool-calling support."""
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("No ANTHROPIC_API_KEY")

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        response = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        )
        response.raise_for_status()
        data = response.json()

        text_parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        tool_calls = [b for b in data.get("content", []) if b.get("type") == "tool_use"]

        return LLMResponse(
            text="\n".join(text_parts),
            model=ANTHROPIC_MODEL,
            tool_calls=tool_calls,
            tokens=data.get("usage", {}).get("output_tokens", 0)
        )

    async def _stream_anthropic(self, system: str, prompt: str,
                                model: str) -> AsyncGenerator[str, None]:
        """Stream from Anthropic API."""
        if not ANTHROPIC_API_KEY:
            yield "Error: no Anthropic API key"
            return

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with self._http.stream(
            "POST", "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "anthropic-beta": "messages-2023-06-01",
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            chunk = data.get("delta", {}).get("text", "")
                            if chunk:
                                yield chunk
                    except json.JSONDecodeError:
                        pass

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_fallback_chain(self) -> List[str]:
        """Get the fallback chain starting from the configured backend."""
        chain = [self.config.backend]
        if self.config.fallback_to_anthropic and "anthropic" not in chain:
            chain.append("anthropic")
        return chain

    async def health_check(self) -> Dict:
        """Check which backends are available."""
        status = {}
        # Check Ollama
        try:
            r = await self._http.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            status["ollama"] = {"ok": True, "models": models}
        except Exception as e:
            status["ollama"] = {"ok": False, "error": str(e)}
        # Check Anthropic
        status["anthropic"] = {"ok": bool(ANTHROPIC_API_KEY), "key_set": bool(ANTHROPIC_API_KEY)}
        return status

    async def list_ollama_models(self) -> List[str]:
        """List models available in local Ollama instance."""
        try:
            r = await self._http.get(f"{self.config.ollama_url}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    async def close(self):
        await self._http.aclose()
