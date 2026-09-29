"""LLM backends behind one small interface.

Modes (env ``LLM_MODE``, default ``"mock"``):
- ``mock``   — deterministic, zero-dependency responses. Used by tests, the
  demo page, and anyone without API keys.
- ``openai`` — OpenAI chat completions via httpx. Needs ``OPENAI_API_KEY``.
- ``ollama`` — local Ollama server. Needs ``OLLAMA_MODEL`` (default ``llama3.1``).

Agent prompts are tagged (``[SUPERVISOR]``, ``[ANALYST]``, ``[CRITIC]``,
``[FINAL]``) so the mock backend can return structurally valid, deterministic
outputs. Real backends ignore the tags and just answer the prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re


class LLMError(RuntimeError):
    """Raised for bad config (unknown mode, missing key, HTTP failure)."""


class BaseLLM:
    """Minimal interface the agents rely on."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def stream(self, prompt: str):
        """Yield token-ish string chunks. Default impl chunks ``generate()``."""
        text = self.generate(prompt)
        for m in re.finditer(r"\S+\s*", text):
            yield m.group(0)


_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "doing", "done", "with", "from", "into", "about", "that",
    "this", "these", "those", "and", "the", "for", "are", "was",
    "were", "will", "would", "should", "could", "there", "their",
}


def _split_tag(prompt: str) -> tuple[str, str]:
    m = re.match(r"\s*\[([A-Z]+)\]\s*(.*)$", prompt, re.DOTALL)
    if not m:
        return "GENERIC", prompt
    return m.group(1), m.group(2)


def _keywords(text: str, n: int = 4) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) == n:
            break
    return out or ["general"]


def _mock_supervisor(body: str) -> str:
    question = body.splitlines()[0].replace("QUESTION:", "").strip() or body.strip().splitlines()[0]
    kws = _keywords(question, 4)
    q1 = " ".join(kws[:2])
    q2 = " ".join(kws[2:]) if len(kws) > 2 else kws[0]
    return (
        "SUBQUERIES:\n"
        f"1. {q1} latest news and developments\n"
        f"2. {q2} analysis and outlook"
    )


def _parse_context(body: str) -> list[tuple[str, str]]:
    """Parse ``[[n]] title — snippet`` lines from an analyst/critic prompt."""
    items = []
    for line in body.splitlines():
        m = re.match(r"\s*\[\[(\d+)\]\]\s*(.+)$", line)
        if m:
            items.append((m.group(1), m.group(2).strip()))
    return items


def _first_sentence(text: str) -> str:
    m = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    s = m[0].strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def _mock_analyst(body: str) -> str:
    ctx = _parse_context(body)
    feedback = ""
    m = re.search(r"FEEDBACK:\s*(.+?)(?:\n[A-Z]+:|\Z)", body, re.DOTALL)
    if m:
        feedback = m.group(1).strip().splitlines()[0][:160]
    lines = []
    for i, (cid, text) in enumerate(ctx[:3]):
        # strip a leading "title — " if present
        snippet = text.split("—", 1)[-1].strip() if "—" in text else text
        bullet = f"- {_first_sentence(snippet)} [{cid}]"
        if i == 0 and feedback:
            bullet = f"- Revised per critique ({feedback}): {_first_sentence(snippet)} [{cid}]"
        lines.append(bullet)
    if not lines:
        lines = ["- No evidence was retrieved for this question."]
    return "\n".join(lines)


def _mock_critic(body: str) -> str:
    m = re.search(r"VALID_IDS:\s*([\d,\s]+)", body)
    valid = set(re.findall(r"\d+", m.group(1))) if m else set()
    draft = re.search(r"DRAFT:\s*(.+?)(?:\nVALID_IDS:|\Z)", body, re.DOTALL)
    bullets = [ln for ln in (draft.group(1) if draft else body).splitlines()
               if ln.strip().startswith("-")]
    bad = []
    for idx, b in enumerate(bullets, 1):
        cites = re.findall(r"\[(\d+)\]", b)
        if not cites or (valid and not any(c in valid for c in cites)):
            bad.append(idx)
    if not bad:
        return "VERDICT: PASS"
    return "VERDICT: FAIL\nREASON: bullet(s) {} lack a valid citation like [n]".format(
        ", ".join(map(str, bad)))


def _mock_final(body: str) -> str:
    draft = re.search(r"DRAFT:\s*(.+?)(?:\nCONTEXT:|\Z)", body, re.DOTALL)
    text = (draft.group(1).strip() if draft else body.strip().splitlines()[0])
    return text + "\n\n—\nBriefing complete."


class MockLLM(BaseLLM):
    """Deterministic stand-in. Same prompt -> same output, no network, no keys."""

    def generate(self, prompt: str) -> str:
        tag, body = _split_tag(prompt)
        if tag == "SUPERVISOR":
            return _mock_supervisor(body)
        if tag == "ANALYST":
            return _mock_analyst(body)
        if tag == "CRITIC":
            return _mock_critic(body)
        if tag == "FINAL":
            return _mock_final(body)
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        first = body.strip().split("\n")[0][:160]
        return f"[mock:{h}] {first}"


class OpenAILLM(BaseLLM):
    """OpenAI chat completions backend (httpx, no SDK needed)."""

    def __init__(self) -> None:
        import httpx  # local import: only needed for this backend

        self._httpx = httpx
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set (LLM_MODE=openai)")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate(self, prompt: str) -> str:
        r = self._httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        if r.status_code >= 400:
            raise LLMError(f"OpenAI error {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

    def stream(self, prompt: str):
        with self._httpx.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": True},
            timeout=180,
        ) as r:
            if r.status_code >= 400:
                raise LLMError(f"OpenAI error {r.status_code}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                except (KeyError, ValueError):
                    continue
                if delta:
                    yield delta


class OllamaLLM(BaseLLM):
    """Local Ollama backend (httpx)."""

    def __init__(self) -> None:
        import httpx  # local import: only needed for this backend

        self._httpx = httpx
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def generate(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str):
        try:
            with self._httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
                timeout=300,
            ) as r:
                if r.status_code >= 400:
                    raise LLMError(f"Ollama error {r.status_code}: is the server running?")
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)["response"]
                    except (KeyError, ValueError):
                        continue
                    if chunk:
                        yield chunk
        except self._httpx.ConnectError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {e}")


def get_llm() -> BaseLLM:
    """Build the backend selected by ``LLM_MODE`` (default ``mock``)."""
    mode = os.getenv("LLM_MODE", "mock").strip().lower()
    if mode == "mock":
        return MockLLM()
    if mode == "openai":
        return OpenAILLM()
    if mode == "ollama":
        return OllamaLLM()
    raise LLMError(f"Unknown LLM_MODE={mode!r} (expected mock|openai|ollama)")
