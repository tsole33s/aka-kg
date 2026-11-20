# llm_environment.py
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class LLMEnvironment:
    """Frozen LLM wrapper. Caches by (model, prompt, temperature) and retries
    transient failures. Exposes a single `query(prompt) -> str` interface.

    Parameters
    ----------
    model : str
        litellm-style model string, e.g. "gpt-4o-mini", "claude-sonnet-4-20250514",
        "ollama/llama3.2".
    cache_dir : Path
        Disk cache location. Set to None to disable caching (not recommended
        for bandit attribution).
    temperature : float
        Held constant for the entire run. Varying it mid-run destroys reward
        comparability across strategies.
    max_retries : int
        Transient failures retried with exponential backoff.
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path | str | None = ".aka_cache",
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_base_wait: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_wait = retry_base_wait

        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Provider-neutral completion callable. Default uses litellm; swap
        # for your own if you don't want the dependency.
        self._complete: Callable[[str], str] = self._default_complete

    def _default_complete(self, prompt: str) -> str:
        import litellm
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    def set_completion_fn(self, fn: Callable[[str], str]) -> None:
        """Inject a custom completion function — useful for testing or for
        providers litellm doesn't cover."""
        self._complete = fn

    # -- cache --------------------------------------------------------------

    def _cache_key(self, prompt: str) -> str:
        content = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if not self._cache_dir:
            return None
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path and path.exists():
            try:
                return json.loads(path.read_text())["response"]
            except (json.JSONDecodeError, KeyError, OSError):
                return None
        return None

    def _write_cache(self, key: str, response: str) -> None:
        path = self._cache_path(key)
        if not path:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"response": response, "ts": time.time()}))
        tmp.replace(path)  # atomic

    # -- public API ---------------------------------------------------------

    def query(self, prompt: str) -> str:
        """Cached, retried completion. This is the only method the loop calls."""
        key = self._cache_key(prompt)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._complete(prompt)
                self._write_cache(key, response)
                return response
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    wait = self.retry_base_wait * (2 ** attempt)
                    logger.warning(f"LLM call failed ({e}), retrying in {wait}s")
                    time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_err