from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import diskcache

from src.phase1.models import EvalResult


class EvalCache:
    """Deterministic result cache backed by diskcache (SQLite)."""

    def __init__(self, cache_dir: str = "./cache/eval_cache"):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir)

    @staticmethod
    def _make_key(
        prompt: str, input_text: str, model: str, temperature: float,
        max_tokens: int = 1024,
    ) -> str:
        raw = (
            f"{prompt}\n---INPUT---\n{input_text}\n---MODEL---\n{model}"
            f"\n---TEMP---\n{temperature}\n---MAXTOK---\n{max_tokens}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self, prompt: str, input_text: str, model: str, temperature: float,
        max_tokens: int = 1024,
    ) -> EvalResult | None:
        key = self._make_key(prompt, input_text, model, temperature, max_tokens)
        data = self._cache.get(key)
        if data is None:
            return None
        return EvalResult(**json.loads(data))

    def put(
        self, prompt: str, input_text: str, model: str, temperature: float,
        result: EvalResult, max_tokens: int = 1024,
    ) -> None:
        key = self._make_key(prompt, input_text, model, temperature, max_tokens)
        self._cache.set(key, json.dumps(asdict(result)))

    def close(self) -> None:
        self._cache.close()
