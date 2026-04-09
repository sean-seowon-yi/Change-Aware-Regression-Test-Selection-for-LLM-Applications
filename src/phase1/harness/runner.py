from __future__ import annotations

import asyncio
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from litellm import acompletion

from src.phase1.harness.cache import EvalCache
from src.phase1.harness.monitors import run_monitor
from src.phase1.models import EvalResult, TestCase

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"

_MODEL_COST_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def load_settings() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_eval_suite(domain: str) -> list[TestCase]:
    """Load test cases from data/{domain}/eval_suite.yaml."""
    suite_path = _PROJECT_ROOT / "data" / domain / "eval_suite.yaml"
    with open(suite_path) as f:
        raw = yaml.safe_load(f)
    return [
        TestCase(
            test_id=t["test_id"],
            input_text=t["input_text"],
            monitor_type=t["monitor_type"],
            monitor_config=t.get("monitor_config", {}),
            tags=t.get("tags", []),
            sensitive_to=t.get("sensitive_to", []),
        )
        for t in raw["tests"]
    ]


async def run_test(
    prompt_text: str,
    test_case: TestCase,
    *,
    version_id: str = "base",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 1024,
    num_retries: int = 5,
    cache: EvalCache | None = None,
) -> EvalResult:
    """Run a single test case against a prompt and return an EvalResult.

    Rate-limit retry handling is delegated to litellm's built-in logic.
    """
    if cache is not None:
        cached_result = cache.get(prompt_text, test_case.input_text, model, temperature, max_tokens)
        if cached_result is not None:
            cached_result.cached = True
            cached_result.version_id = version_id
            return cached_result

    start = time.perf_counter()

    try:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": test_case.input_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=num_retries,
            timeout=120,
        )
        raw_output = response.choices[0].message.content or ""
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        input_cpm, output_cpm = _MODEL_COST_PER_M.get(model, (0.15, 0.60))
        cost_usd = (prompt_tokens * input_cpm + completion_tokens * output_cpm) / 1_000_000
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return EvalResult(
            version_id=version_id,
            test_id=test_case.test_id,
            outcome="fail",
            raw_output=f"ERROR: {exc}",
            monitor_details={"error": str(exc)},
            latency_ms=elapsed,
            cost_usd=0.0,
            cached=False,
        )

    elapsed = (time.perf_counter() - start) * 1000

    monitor_result = run_monitor(raw_output, test_case.monitor_type, test_case.monitor_config)

    result = EvalResult(
        version_id=version_id,
        test_id=test_case.test_id,
        outcome="pass" if monitor_result.passed else "fail",
        raw_output=raw_output,
        monitor_details=monitor_result.details,
        latency_ms=round(elapsed, 1),
        cost_usd=cost_usd,
        cached=False,
    )

    if cache is not None and result.cost_usd > 0:
        cache.put(prompt_text, test_case.input_text, model, temperature, result, max_tokens)

    return result


async def run_suite(
    prompt_text: str,
    test_cases: list[TestCase],
    *,
    version_id: str = "base",
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    max_tokens: int = 1024,
    max_concurrent: int = 3,
    num_retries: int = 5,
    cache: EvalCache | None = None,
) -> list[EvalResult]:
    """Run all test cases concurrently (bounded by semaphore)."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded(tc: TestCase) -> EvalResult:
        async with sem:
            return await run_test(
                prompt_text,
                tc,
                version_id=version_id,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                num_retries=num_retries,
                cache=cache,
            )

    tasks = [_bounded(tc) for tc in test_cases]
    return await asyncio.gather(*tasks)
