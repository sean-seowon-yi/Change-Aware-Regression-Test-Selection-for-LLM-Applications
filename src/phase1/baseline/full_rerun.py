"""Full-rerun baseline: run every test on every version pair, record ground truth.

For each domain this module:
1. Runs the full eval suite on the base prompt FIRST (reliable baseline).
2. Runs the full eval suite on all 50 mutated versions in parallel.
3. Compares outcomes pairwise to identify *impacted* tests (outcome changed).
4. Writes per-pair detail records and an aggregate summary.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import yaml
from tqdm.asyncio import tqdm as atqdm

from src.phase1.harness.cache import EvalCache
from src.phase1.harness.runner import load_eval_suite, load_settings, run_test
from src.phase1.models import EvalResult, TestCase


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_versions(domain: str) -> list[dict]:
    """Load all version YAML files for a domain, sorted by version_id."""
    versions_dir = _PROJECT_ROOT / "data" / domain / "versions"
    if not versions_dir.exists():
        raise FileNotFoundError(f"Versions directory not found: {versions_dir}")
    version_files = sorted(versions_dir.glob("v*.yaml"))
    versions = []
    for vf in version_files:
        with open(vf) as f:
            versions.append(yaml.safe_load(f))
    return versions


async def run_full_baseline(
    domain: str,
    *,
    base_prompt: str | None = None,
    settings: dict | None = None,
    cache: EvalCache | None = None,
) -> dict:
    """Run the full-rerun baseline for a domain.

    Phase 1: run base prompt to establish reliable ground truth.
    Phase 2: run all versions in parallel (litellm handles rate limits).
    """
    if settings is None:
        settings = load_settings()

    model = settings["model"]
    temperature = settings["temperature"]
    max_tokens = settings.get("max_tokens", 1024)
    max_concurrent = settings.get("max_concurrent_requests", 50)
    num_retries = settings.get("num_retries", 10)

    test_cases = load_eval_suite(domain)
    versions = load_versions(domain)

    if base_prompt is None:
        base_path = _PROJECT_ROOT / "data" / domain / "base_prompt.txt"
        base_prompt = base_path.read_text()

    wall_start = time.time()
    total_cost = 0.0
    total_calls = 0

    sem = asyncio.Semaphore(max_concurrent)

    async def _run(prompt: str, tc: TestCase, vid: str) -> EvalResult:
        async with sem:
            return await run_test(
                prompt, tc, version_id=vid,
                model=model, temperature=temperature, max_tokens=max_tokens,
                num_retries=num_retries, cache=cache,
            )

    # --- Phase 1: Base prompt ---
    print(f"\n[{domain}] Phase 1: base prompt × {len(test_cases)} tests (concurrency={max_concurrent})")

    base_coros = [_run(base_prompt, tc, "base") for tc in test_cases]
    base_results: list[EvalResult] = await atqdm.gather(
        *base_coros, desc="Base", unit="test",
    )

    base_outcomes = {r.test_id: r.outcome for r in base_results}
    base_pass = sum(1 for r in base_results if r.outcome == "pass")
    base_errors = sum(1 for r in base_results if r.raw_output.startswith("ERROR:"))
    base_cost = sum(r.cost_usd for r in base_results)
    total_cost += base_cost
    total_calls += len(base_results)

    print(f"[{domain}] Base: {base_pass}/{len(test_cases)} pass, "
          f"{base_errors} errors, ${base_cost:.4f}")
    if base_errors:
        print(f"[{domain}] WARNING: {base_errors} base test(s) errored")

    # --- Phase 2: All versions in parallel ---
    total_version_tasks = len(versions) * len(test_cases)
    print(f"\n[{domain}] Phase 2: {len(versions)} versions × {len(test_cases)} tests "
          f"= {total_version_tasks} calls (concurrency={max_concurrent})")

    all_version_coros: list = []
    version_offsets: list[tuple[dict, int]] = []
    for vdata in versions:
        vid = vdata["version_id"]
        prompt_text = vdata["prompt_text"]
        offset = len(all_version_coros)
        version_offsets.append((vdata, offset))
        for tc in test_cases:
            all_version_coros.append(_run(prompt_text, tc, vid))

    all_version_results: list[EvalResult] = await atqdm.gather(
        *all_version_coros, desc="Versions", unit="test",
    )

    # --- Process results ---
    detail_records: list[dict] = []
    impact_summary: dict[str, dict] = {}
    n_tests = len(test_cases)

    for vdata, offset in version_offsets:
        vid = vdata["version_id"]
        change_types = [
            f"{c['unit_type']}:{c['change_type']}"
            for c in vdata.get("changes", [])
        ]
        version_results = all_version_results[offset: offset + n_tests]

        version_outcomes = {r.test_id: r.outcome for r in version_results}
        v_cost = sum(r.cost_usd for r in version_results)
        v_errors = sum(1 for r in version_results if r.raw_output.startswith("ERROR:"))
        total_cost += v_cost
        total_calls += len(version_results)

        impacted_tests = []
        for tc in test_cases:
            tid = tc.test_id
            ob = base_outcomes.get(tid, "fail")
            ov = version_outcomes.get(tid, "fail")
            if ob != ov:
                impacted_tests.append(tid)
            detail_records.append({
                "domain": domain,
                "version_id": vid,
                "test_id": tid,
                "outcome_base": ob,
                "outcome_version": ov,
                "impacted": ob != ov,
                "change_types": change_types,
            })

        impact_summary[vid] = {
            "total_impacted": len(impacted_tests),
            "impacted_tests": impacted_tests,
            "cost_usd": round(v_cost, 6),
            "errors": v_errors,
        }
        passed = sum(1 for r in version_results if r.outcome == "pass")
        print(f"  {vid}: {passed}/{n_tests} pass, {len(impacted_tests)} impacted, "
              f"{v_errors} err, ${v_cost:.4f}")

    wall_seconds = round(time.time() - wall_start, 1)

    summary = {
        "domain": domain,
        "total_versions": len(versions),
        "total_tests": len(test_cases),
        "total_llm_calls": total_calls,
        "total_cost_usd": round(total_cost, 4),
        "wall_clock_seconds": wall_seconds,
        "base_pass_rate": round(base_pass / max(len(test_cases), 1), 2),
        "base_error_count": base_errors,
        "impact_summary": impact_summary,
    }

    return {
        "domain": domain,
        "detail_records": detail_records,
        "summary": summary,
    }


def write_results(results: dict, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ground truth detail and summary files."""
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "results" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    domain = results["domain"]
    detail_path = output_dir / f"ground_truth_{domain}.jsonl"
    summary_path = output_dir / f"summary_{domain}.json"

    with open(detail_path, "w") as f:
        for record in results["detail_records"]:
            f.write(json.dumps(record) + "\n")

    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, indent=2)

    return detail_path, summary_path
