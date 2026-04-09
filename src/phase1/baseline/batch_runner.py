"""Batch API baseline runner.

Uses OpenAI's Batch API for bulk evaluation — separate rate limits from
the real-time API, 50% cost discount, and handles thousands of requests
without hitting RPM/RPD/TPM limits.

Flow:
  1. Build a JSONL file with all (prompt, test_case) combinations.
  2. Upload the file to OpenAI.
  3. Create a batch job.
  4. Poll until complete.
  5. Download results and map back to EvalResults.
  6. Run monitors locally and compute impact.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import yaml
from openai import OpenAI
from dotenv import load_dotenv

from src.phase1.harness.monitors import run_monitor
from src.phase1.harness.runner import load_eval_suite, load_settings
from src.phase1.models import EvalResult, TestCase

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_BATCH_COST_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.075, 0.30),
    "gpt-4o": (1.25, 5.00),
    "gpt-3.5-turbo": (0.25, 0.75),
}


def load_versions(domain: str) -> list[dict]:
    versions_dir = _PROJECT_ROOT / "data" / domain / "versions"
    version_files = sorted(versions_dir.glob("v*.yaml"))
    versions = []
    for vf in version_files:
        with open(vf) as f:
            versions.append(yaml.safe_load(f))
    return versions


def _build_batch_jsonl(
    base_prompt: str,
    versions: list[dict],
    test_cases: list[TestCase],
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, tuple[str, TestCase]]]:
    """Build a JSONL string and a mapping from custom_id to (version_id, TestCase)."""
    lines: list[str] = []
    id_map: dict[str, tuple[str, TestCase]] = {}

    # Base prompt tests
    for tc in test_cases:
        cid = f"base__{tc.test_id}"
        id_map[cid] = ("base", tc)
        lines.append(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": tc.input_text},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }))

    # Version tests
    for vdata in versions:
        vid = vdata["version_id"]
        prompt_text = vdata["prompt_text"]
        for tc in test_cases:
            cid = f"{vid}__{tc.test_id}"
            id_map[cid] = (vid, tc)
            lines.append(json.dumps({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": tc.input_text},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }))

    return "\n".join(lines), id_map


def run_batch_baseline(domain: str, *, settings: dict | None = None) -> dict:
    """Run baseline via OpenAI Batch API. Blocking — polls until complete."""
    if settings is None:
        settings = load_settings()

    model = settings["model"]
    temperature = settings["temperature"]
    max_tokens = settings.get("max_tokens", 1024)

    test_cases = load_eval_suite(domain)
    versions = load_versions(domain)

    base_path = _PROJECT_ROOT / "data" / domain / "base_prompt.txt"
    base_prompt = base_path.read_text()

    total_requests = (1 + len(versions)) * len(test_cases)
    print(f"[{domain}] Building batch: {1 + len(versions)} prompts × {len(test_cases)} tests = {total_requests} requests")

    jsonl_content, id_map = _build_batch_jsonl(
        base_prompt, versions, test_cases, model, temperature, max_tokens,
    )

    client = OpenAI()

    # Upload JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(jsonl_content)
        tmp_path = f.name

    print(f"[{domain}] Uploading batch file ({len(jsonl_content) / 1024:.0f} KB, {total_requests} requests)...")
    try:
        with open(tmp_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    print(f"[{domain}] File uploaded: {batch_file.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"domain": domain, "description": f"CARTS baseline {domain}"},
    )
    print(f"[{domain}] Batch created: {batch.id}")

    # Poll for completion (timeout after 24h to avoid infinite loop)
    wall_start = time.time()
    max_poll_seconds = 24 * 3600
    while True:
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        completed = counts.completed if counts else 0
        failed = counts.failed if counts else 0
        total = counts.total if counts else total_requests
        elapsed = time.time() - wall_start

        print(f"  [{elapsed:>6.0f}s] {batch.status}: {completed}/{total} done, {failed} failed", end="\r")

        if batch.status in ("completed", "failed", "expired", "cancelled"):
            print()
            break
        if elapsed > max_poll_seconds:
            raise RuntimeError(f"Batch {batch.id} timed out after {elapsed:.0f}s (status: {batch.status})")
        time.sleep(15)

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

    wall_seconds = round(time.time() - wall_start, 1)
    print(f"[{domain}] Batch completed in {wall_seconds}s")

    # Download results
    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed but has no output_file_id")
    output_content = client.files.content(batch.output_file_id).text
    result_lines = [json.loads(line) for line in output_content.strip().split("\n") if line.strip()]

    # Map results back
    eval_results: dict[str, list[EvalResult]] = {}
    total_cost = 0.0

    for rline in result_lines:
        cid = rline["custom_id"]
        vid, tc = id_map[cid]

        response_body = rline.get("response", {}).get("body", {})
        error = rline.get("error")

        if error or not response_body.get("choices"):
            result = EvalResult(
                version_id=vid,
                test_id=tc.test_id,
                outcome="fail",
                raw_output=f"ERROR: {error or 'no choices'}",
                monitor_details={"error": str(error or "no choices")},
                cost_usd=0.0,
            )
        else:
            raw_output = response_body["choices"][0]["message"]["content"] or ""
            usage = response_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            input_cpm, output_cpm = _BATCH_COST_PER_M.get(model, (0.075, 0.30))
            cost_usd = (prompt_tokens * input_cpm + completion_tokens * output_cpm) / 1_000_000

            monitor_result = run_monitor(raw_output, tc.monitor_type, tc.monitor_config)

            result = EvalResult(
                version_id=vid,
                test_id=tc.test_id,
                outcome="pass" if monitor_result.passed else "fail",
                raw_output=raw_output,
                monitor_details=monitor_result.details,
                cost_usd=cost_usd,
            )
            total_cost += cost_usd

        eval_results.setdefault(vid, []).append(result)

    # Compute impact
    base_outcomes = {r.test_id: r.outcome for r in eval_results.get("base", [])}
    base_pass = sum(1 for o in base_outcomes.values() if o == "pass")
    base_errors = sum(1 for r in eval_results.get("base", []) if r.raw_output.startswith("ERROR:"))
    print(f"[{domain}] Base: {base_pass}/{len(test_cases)} pass, {base_errors} errors")

    detail_records: list[dict] = []
    impact_summary: dict[str, dict] = {}

    for vdata in versions:
        vid = vdata["version_id"]
        change_types = [f"{c['unit_type']}:{c['change_type']}" for c in vdata.get("changes", [])]
        vresults = eval_results.get(vid, [])
        version_outcomes = {r.test_id: r.outcome for r in vresults}
        v_cost = sum(r.cost_usd for r in vresults)
        v_errors = sum(1 for r in vresults if r.raw_output.startswith("ERROR:"))

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
        passed = sum(1 for r in vresults if r.outcome == "pass")
        print(f"  {vid}: {passed}/{len(test_cases)} pass, {len(impacted_tests)} impacted, {v_errors} err")

    summary = {
        "domain": domain,
        "total_versions": len(versions),
        "total_tests": len(test_cases),
        "total_llm_calls": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "wall_clock_seconds": wall_seconds,
        "base_pass_rate": round(base_pass / max(len(test_cases), 1), 2),
        "base_error_count": base_errors,
        "batch_id": batch.id,
        "impact_summary": impact_summary,
    }

    return {
        "domain": domain,
        "detail_records": detail_records,
        "summary": summary,
    }


def write_results(results: dict, output_dir: Path | None = None) -> tuple[Path, Path]:
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
