"""Phase 2 Sprint 3: Evaluator.

Runs the rule-based selector against Phase 1 ground truth for every
prompt-version pair, computes recall / call-reduction metrics, compares
against baselines, and aggregates results along multiple dimensions.

Ground truth is loaded *only* for post-hoc comparison — the selector
itself never sees it (no oracle leakage).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from src.phase1.models import TestCase
from src.phase1.prompts.loader import load_prompt_sections, assemble_prompt
from src.phase2.selector import SelectionResult, select_tests

# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def load_base_prompt(domain: str) -> str:
    return assemble_prompt(load_prompt_sections(domain))


def load_version_prompt(domain: str, version_id: str) -> str:
    path = _DATA_DIR / domain / "versions" / f"{version_id}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["prompt_text"]


def load_test_cases(domain: str) -> list[TestCase]:
    path = _DATA_DIR / domain / "eval_suite.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        TestCase(
            test_id=t["test_id"],
            input_text=t.get("input_text", ""),
            monitor_type=t["monitor_type"],
            monitor_config=t.get("monitor_config", {}),
            tags=t.get("tags", []),
            sensitive_to=t.get("sensitive_to", []),
        )
        for t in data["tests"]
    ]


def list_version_ids(domain: str) -> list[str]:
    vdir = _DATA_DIR / domain / "versions"
    return sorted(p.stem for p in vdir.glob("v*.yaml"))


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Per-version ground truth extracted from Phase 1 JSONL."""

    impacted_by_version: dict[str, set[str]] = field(default_factory=dict)
    change_types_by_version: dict[str, list[str]] = field(default_factory=dict)


def load_ground_truth(domain: str) -> GroundTruth:
    path = _RESULTS_DIR / "baseline" / f"ground_truth_{domain}.jsonl"
    gt = GroundTruth()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vid = row["version_id"]
            tid = row["test_id"]
            if vid not in gt.impacted_by_version:
                gt.impacted_by_version[vid] = set()
                gt.change_types_by_version[vid] = row.get("change_types", [])
            if row["impacted"]:
                gt.impacted_by_version[vid].add(tid)
    return gt


# ---------------------------------------------------------------------------
# Per-version metrics
# ---------------------------------------------------------------------------

@dataclass
class VersionMetrics:
    version_id: str
    domain: str
    num_impacted: int
    num_predicted: int
    num_sentinel: int
    num_selected: int
    total_tests: int
    recall_predictor: float
    recall_with_sentinel: float
    call_reduction: float
    false_omission_rate: float
    false_omissions: list[str]
    sentinel_hit: bool
    changes_detected: list[dict]
    change_types_gt: list[str]
    magnitude_max: float
    mutation_category: str


def _version_to_category(version_id: str) -> str:
    num = int(re.search(r"\d+", version_id).group())  # type: ignore[union-attr]
    return "mechanical" if num <= 50 else "llm_generated"


def _magnitude_bucket(mag: float) -> str:
    if mag < 0.05:
        return "low"
    if mag <= 0.20:
        return "medium"
    return "high"


def _primary_unit_type(changes_detected: list[dict]) -> str:
    """Return unit_type label; 'compound' if >1 distinct unit_type."""
    types = {c["unit_type"] for c in changes_detected}
    if len(types) == 0:
        return "none"
    if len(types) == 1:
        return next(iter(types))
    return "compound"


def evaluate_version(
    domain: str,
    version_id: str,
    base_prompt: str,
    test_cases: list[TestCase],
    gt: GroundTruth,
    *,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int | None = 42,
) -> VersionMetrics:
    """Run the selector on one version and compare against ground truth."""

    version_prompt = load_version_prompt(domain, version_id)
    res: SelectionResult = select_tests(
        base_prompt,
        version_prompt,
        test_cases,
        sentinel_fraction=sentinel_fraction,
        magnitude_threshold=magnitude_threshold,
        sentinel_seed=sentinel_seed,
    )

    impacted: set[str] = gt.impacted_by_version.get(version_id, set())
    n_impacted = len(impacted)

    predicted_hit = impacted & res.predicted_ids
    selected_hit = impacted & res.selected_ids
    missed = impacted - res.selected_ids
    sentinel_caught = len(impacted & res.sentinel_ids) > 0

    recall_pred = len(predicted_hit) / n_impacted if n_impacted > 0 else 1.0
    recall_sel = len(selected_hit) / n_impacted if n_impacted > 0 else 1.0
    not_selected = res.total_tests - len(res.selected_ids)
    for_rate = len(missed) / not_selected if not_selected > 0 else 0.0

    changes_info = [
        {
            "unit_type": c.change.unit_type,
            "change_type": c.change.change_type,
            "magnitude": c.change.magnitude,
            "affected_keys": c.affected_keys,
            "affected_demo_labels": c.affected_demo_labels,
        }
        for c in res.changes
    ]
    mag_max = max((c.change.magnitude for c in res.changes), default=0.0)

    return VersionMetrics(
        version_id=version_id,
        domain=domain,
        num_impacted=n_impacted,
        num_predicted=len(res.predicted_ids),
        num_sentinel=len(res.sentinel_ids),
        num_selected=len(res.selected_ids),
        total_tests=res.total_tests,
        recall_predictor=recall_pred,
        recall_with_sentinel=recall_sel,
        call_reduction=res.call_reduction,
        false_omission_rate=for_rate,
        false_omissions=sorted(missed),
        sentinel_hit=sentinel_caught,
        changes_detected=changes_info,
        change_types_gt=gt.change_types_by_version.get(version_id, []),
        magnitude_max=mag_max,
        mutation_category=_version_to_category(version_id),
    )


# ---------------------------------------------------------------------------
# Baseline selectors
# ---------------------------------------------------------------------------

_HEURISTIC_MAP: dict[str, set[str] | None] = {
    "format": {"schema", "required_keys", "format"},
    "demonstration": {"keyword_presence", "code_execution"},
    "demo_item": {"keyword_presence", "code_execution"},
    "policy": {"policy"},
    "workflow": {"keyword_presence"},
    "role": None,  # select all
}


def baseline_full_rerun(test_cases: list[TestCase]) -> set[str]:
    return {tc.test_id for tc in test_cases}


def baseline_random_50(
    test_cases: list[TestCase], *, seed: int = 42
) -> set[str]:
    rng = random.Random(seed)
    ids = [tc.test_id for tc in test_cases]
    k = max(1, len(ids) // 2)
    return set(rng.sample(ids, k))


def baseline_monitor_heuristic(
    test_cases: list[TestCase],
    change_unit_types: list[str],
) -> set[str]:
    """Select tests whose monitor_type matches the changed section family.

    Args:
        change_unit_types: list of unit_type strings (e.g. ["format", "policy"]).
    """
    unit_types = set(change_unit_types)

    allowed_monitors: set[str] | None = set()
    for ut in unit_types:
        monitors = _HEURISTIC_MAP.get(ut)
        if monitors is None:
            allowed_monitors = None
            break
        allowed_monitors |= monitors  # type: ignore[operator]

    if allowed_monitors is None:
        return {tc.test_id for tc in test_cases}
    return {tc.test_id for tc in test_cases if tc.monitor_type in allowed_monitors}


def _evaluate_baseline(
    selected: set[str],
    impacted: set[str],
    total_tests: int,
) -> dict:
    n_i = len(impacted)
    hit = impacted & selected
    recall = len(hit) / n_i if n_i > 0 else 1.0
    reduction = 1.0 - len(selected) / total_tests if total_tests > 0 else 0.0
    not_selected = total_tests - len(selected)
    for_rate = (n_i - len(hit)) / not_selected if not_selected > 0 else 0.0
    return {
        "recall": recall,
        "call_reduction": reduction,
        "false_omission_rate": for_rate,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _aggregate_group(metrics: list[VersionMetrics]) -> dict:
    if not metrics:
        return {}
    return {
        "count": len(metrics),
        "mean_recall_predictor": round(_mean([m.recall_predictor for m in metrics]), 4),
        "mean_recall_with_sentinel": round(_mean([m.recall_with_sentinel for m in metrics]), 4),
        "mean_call_reduction": round(_mean([m.call_reduction for m in metrics]), 4),
        "mean_false_omission_rate": round(_mean([m.false_omission_rate for m in metrics]), 4),
        "sentinel_catch_rate": round(
            sum(1 for m in metrics if m.sentinel_hit and m.num_impacted > 0 and m.recall_predictor < 1.0)
            / sum(1 for m in metrics if m.num_impacted > 0 and m.recall_predictor < 1.0),
            4,
        ) if sum(1 for m in metrics if m.num_impacted > 0 and m.recall_predictor < 1.0) > 0 else None,
        "versions_with_perfect_recall": sum(
            1 for m in metrics if m.recall_with_sentinel >= 1.0 - 1e-9
        ),
    }


def aggregate_results(
    all_metrics: list[VersionMetrics],
    test_cases: list[TestCase],
    gt: GroundTruth,
) -> dict:
    """Build the full metrics JSON structure with breakdowns and baselines."""

    total_tests = len(test_cases)

    # --- overall aggregate ---
    agg = _aggregate_group(all_metrics)
    agg["total_versions"] = len(all_metrics)
    agg["total_tests"] = total_tests

    # --- by change unit_type ---
    by_type: dict[str, list[VersionMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_type[_primary_unit_type(m.changes_detected)].append(m)
    by_type_agg = {k: _aggregate_group(v) for k, v in sorted(by_type.items())}

    # --- by magnitude bucket ---
    by_mag: dict[str, list[VersionMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_mag[_magnitude_bucket(m.magnitude_max)].append(m)
    by_mag_agg = {k: _aggregate_group(v) for k, v in sorted(by_mag.items())}

    # --- by mutation category ---
    by_cat: dict[str, list[VersionMetrics]] = defaultdict(list)
    for m in all_metrics:
        by_cat[m.mutation_category].append(m)
    by_cat_agg = {k: _aggregate_group(v) for k, v in sorted(by_cat.items())}

    # --- baselines (aggregate over all versions) ---
    full_rerun_recalls: list[float] = []
    random50_recalls: list[float] = []
    random50_reductions: list[float] = []
    heuristic_recalls: list[float] = []
    heuristic_reductions: list[float] = []

    all_ids = {tc.test_id for tc in test_cases}
    rand_sel = baseline_random_50(test_cases, seed=42)

    for m in all_metrics:
        impacted = gt.impacted_by_version.get(m.version_id, set())

        fr = _evaluate_baseline(all_ids, impacted, total_tests)
        full_rerun_recalls.append(fr["recall"])

        r50 = _evaluate_baseline(rand_sel, impacted, total_tests)
        random50_recalls.append(r50["recall"])
        random50_reductions.append(r50["call_reduction"])

        detected_types = [c["unit_type"] for c in m.changes_detected]
        heur_sel = baseline_monitor_heuristic(test_cases, detected_types)
        h = _evaluate_baseline(heur_sel, impacted, total_tests)
        heuristic_recalls.append(h["recall"])
        heuristic_reductions.append(h["call_reduction"])

    baselines = {
        "full_rerun": {
            "mean_recall": round(_mean(full_rerun_recalls), 4),
            "call_reduction": 0.0,
        },
        "random_50": {
            "mean_recall": round(_mean(random50_recalls), 4),
            "call_reduction": round(_mean(random50_reductions), 4),
        },
        "monitor_heuristic": {
            "mean_recall": round(_mean(heuristic_recalls), 4),
            "call_reduction": round(_mean(heuristic_reductions), 4),
        },
    }

    domain = all_metrics[0].domain if all_metrics else "unknown"
    return {
        "domain": domain,
        "aggregate": agg,
        "by_change_type": by_type_agg,
        "by_magnitude_bucket": by_mag_agg,
        "by_category": by_cat_agg,
        "baselines": baselines,
    }


# ---------------------------------------------------------------------------
# Full evaluation driver
# ---------------------------------------------------------------------------

def run_evaluation(
    domain: str,
    *,
    sentinel_fraction: float = 0.10,
    magnitude_threshold: float = 0.005,
    sentinel_seed: int = 42,
    progress: bool = False,
) -> tuple[list[VersionMetrics], dict]:
    """Run the full evaluation for one domain.

    Returns:
        (per_version_metrics, aggregated_metrics_dict)
    """
    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    version_ids = list_version_ids(domain)

    iterator = version_ids
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(version_ids, desc=f"Evaluating {domain}")
        except ImportError:
            pass

    all_metrics: list[VersionMetrics] = []
    for vid in iterator:
        vm = evaluate_version(
            domain,
            vid,
            base_prompt,
            test_cases,
            gt,
            sentinel_fraction=sentinel_fraction,
            magnitude_threshold=magnitude_threshold,
            sentinel_seed=sentinel_seed,
        )
        all_metrics.append(vm)

    agg = aggregate_results(all_metrics, test_cases, gt)
    return all_metrics, agg


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

_SENTINEL_FRACTIONS = [0.05, 0.10, 0.15, 0.20]
_MAGNITUDE_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]


def run_sweep(
    domain: str,
    *,
    sentinel_seed: int = 42,
    progress: bool = False,
) -> list[dict]:
    """Sweep sentinel_fraction x magnitude_threshold grid and return results."""

    base_prompt = load_base_prompt(domain)
    test_cases = load_test_cases(domain)
    gt = load_ground_truth(domain)
    version_ids = list_version_ids(domain)

    configs = [
        (sf, mt)
        for sf in _SENTINEL_FRACTIONS
        for mt in _MAGNITUDE_THRESHOLDS
    ]

    sweep_results: list[dict] = []

    outer = configs
    if progress:
        try:
            from tqdm import tqdm
            outer = tqdm(configs, desc=f"Sweep {domain}")
        except ImportError:
            pass

    for sf, mt in outer:
        version_metrics: list[VersionMetrics] = []
        for vid in version_ids:
            vm = evaluate_version(
                domain,
                vid,
                base_prompt,
                test_cases,
                gt,
                sentinel_fraction=sf,
                magnitude_threshold=mt,
                sentinel_seed=sentinel_seed,
            )
            version_metrics.append(vm)

        agg = aggregate_results(version_metrics, test_cases, gt)
        sweep_results.append({
            "sentinel_fraction": sf,
            "magnitude_threshold": mt,
            "aggregate": agg["aggregate"],
        })

    return sweep_results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(
    domain: str,
    all_metrics: list[VersionMetrics],
    agg: dict,
    sweep: list[dict] | None = None,
) -> Path:
    """Write metrics JSON, detail JSONL, and optional sweep JSON."""
    out_dir = _RESULTS_DIR / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"metrics_{domain}.json"
    with open(metrics_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)

    detail_path = out_dir / f"detail_{domain}.jsonl"
    with open(detail_path, "w") as f:
        for m in all_metrics:
            row = {
                "version_id": m.version_id,
                "changes": m.changes_detected,
                "predicted_count": m.num_predicted,
                "sentinel_count": m.num_sentinel,
                "selected_count": m.num_selected,
                "impacted_count": m.num_impacted,
                "recall_predictor": round(m.recall_predictor, 4),
                "recall_with_sentinel": round(m.recall_with_sentinel, 4),
                "call_reduction": round(m.call_reduction, 4),
                "false_omission_rate": round(m.false_omission_rate, 4),
                "false_omissions": m.false_omissions,
                "sentinel_hit": m.sentinel_hit,
                "magnitude_max": round(m.magnitude_max, 4),
                "mutation_category": m.mutation_category,
            }
            f.write(json.dumps(row) + "\n")

    if sweep is not None:
        sweep_path = out_dir / f"sweep_{domain}.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep, f, indent=2, default=str)

    return out_dir
