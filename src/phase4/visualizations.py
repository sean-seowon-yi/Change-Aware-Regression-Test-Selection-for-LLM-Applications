"""Publication-quality figure generation for Phase 4.

All figures are saved as PNG at 150 DPI.  Uses matplotlib with seaborn
styling for a clean, consistent look.  Each function takes pre-computed
data and an output path — no data loading or computation happens here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_STYLE_APPLIED = False


def _apply_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.1)
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# F1: Recall–Reduction Pareto Front
# ---------------------------------------------------------------------------

def plot_pareto_fronts(
    threshold_data: dict[str, dict[str, list[dict]]],
    output_path: Path,
) -> None:
    """Three-panel Pareto front (one per domain).

    *threshold_data* maps domain → selector → curve points.  Each point may include
    ``mean_effective_recall`` / ``mean_effective_call_reduction`` (preferred) or
    legacy ``mean_recall_with_sentinel`` / ``mean_call_reduction``.
    """
    _apply_style()
    import matplotlib.pyplot as plt

    domains = sorted(threshold_data.keys())
    n = len(domains)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5), sharey=True, squeeze=False)

    for idx, domain in enumerate(domains):
        ax = axes[0, idx]
        for label, curve in sorted(threshold_data[domain].items()):
            use_effective = bool(curve) and all(
                r.get("mean_effective_recall") is not None
                and r.get("mean_effective_call_reduction") is not None
                for r in curve
            )
            if use_effective:
                recalls = [float(r["mean_effective_recall"]) for r in curve]
                reductions = [float(r["mean_effective_call_reduction"]) for r in curve]
            else:
                recalls = [
                    r.get("mean_recall_with_sentinel", r.get("recall", 0))
                    for r in curve
                ]
                reductions = [
                    r.get("mean_call_reduction", r.get("call_reduction", 0))
                    for r in curve
                ]
            ax.plot(reductions, recalls, label=label, marker="o", markersize=2, linewidth=1.5)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% recall target")
        ax.set_xlabel("Call Reduction")
        if idx == 0:
            ax.set_ylabel("Recall")
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F2: Feature Importance Bar Chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_records: list[dict],
    output_path: Path,
    *,
    top_k: int = 20,
    title: str = "Feature Importance (Gain)",
) -> None:
    """Horizontal bar chart of top-k features by importance."""
    _apply_style()
    import matplotlib.pyplot as plt

    sorted_records = sorted(importance_records, key=lambda x: x["importance"], reverse=True)[:top_k]
    names = [r["feature"] for r in sorted_records][::-1]
    values = [r["importance"] for r in sorted_records][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names))))
    ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F3: Recall by Change Type
# ---------------------------------------------------------------------------

def plot_recall_by_change_type(
    by_change_data: dict[str, dict[str, float]],
    output_path: Path,
    *,
    title: str = "Recall by Change Type",
) -> None:
    """Grouped bar chart: change_type on x-axis, bars for each selector.

    *by_change_data* maps ``selector_label -> {change_type: recall}``
    """
    _apply_style()
    import matplotlib.pyplot as plt

    selectors = sorted(by_change_data.keys())
    all_types = sorted({ct for s in by_change_data.values() for ct in s})
    x = np.arange(len(all_types))
    width = 0.8 / max(len(selectors), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(all_types) * 1.5), 5))
    for i, sel in enumerate(selectors):
        vals = [by_change_data[sel].get(ct, 0) for ct in all_types]
        ax.bar(x + i * width, vals, width, label=sel)

    ax.set_xticks(x + width * (len(selectors) - 1) / 2)
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.set_ylabel("Recall")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F4: Cross-Domain Heatmap
# ---------------------------------------------------------------------------

def plot_cross_domain_heatmap(
    matrix_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Annotated 3×3 heatmap of cross-domain recall."""
    _apply_style()
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    fig, ax = plt.subplots(figsize=(6, 5))
    data = matrix_df.apply(pd.to_numeric, errors="coerce")

    if has_seaborn:
        sns.heatmap(
            data, annot=True, fmt=".3f", cmap="YlGnBu",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
    else:
        im = ax.imshow(data.values, cmap="YlGnBu", vmin=0, vmax=1)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.iloc[i, j]
                text = f"{val:.3f}" if pd.notna(val) else "—"
                ax.text(j, i, text, ha="center", va="center", fontsize=10)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)
        fig.colorbar(im, ax=ax)

    ax.set_xlabel("Test Domain")
    ax.set_ylabel("Train Domain")
    ax.set_title("Cross-Domain Recall Transfer")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F5: Magnitude vs. Recall Scatter
# ---------------------------------------------------------------------------

def plot_magnitude_scatter(
    per_version_data: list[dict],
    output_path: Path,
) -> None:
    """Scatter plot: x=max magnitude, y=recall, color=primary unit_type."""
    _apply_style()
    import matplotlib.pyplot as plt

    if not per_version_data:
        logger.warning("No data for magnitude scatter")
        return

    df = pd.DataFrame(per_version_data)
    recall_col = "recall_with_sentinel" if "recall_with_sentinel" in df.columns else "recall_predictor"
    mag_col = "magnitude_max" if "magnitude_max" in df.columns else "magnitude"
    if mag_col not in df.columns or recall_col not in df.columns:
        logger.warning("Missing columns for scatter")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if "change_type" in df.columns:
        for ct, grp in df.groupby("change_type"):
            ax.scatter(grp[mag_col], grp[recall_col], label=ct, alpha=0.7, s=30)
        ax.legend(fontsize=7, title="Change Type")
    else:
        ax.scatter(df[mag_col], df[recall_col], alpha=0.7, s=30)

    ax.set_xlabel("Max Change Magnitude")
    ax.set_ylabel("Recall")
    ax.set_title("Magnitude vs. Recall")
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F6: Confusion-Style Breakdown
# ---------------------------------------------------------------------------

def plot_confusion_breakdown(
    per_version_data: list[dict],
    total_tests: int,
    output_path: Path,
) -> None:
    """Stacked bar of TP/FP/FN/TN averaged across versions."""
    _apply_style()
    import matplotlib.pyplot as plt

    if not per_version_data:
        return

    tp_vals, fp_vals, fn_vals, tn_vals = [], [], [], []
    for v in per_version_data:
        n_imp = v.get("impacted_count", 0)
        n_sel = v.get("selected_count", 0)
        n_missed = len(v.get("false_omissions", []))
        tp = n_imp - n_missed
        fn = n_missed
        fp = max(0, n_sel - tp)
        tn = max(0, total_tests - tp - fp - fn)
        tp_vals.append(tp)
        fp_vals.append(fp)
        fn_vals.append(fn)
        tn_vals.append(tn)

    categories = ["True Positive", "False Positive", "False Negative", "True Negative"]
    means = [np.mean(tp_vals), np.mean(fp_vals), np.mean(fn_vals), np.mean(tn_vals)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(["Avg. per version"], [means[0]], color=colors[0], label=categories[0])
    left = means[0]
    for i in range(1, 4):
        ax.barh(["Avg. per version"], [means[i]], left=left, color=colors[i], label=categories[i])
        left += means[i]

    ax.set_xlabel("Number of Tests")
    ax.set_title("Selection Outcome Breakdown (Average per Version)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# ---------------------------------------------------------------------------
# F7: Cost Savings Curve
# ---------------------------------------------------------------------------

def plot_cost_savings(
    cost_data: list[dict],
    output_path: Path,
) -> None:
    """Monthly savings vs. suite size at varying call reduction rates.

    *cost_data*: list of dicts from ``cost_model.cost_vs_suite_size``.
    """
    _apply_style()
    import matplotlib.pyplot as plt

    if not cost_data:
        return

    df = pd.DataFrame(cost_data)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "call_reduction" in df.columns and df["call_reduction"].nunique() > 1:
        for cr, grp in df.groupby("call_reduction"):
            ax.plot(grp["suite_size"], grp["savings_absolute"],
                    marker="o", label=f"CR={cr:.0%}")
    else:
        ax.plot(df["suite_size"], df["savings_absolute"], marker="o")

    ax.set_xlabel("Suite Size (N tests)")
    ax.set_ylabel("Monthly Savings ($)")
    ax.set_title("Cost Savings vs. Suite Size")
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
