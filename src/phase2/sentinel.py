"""Sentinel sampler: draw a random safety sample from non-predicted tests.

If any sentinel test turns out to be impacted during a live run (or in
evaluation against ground truth), the impact predictor missed something
and a full rerun should be triggered.

No ground truth or version metadata is used — only test IDs and the
predicted set.
"""

from __future__ import annotations

import math
import random


def sample_sentinels(
    all_test_ids: set[str],
    predicted_ids: set[str],
    *,
    fraction: float = 0.10,
    min_sentinels: int = 5,
    seed: int | None = None,
) -> set[str]:
    """Sample sentinel tests from the non-predicted pool.

    Args:
        all_test_ids: Every test ID in the eval suite.
        predicted_ids: Tests already selected by the impact predictor.
        fraction: Proportion of the non-predicted pool to sample.
        min_sentinels: Minimum number of sentinels to draw.
        seed: RNG seed for reproducibility.

    Returns:
        Set of sentinel test IDs. Empty if all tests are already predicted.
    """
    pool = sorted(all_test_ids - predicted_ids)
    if not pool:
        return set()

    count = max(min_sentinels, math.ceil(fraction * len(pool)))
    count = min(count, len(pool))

    rng = random.Random(seed)
    return set(rng.sample(pool, count))
