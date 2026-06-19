"""Scoring pipeline with exclusion and score-cache fast path."""

from __future__ import annotations

from typing import Any

from ..exclude import ExcludeManager
from ..inference_pool import InferencePool


async def score_texts(
    pool: InferencePool,
    manager: ExcludeManager,
    texts: list[str],
    *,
    use_exclude: bool = True,
    use_score_cache: bool = True,
) -> list[dict[str, Any]]:
    if not texts:
        return []

    if not use_exclude and not use_score_cache:
        scored = await pool.score_batch(texts)
        return [ExcludeManager.scored_result(item) for item in scored]

    excludes = manager.check_exclude_many(texts) if use_exclude else [None] * len(texts)
    caches = manager.get_cached_scores_many(texts) if use_score_cache else [None] * len(texts)

    results: list[dict[str, Any] | None] = [None] * len(texts)
    pending_indices: list[int] = []

    for index, text in enumerate(texts):
        if use_exclude and excludes[index] is not None:
            results[index] = ExcludeManager.excluded_result(excludes[index])
            continue
        if use_score_cache and caches[index] is not None:
            results[index] = ExcludeManager.cached_result(caches[index])
            continue
        pending_indices.append(index)

    if pending_indices:
        pending_texts = [texts[index] for index in pending_indices]
        scored = await pool.score_batch(pending_texts)
        for index, item in zip(pending_indices, scored, strict=True):
            enriched = ExcludeManager.scored_result(item)
            results[index] = enriched
        if use_score_cache:
            manager.store_scores_many(pending_texts, scored)

    return [item for item in results if item is not None]


async def score_one(
    pool: InferencePool,
    manager: ExcludeManager,
    text: str,
    *,
    use_exclude: bool = True,
    use_score_cache: bool = True,
) -> dict[str, Any]:
    results = await score_texts(
        pool,
        manager,
        [text],
        use_exclude=use_exclude,
        use_score_cache=use_score_cache,
    )
    return results[0]
