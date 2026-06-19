"""High-level exclusion manager with zero-inference fast path."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from ..config import DEFAULT_CACHE_DIR
from .matcher import WildcardIndex, WildcardRule
from .normalize import cache_key, domain_ancestors, extract_domain, normalize_domain_pattern, normalize_text
from .store import ExcludeStore, RuleType

RuleInputType = Literal["exact", "domain", "suffix", "prefix", "glob", "wildcard"]


@dataclass(frozen=True)
class ExcludeMatch:
    reason: str
    rule_type: str
    pattern: str


@dataclass(frozen=True)
class ScoreCacheHit:
    score: int
    label: str
    confidence: str
    breakdown: dict[str, int]


def _parse_rule(pattern: str, rule_type: RuleInputType) -> tuple[RuleType, str, str]:
    raw = pattern.strip()
    if not raw:
        raise ValueError("Pattern must not be empty.")

    if rule_type == "wildcard":
        if raw.startswith("*."):
            rule_type = "suffix"
        elif "*" in raw or "?" in raw:
            rule_type = "glob"
        elif raw.startswith("*"):
            rule_type = "suffix"
        else:
            rule_type = "exact"

    if rule_type == "exact":
        return "exact", normalize_text(raw), normalize_text(raw)

    if rule_type == "domain":
        domain = normalize_domain_pattern(raw)
        if not domain:
            raise ValueError("Invalid domain pattern.")
        return "domain", raw.lower(), domain

    if rule_type == "suffix":
        suffix = normalize_domain_pattern(raw)
        if not suffix:
            raise ValueError("Invalid suffix pattern.")
        return "suffix", raw.lower(), suffix

    if rule_type == "prefix":
        prefix = normalize_text(raw).rstrip("*")
        if not prefix:
            raise ValueError("Invalid prefix pattern.")
        return "prefix", raw.lower(), prefix

    if rule_type == "glob":
        glob_pattern = normalize_text(raw)
        return "glob", raw.lower(), glob_pattern

    raise ValueError(f"Unsupported rule type: {rule_type}")


class ExcludeManager:
    """Fast exclusion and score-cache checks before inference."""

    def __init__(
        self,
        store: ExcludeStore,
        *,
        enabled: bool = True,
        skip_cache_enabled: bool = True,
        skip_score_threshold: int = 30,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.skip_cache_enabled = skip_cache_enabled
        self.skip_score_threshold = skip_score_threshold
        self._wildcard_index = WildcardIndex()
        self._wildcard_lock = threading.RLock()
        self.reload_wildcards()

    @classmethod
    def open(
        cls,
        cache_dir: str | Path | None = None,
        *,
        db_name: str = "exclude.db",
    ) -> "ExcludeManager":
        base = Path(cache_dir or os.environ.get("RANDOMNESS_CACHE_DIR", DEFAULT_CACHE_DIR))
        db_path = Path(os.environ.get("RANDOMNESS_EXCLUDE_DB_PATH", str(base / db_name)))
        enabled = os.environ.get("RANDOMNESS_EXCLUDE_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        skip_cache_enabled = os.environ.get("RANDOMNESS_SKIP_CACHE_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        skip_score_threshold = int(os.environ.get("RANDOMNESS_SKIP_SCORE_THRESHOLD", "30"))
        store = ExcludeStore(db_path)
        return cls(
            store,
            enabled=enabled,
            skip_cache_enabled=skip_cache_enabled,
            skip_score_threshold=skip_score_threshold,
        )

    def close(self) -> None:
        self.store.close()

    def reload_wildcards(self) -> int:
        rules = [
            WildcardRule(pattern=pattern, rule_type=rule_type, normalized=normalized)
            for pattern, rule_type, normalized in self.store.list_wildcards()
        ]
        with self._wildcard_lock:
            self._wildcard_index.rebuild(rules)
        return len(rules)

    def stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "skip_cache_enabled": self.skip_cache_enabled,
            "skip_score_threshold": self.skip_score_threshold,
            "exact_rules": self.store.count_exact(),
            "wildcard_rules": self.store.count_wildcards(),
            "score_cache_entries": self.store.count_score_cache(),
            "wildcard_index_rules": self._wildcard_index.rule_count,
        }

    def add_rules(
        self,
        patterns: Iterable[tuple[str, RuleInputType]],
    ) -> dict[str, int]:
        exact_items: list[tuple[str, RuleType]] = []
        wildcard_items: list[tuple[str, RuleType, str]] = []

        for pattern, rule_type in patterns:
            parsed_type, stored_pattern, normalized = _parse_rule(pattern, rule_type)
            if parsed_type == "exact":
                exact_items.append((normalized, parsed_type))
            elif parsed_type == "domain":
                exact_items.append((normalized, parsed_type))
                wildcard_items.append((stored_pattern, "domain", normalized))
            else:
                wildcard_items.append((stored_pattern, parsed_type, normalized))

        exact_added, exact_dupes = self.store.add_exact_many(exact_items)
        wildcard_added, wildcard_dupes = self.store.add_wildcards_many(wildcard_items)
        self.reload_wildcards()
        return {
            "added": exact_added + wildcard_added,
            "duplicates": exact_dupes + wildcard_dupes,
            "exact_rules": self.store.count_exact(),
            "wildcard_rules": self.store.count_wildcards(),
        }

    def remove_rules(self, patterns: Iterable[str]) -> dict[str, int]:
        normalized_patterns = [normalize_text(pattern) for pattern in patterns]
        removed_exact = self.store.remove_exact_many(normalized_patterns)
        removed_wild = self.store.remove_wildcards_many([p.lower() for p in patterns])
        self.reload_wildcards()
        return {
            "removed": removed_exact + removed_wild,
            "exact_rules": self.store.count_exact(),
            "wildcard_rules": self.store.count_wildcards(),
        }

    def check_exclude(self, text: str) -> ExcludeMatch | None:
        if not self.enabled:
            return None

        normalized = normalize_text(text)
        domain = extract_domain(text)

        if self.store.has_exact(normalized):
            return ExcludeMatch(reason=f"exact:{normalized}", rule_type="exact", pattern=normalized)

        if domain and self.store.has_exact(domain):
            return ExcludeMatch(reason=f"domain:{domain}", rule_type="domain", pattern=domain)

        if domain:
            ancestors = domain_ancestors(domain)
            blocked = self.store.find_exact_domains(ancestors)
            if blocked:
                pattern = sorted(blocked, key=len)[0]
                return ExcludeMatch(reason=f"domain:{pattern}", rule_type="domain", pattern=pattern)

        with self._wildcard_lock:
            wildcard_reason = self._wildcard_index.match_text(normalized, domain)
        if wildcard_reason:
            rule_type, pattern = wildcard_reason.split(":", 1)
            return ExcludeMatch(reason=wildcard_reason, rule_type=rule_type, pattern=pattern)

        return None

    def check_exclude_many(self, texts: list[str]) -> list[ExcludeMatch | None]:
        if not self.enabled:
            return [None] * len(texts)

        normalized_list = [normalize_text(text) for text in texts]
        domains = [extract_domain(text) for text in texts]

        exact_keys: list[str] = []
        domain_keys: list[str] = []
        key_positions: dict[str, list[int]] = {}
        domain_positions: dict[str, list[int]] = {}
        for index, (normalized, domain) in enumerate(zip(normalized_list, domains, strict=True)):
            exact_keys.append(normalized)
            key_positions.setdefault(normalized, []).append(index)
            if domain:
                domain_keys.append(domain)
                key_positions.setdefault(domain, []).append(index)
                for ancestor in domain_ancestors(domain):
                    domain_keys.append(ancestor)
                    domain_positions.setdefault(ancestor, []).append(index)

        hits: list[ExcludeMatch | None] = [None] * len(texts)
        if exact_keys:
            found = self.store.find_exact_many(list(set(exact_keys)))
            for key in found:
                for index in key_positions.get(key, []):
                    if hits[index] is None:
                        if key == normalized_list[index]:
                            hits[index] = ExcludeMatch(
                                reason=f"exact:{key}",
                                rule_type="exact",
                                pattern=key,
                            )
                        else:
                            hits[index] = ExcludeMatch(
                                reason=f"domain:{key}",
                                rule_type="domain",
                                pattern=key,
                            )

        if domain_keys:
            blocked_domains = self.store.find_exact_domains(list(set(domain_keys)))
            for key in blocked_domains:
                for index in domain_positions.get(key, []):
                    if hits[index] is None:
                        hits[index] = ExcludeMatch(
                            reason=f"domain:{key}",
                            rule_type="domain",
                            pattern=key,
                        )

        with self._wildcard_lock:
            for index, (normalized, domain) in enumerate(zip(normalized_list, domains, strict=True)):
                if hits[index] is not None:
                    continue
                wildcard_reason = self._wildcard_index.match_text(normalized, domain)
                if wildcard_reason:
                    rule_type, pattern = wildcard_reason.split(":", 1)
                    hits[index] = ExcludeMatch(
                        reason=wildcard_reason,
                        rule_type=rule_type,
                        pattern=pattern,
                    )
        return hits

    def get_cached_score(self, text: str) -> ScoreCacheHit | None:
        if not self.skip_cache_enabled:
            return None
        row = self.store.get_score_cache(cache_key(text))
        if row is None:
            return None
        score = int(row["score"])
        if score > self.skip_score_threshold:
            return None
        return ScoreCacheHit(
            score=score,
            label=str(row["label"]),
            confidence=str(row["confidence"]),
            breakdown=json.loads(str(row["breakdown_json"])),
        )

    def get_cached_scores_many(self, texts: list[str]) -> list[ScoreCacheHit | None]:
        if not self.skip_cache_enabled:
            return [None] * len(texts)
        keys = [cache_key(text) for text in texts]
        rows = self.store.get_score_cache_many(keys)
        hits: list[ScoreCacheHit | None] = []
        for key in keys:
            row = rows.get(key)
            if row is None:
                hits.append(None)
                continue
            score = int(row["score"])
            if score > self.skip_score_threshold:
                hits.append(None)
                continue
            hits.append(
                ScoreCacheHit(
                    score=score,
                    label=str(row["label"]),
                    confidence=str(row["confidence"]),
                    breakdown=json.loads(str(row["breakdown_json"])),
                )
            )
        return hits

    def store_score(self, text: str, result: dict[str, Any]) -> None:
        if not self.skip_cache_enabled:
            return
        self.store.put_score_cache_many(
            [
                (
                    cache_key(text),
                    int(result["score"]),
                    str(result["label"]),
                    str(result["confidence"]),
                    json.dumps(result["breakdown"], separators=(",", ":")),
                )
            ]
        )

    def store_scores_many(self, texts: list[str], results: list[dict[str, Any]]) -> None:
        if not self.skip_cache_enabled:
            return
        rows = []
        for text, result in zip(texts, results, strict=True):
            rows.append(
                (
                    cache_key(text),
                    int(result["score"]),
                    str(result["label"]),
                    str(result["confidence"]),
                    json.dumps(result["breakdown"], separators=(",", ":")),
                )
            )
        if rows:
            self.store.put_score_cache_many(rows)

    @staticmethod
    def excluded_result(match: ExcludeMatch) -> dict[str, Any]:
        return {
            "score": 0,
            "label": "excluded",
            "confidence": "high",
            "breakdown": {"freq": 0, "entropy": 0, "compression": 0},
            "features": None,
            "excluded": True,
            "exclude_reason": match.reason,
            "exclude_rule_type": match.rule_type,
            "exclude_pattern": match.pattern,
            "cached": False,
            "skipped": True,
            "skipped_reason": "excluded",
        }

    @staticmethod
    def cached_result(hit: ScoreCacheHit) -> dict[str, Any]:
        return {
            "score": hit.score,
            "label": hit.label,
            "confidence": hit.confidence,
            "breakdown": hit.breakdown,
            "features": None,
            "excluded": False,
            "exclude_reason": None,
            "cached": True,
            "skipped": True,
            "skipped_reason": "score_cache_below_threshold",
        }

    @staticmethod
    def scored_result(result: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(result)
        enriched.setdefault("excluded", False)
        enriched.setdefault("exclude_reason", None)
        enriched.setdefault("cached", False)
        enriched.setdefault("skipped", False)
        enriched.setdefault("skipped_reason", None)
        return enriched
