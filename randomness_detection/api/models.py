"""Pydantic request/response models with strict validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RuleTypeInput = Literal["exact", "domain", "suffix", "prefix", "glob", "wildcard"]


MAX_TEXT_LENGTH = 4096
MAX_BATCH_SIZE = 500
MAX_BATCH_ITEM_LENGTH = 1024


def _validate_text_value(value: str, *, max_length: int) -> str:
    if "\x00" in value:
        raise ValueError("Text must not contain null bytes.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Text must not be empty.")
    if len(cleaned) > max_length:
        raise ValueError(f"Text exceeds maximum length of {max_length} characters.")
    return cleaned


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _validate_text_value(value, max_length=MAX_TEXT_LENGTH)


class BatchScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        return [
            _validate_text_value(item, max_length=MAX_BATCH_ITEM_LENGTH)
            for item in values
        ]


class ScoreResponse(BaseModel):
    score: int = Field(..., ge=0, le=100)
    label: str
    confidence: str
    breakdown: dict[str, int]
    features: dict[str, Any] | None = None
    excluded: bool = False
    exclude_reason: str | None = None
    exclude_rule_type: str | None = None
    exclude_pattern: str | None = None
    cached: bool = False
    skipped: bool = False
    skipped_reason: str | None = None


class BatchScoreItem(BaseModel):
    text: str
    score: int = Field(..., ge=0, le=100)
    label: str
    confidence: str
    breakdown: dict[str, int]
    features: dict[str, Any] | None = None
    excluded: bool = False
    exclude_reason: str | None = None
    exclude_rule_type: str | None = None
    exclude_pattern: str | None = None
    cached: bool = False
    skipped: bool = False
    skipped_reason: str | None = None


class BatchScoreResponse(BaseModel):
    count: int
    results: list[BatchScoreItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_ready: bool
    parallel_backend: str
    inference_workers: int
    inference_threads: int
    exclude_enabled: bool
    skip_cache_enabled: bool
    skip_score_threshold: int
    exact_exclude_rules: int
    wildcard_exclude_rules: int
    score_cache_entries: int


class ExcludeRuleItem(BaseModel):
    pattern: str = Field(..., min_length=1, max_length=512)
    rule_type: RuleTypeInput = "wildcard"


class ExcludeAddRequest(BaseModel):
    rules: list[ExcludeRuleItem] = Field(..., min_length=1, max_length=10_000)


class ExcludeRemoveRequest(BaseModel):
    patterns: list[str] = Field(..., min_length=1, max_length=10_000)


class ExcludeMutationResponse(BaseModel):
    added: int | None = None
    removed: int | None = None
    duplicates: int | None = None
    exact_rules: int
    wildcard_rules: int


class ExcludeStatsResponse(BaseModel):
    enabled: bool
    skip_cache_enabled: bool
    skip_score_threshold: int
    exact_rules: int
    wildcard_rules: int
    score_cache_entries: int
    wildcard_index_rules: int


class ExcludeCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


class ExcludeCheckResponse(BaseModel):
    text: str
    excluded: bool
    exclude_reason: str | None = None
    exclude_rule_type: str | None = None
    exclude_pattern: str | None = None
    cached: bool
    cached_score: int | None = None
    would_skip: bool


class ErrorResponse(BaseModel):
    detail: str
