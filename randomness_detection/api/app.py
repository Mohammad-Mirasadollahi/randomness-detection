"""FastAPI application for randomness scoring."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .. import __version__
from ..config import DEFAULT_CACHE_DIR
from ..exclude import ExcludeManager
from ..inference_pool import (
    InferencePool,
    inference_thread_count,
    inference_worker_count,
)
from ..parallel import resolve_parallel_backend
from ..scorer import Scorer
from .auth import load_auth_config, require_api_key
from .models import (
    BatchScoreItem,
    BatchScoreRequest,
    BatchScoreResponse,
    ErrorResponse,
    ExcludeAddRequest,
    ExcludeCheckRequest,
    ExcludeCheckResponse,
    ExcludeMutationResponse,
    ExcludeRemoveRequest,
    ExcludeStatsResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from .scoring import score_one as score_one_text
from .scoring import score_texts


def _cache_dir() -> str:
    return os.environ.get("RANDOMNESS_CACHE_DIR", str(DEFAULT_CACHE_DIR))


def _to_score_response(data: dict[str, Any], *, include_features: bool) -> ScoreResponse:
    features = data.get("features") if include_features else None
    return ScoreResponse(
        score=data["score"],
        label=data["label"],
        confidence=data["confidence"],
        breakdown=data["breakdown"],
        features=features,
        excluded=bool(data.get("excluded", False)),
        exclude_reason=data.get("exclude_reason"),
        exclude_rule_type=data.get("exclude_rule_type"),
        exclude_pattern=data.get("exclude_pattern"),
        cached=bool(data.get("cached", False)),
        skipped=bool(data.get("skipped", False)),
        skipped_reason=data.get("skipped_reason"),
    )


def _to_batch_item(text: str, data: dict[str, Any], *, include_features: bool) -> BatchScoreItem:
    return BatchScoreItem(
        text=text,
        score=data["score"],
        label=data["label"],
        confidence=data["confidence"],
        breakdown=data["breakdown"],
        features=data.get("features") if include_features else None,
        excluded=bool(data.get("excluded", False)),
        exclude_reason=data.get("exclude_reason"),
        exclude_rule_type=data.get("exclude_rule_type"),
        exclude_pattern=data.get("exclude_pattern"),
        cached=bool(data.get("cached", False)),
        skipped=bool(data.get("skipped", False)),
        skipped_reason=data.get("skipped_reason"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_auth_config()
    cache_dir = _cache_dir()

    Scorer(cache_dir=cache_dir, auto_bootstrap=True)

    workers = inference_worker_count()
    threads = inference_thread_count()
    backend = resolve_parallel_backend()
    pool = InferencePool(cache_dir, workers=workers, threads=threads, backend=backend)
    pool.start()

    exclude_manager = ExcludeManager.open(cache_dir)

    app.state.inference_pool = pool
    app.state.inference_workers = workers
    app.state.inference_threads = threads
    app.state.parallel_backend = backend
    app.state.exclude_manager = exclude_manager
    yield
    exclude_manager.close()
    pool.stop()


def create_app() -> FastAPI:
    application = FastAPI(
        title="Randomness Detection API",
        version=__version__,
        description="Score how random a string looks (1-100, higher = more random).",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    allowed_hosts = os.environ.get("RANDOMNESS_ALLOWED_HOSTS", "").strip()
    if allowed_hosts:
        application.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[host.strip() for host in allowed_hosts.split(",") if host.strip()],
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail="Invalid request payload.").model_dump(),
        )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        pool: InferencePool | None = getattr(request.app.state, "inference_pool", None)
        manager: ExcludeManager | None = getattr(request.app.state, "exclude_manager", None)
        stats = manager.stats() if manager is not None else {}
        return HealthResponse(
            status="ok",
            version=__version__,
            model_ready=pool is not None,
            parallel_backend=getattr(request.app.state, "parallel_backend", "process"),
            inference_workers=getattr(request.app.state, "inference_workers", 0),
            inference_threads=getattr(request.app.state, "inference_threads", 0),
            exclude_enabled=bool(stats.get("enabled", False)),
            skip_cache_enabled=bool(stats.get("skip_cache_enabled", False)),
            skip_score_threshold=int(stats.get("skip_score_threshold", 0)),
            exact_exclude_rules=int(stats.get("exact_rules", 0)),
            wildcard_exclude_rules=int(stats.get("wildcard_rules", 0)),
            score_cache_entries=int(stats.get("score_cache_entries", 0)),
        )

    @application.get(
        "/exclude/stats",
        response_model=ExcludeStatsResponse,
        tags=["exclude"],
        dependencies=[Depends(require_api_key)],
    )
    async def exclude_stats(request: Request) -> ExcludeStatsResponse:
        manager: ExcludeManager = request.app.state.exclude_manager
        stats = manager.stats()
        return ExcludeStatsResponse(
            enabled=bool(stats["enabled"]),
            skip_cache_enabled=bool(stats["skip_cache_enabled"]),
            skip_score_threshold=int(stats["skip_score_threshold"]),
            exact_rules=int(stats["exact_rules"]),
            wildcard_rules=int(stats["wildcard_rules"]),
            score_cache_entries=int(stats["score_cache_entries"]),
            wildcard_index_rules=int(stats["wildcard_index_rules"]),
        )

    @application.post(
        "/exclude",
        response_model=ExcludeMutationResponse,
        tags=["exclude"],
        dependencies=[Depends(require_api_key)],
    )
    async def exclude_add(request: Request, body: ExcludeAddRequest) -> ExcludeMutationResponse:
        manager: ExcludeManager = request.app.state.exclude_manager
        result = manager.add_rules([(rule.pattern, rule.rule_type) for rule in body.rules])
        return ExcludeMutationResponse(
            added=result["added"],
            duplicates=result["duplicates"],
            exact_rules=result["exact_rules"],
            wildcard_rules=result["wildcard_rules"],
        )

    @application.delete(
        "/exclude",
        response_model=ExcludeMutationResponse,
        tags=["exclude"],
        dependencies=[Depends(require_api_key)],
    )
    async def exclude_remove(request: Request, body: ExcludeRemoveRequest) -> ExcludeMutationResponse:
        manager: ExcludeManager = request.app.state.exclude_manager
        result = manager.remove_rules(body.patterns)
        return ExcludeMutationResponse(
            removed=result["removed"],
            exact_rules=result["exact_rules"],
            wildcard_rules=result["wildcard_rules"],
        )

    @application.post(
        "/exclude/check",
        response_model=ExcludeCheckResponse,
        tags=["exclude"],
        dependencies=[Depends(require_api_key)],
    )
    async def exclude_check(request: Request, body: ExcludeCheckRequest) -> ExcludeCheckResponse:
        manager: ExcludeManager = request.app.state.exclude_manager
        match = manager.check_exclude(body.text)
        cached = manager.get_cached_score(body.text)
        return ExcludeCheckResponse(
            text=body.text,
            excluded=match is not None,
            exclude_reason=match.reason if match else None,
            exclude_rule_type=match.rule_type if match else None,
            exclude_pattern=match.pattern if match else None,
            cached=cached is not None,
            cached_score=cached.score if cached else None,
            would_skip=match is not None or cached is not None,
        )

    @application.post(
        "/score",
        response_model=ScoreResponse,
        tags=["scoring"],
        dependencies=[Depends(require_api_key)],
    )
    async def score_one(
        request: Request,
        body: ScoreRequest,
        include_features: bool = Query(default=False),
        use_exclude: bool = Query(default=True),
        use_score_cache: bool = Query(default=True),
    ) -> ScoreResponse:
        pool: InferencePool = request.app.state.inference_pool
        manager: ExcludeManager = request.app.state.exclude_manager
        data = await score_one_text(
            pool,
            manager,
            body.text,
            use_exclude=use_exclude,
            use_score_cache=use_score_cache,
        )
        return _to_score_response(data, include_features=include_features)

    @application.post(
        "/score/batch",
        response_model=BatchScoreResponse,
        tags=["scoring"],
        dependencies=[Depends(require_api_key)],
    )
    async def score_batch(
        request: Request,
        body: BatchScoreRequest,
        include_features: bool = Query(default=False),
        use_exclude: bool = Query(default=True),
        use_score_cache: bool = Query(default=True),
    ) -> BatchScoreResponse:
        pool: InferencePool = request.app.state.inference_pool
        manager: ExcludeManager = request.app.state.exclude_manager
        scored = await score_texts(
            pool,
            manager,
            body.texts,
            use_exclude=use_exclude,
            use_score_cache=use_score_cache,
        )
        results = [
            _to_batch_item(body.texts[index], item, include_features=include_features)
            for index, item in enumerate(scored)
        ]
        return BatchScoreResponse(count=len(results), results=results)

    return application


app = create_app()
