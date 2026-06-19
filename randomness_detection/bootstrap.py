"""Bootstrap: download english words, build freq table, train ensemble."""

from __future__ import annotations

import json
from pathlib import Path

from .bootstrap_progress import BootstrapProgress
from .config import (
    BOOTSTRAP_VERSION,
    CPU_FRACTION,
    ENSEMBLE_MODEL_NAME,
    FREQ_MAX_WORDS,
    FREQ_MIN_WORD_LENGTH,
    FREQ_TABLE_NAME,
    METADATA_NAME,
    NATURAL_REAL_WORD_RATIO,
    TRAIN_SAMPLES_PER_CLASS,
    WORD_SOURCES,
)
from .corpora import filter_words_for_freq, filter_words_for_training, load_merged_words
from .corpus_validator import validate_words
from .freq_model import FreqCounter
from .parallel import start_parallel, stop_parallel, shutdown_joblib, tally_words_parallel, worker_count
from .synthetic import build_labeled_dataset
from .trainer import EnsembleModel, train_ensemble


def load_words(cache_dir: Path, *, force_download: bool = False) -> list[str]:
    return load_merged_words(cache_dir, force_download=force_download)


def build_freq_table(words: list[str], cache_dir: Path) -> FreqCounter:
    freq_path = cache_dir / FREQ_TABLE_NAME
    eligible = filter_words_for_freq(
        words,
        min_length=FREQ_MIN_WORD_LENGTH,
        max_words=FREQ_MAX_WORDS,
    )
    tally_words_parallel(eligible, freq_path, cpu_fraction=CPU_FRACTION)
    counter = FreqCounter()
    counter.load(freq_path)
    counter.set_lexicon(words)
    counter.save(freq_path)
    return counter


def is_bootstrapped(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    if not (
        (cache_dir / FREQ_TABLE_NAME).exists()
        and (cache_dir / ENSEMBLE_MODEL_NAME).exists()
        and (cache_dir / METADATA_NAME).exists()
    ):
        return False

    metadata = json.loads((cache_dir / METADATA_NAME).read_text(encoding="utf-8"))
    return metadata.get("version", 1) >= BOOTSTRAP_VERSION


def bootstrap(
    cache_dir: str | Path,
    *,
    samples_per_class: int = TRAIN_SAMPLES_PER_CLASS,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = BootstrapProgress(enabled=verbose)

    if is_bootstrapped(cache_dir) and not force:
        metadata = json.loads((cache_dir / METADATA_NAME).read_text(encoding="utf-8"))
        progress.skipped("model already trained (use --force to retrain)")
        return metadata

    if verbose:
        progress.banner()
        progress.sources([(source.label, source.url) for source in WORD_SOURCES])

    progress.step(1, "Downloading English word corpora from public sources")
    words = load_merged_words(
        cache_dir,
        force_download=force,
        on_progress=progress.detail if verbose else None,
    )
    progress.step_done(f"{len(words):,} unique words loaded")

    progress.step(2, "Validating corpus safety and quality")
    training_words = filter_words_for_training(words, min_length=FREQ_MIN_WORD_LENGTH)
    corpus_report = validate_words(
        training_words,
        min_length=3,
        training_min_length=3,
    )
    if not corpus_report.passed:
        raise RuntimeError(
            "Corpus validation failed: " + "; ".join(corpus_report.issues)
        )
    if verbose:
        progress.detail(
            f"{corpus_report.eligible_for_training:,} training-eligible words, "
            f"{corpus_report.unique_words_lower:,} unique (lower)"
        )
    progress.step_done("corpus validation passed")

    parallel_info = start_parallel(CPU_FRACTION)
    if verbose:
        progress.detail(
            f"Parallel backend: {parallel_info.get('backend', 'process')} "
            f"({parallel_info.get('process_workers', 0)} process / "
            f"{parallel_info.get('thread_workers', 0)} thread workers)"
        )

    try:
        progress.step(3, "Building bigram frequency table (english.freq)")
        build_freq_table(words, cache_dir)
        freq_words_used = len(
            filter_words_for_freq(
                words,
                min_length=FREQ_MIN_WORD_LENGTH,
                max_words=FREQ_MAX_WORDS,
            )
        )
        progress.step_done(f"{freq_words_used:,} words tallied")

        progress.step(
            4,
            f"Generating training samples ({samples_per_class:,} natural + "
            f"{samples_per_class:,} synthetic random)",
        )
        texts, labels = build_labeled_dataset(
            training_words,
            samples_per_class,
            real_word_ratio=NATURAL_REAL_WORD_RATIO,
            cpu_fraction=CPU_FRACTION,
        )
        progress.step_done(f"{len(texts):,} labeled samples ready")

        progress.step(5, "Extracting features and training ensemble (GridSearchCV + calibration)")
        freq_path = cache_dir / FREQ_TABLE_NAME
        model, metrics = train_ensemble(
            texts,
            labels,
            freq_path,
            cpu_fraction=CPU_FRACTION,
        )
        if verbose:
            progress.detail(
                f"accuracy={metrics.get('accuracy', 0):.3f}  "
                f"auc={metrics.get('auc', 0):.3f}  "
                f"brier={metrics.get('brier_score', 0):.4f}  "
                f"C={metrics.get('best_C', '?')}"
            )
        progress.step_done("ensemble trained")
    finally:
        stop_parallel()

    shutdown_joblib()

    progress.step(6, "Saving model, frequency table, and metadata")
    model.save(cache_dir / ENSEMBLE_MODEL_NAME)

    metadata = {
        "version": BOOTSTRAP_VERSION,
        "cache_dir": str(cache_dir),
        "natural_sources": [
            {"file": source.filename, "url": source.url, "label": source.label}
            for source in WORD_SOURCES
        ],
        "random_source": "synthetic (secrets, uuid, hex, base64 — generated locally)",
        "total_words_loaded": len(words),
        "training_words_available": len(training_words),
        "samples_per_class": samples_per_class,
        "natural_real_word_ratio": NATURAL_REAL_WORD_RATIO,
        "freq_words_used": freq_words_used,
        "cpu_workers": parallel_info.get("process_workers", 0),
        "thread_workers": parallel_info.get("thread_workers", 0),
        "parallel_backend": parallel_info.get("backend", "process"),
        "cpu_fraction": CPU_FRACTION,
        "corpus_validation": corpus_report.to_dict(),
        "metrics": metrics,
    }
    (cache_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    progress.step_done(f"artifacts saved to {cache_dir}")
    progress.finished(metadata)
    return metadata


def load_freq_counter(cache_dir: str | Path) -> FreqCounter:
    cache_dir = Path(cache_dir)
    freq_path = cache_dir / FREQ_TABLE_NAME
    if not freq_path.exists():
        raise FileNotFoundError(
            f"Freq table not found at {freq_path}. Run bootstrap first."
        )
    counter = FreqCounter()
    counter.load(freq_path)
    return counter


def load_ensemble(cache_dir: str | Path) -> EnsembleModel:
    cache_dir = Path(cache_dir)
    model_path = cache_dir / ENSEMBLE_MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found at {model_path}. Run bootstrap first."
        )
    return EnsembleModel.load(model_path)
