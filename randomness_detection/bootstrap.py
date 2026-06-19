"""Bootstrap: download corpus, build freq table, LM, PMI, and train ensemble."""

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
    LM_MODEL_NAME,
    METADATA_NAME,
    NATURAL_REAL_WORD_RATIO,
    PMI_MODEL_NAME,
    TRAIN_SAMPLES_PER_CLASS,
    WORD_SOURCES,
)
from .corpora import filter_words_for_freq, filter_words_for_training, load_merged_words
from .corpus_validator import validate_words
from .ensemble_features import extract_ensemble_features
from .freq_model import FreqCounter
from .ngram_lm import CharacterNgramLM
from .parallel import (
    extract_ensemble_features_parallel,
    start_parallel,
    stop_parallel,
    shutdown_joblib,
    tally_words_parallel,
)
from .pmi_model import WordPMIModel
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
        and (cache_dir / LM_MODEL_NAME).exists()
        and (cache_dir / PMI_MODEL_NAME).exists()
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

        progress.step(4, f"Training character language model ({len(training_words):,} words)")
        language_model = CharacterNgramLM(order=5, discount=0.75)
        language_model.train(training_words)
        language_model.save(cache_dir / LM_MODEL_NAME)
        progress.step_done("language model saved")

        progress.step(
            5,
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

        natural_texts = [text for text, label in zip(texts, labels) if label == 0]
        freq_counter = load_freq_counter(cache_dir)
        lexicon = freq_counter.lexicon

        progress.step(6, "Fitting word PMI model from natural compounds")
        pmi_model = WordPMIModel()
        pmi_model.train_from_words(training_words)
        pmi_model.train_bigrams_from_texts(natural_texts, lexicon)
        pmi_model.save(cache_dir / PMI_MODEL_NAME)
        progress.step_done("PMI model saved")

        progress.step(7, "Extracting ensemble features and training classifier")
        feature_rows = extract_ensemble_features_parallel(
            texts,
            cache_dir,
            cpu_fraction=CPU_FRACTION,
        )
        model, metrics = train_ensemble(
            feature_rows,
            labels,
            cpu_fraction=CPU_FRACTION,
        )
        if verbose:
            progress.detail(
                f"accuracy={metrics.get('accuracy', 0):.3f}  "
                f"f1={metrics.get('f1', 0):.3f}  "
                f"auc={metrics.get('auc', 0):.3f}  "
                f"brier={metrics.get('brier_score', 0):.4f}"
            )
        progress.step_done("ensemble trained")
    finally:
        stop_parallel()

    shutdown_joblib()

    progress.step(8, "Saving model artifacts and metadata")
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
        "signals": [
            "bigram_frequency",
            "shannon_entropy",
            "deflate_compression",
            "lexical_segmentation",
            "character_language_model",
            "word_pmi",
        ],
        "ensemble": "HistGradientBoosting + calibration",
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


def load_language_model(cache_dir: str | Path) -> CharacterNgramLM:
    cache_dir = Path(cache_dir)
    path = cache_dir / LM_MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Language model not found at {path}. Run bootstrap first."
        )
    return CharacterNgramLM.load(path)


def load_pmi_model(cache_dir: str | Path) -> WordPMIModel:
    cache_dir = Path(cache_dir)
    path = cache_dir / PMI_MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(f"PMI model not found at {path}. Run bootstrap first.")
    return WordPMIModel.load(path)


def load_ensemble(cache_dir: str | Path) -> EnsembleModel:
    cache_dir = Path(cache_dir)
    model_path = cache_dir / ENSEMBLE_MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found at {model_path}. Run bootstrap first."
        )
    return EnsembleModel.load(model_path)
