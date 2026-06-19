"""Bootstrap pipeline for LRD-Hybrid (LM + PMI + gradient-boosted ensemble)."""

from __future__ import annotations

import json
from pathlib import Path

from ..bootstrap import bootstrap as bootstrap_production, is_bootstrapped, load_freq_counter
from ..config import (
    CPU_FRACTION,
    FREQ_TABLE_NAME,
    TRAIN_SAMPLES_PER_CLASS,
)
from ..corpora import filter_words_for_training, load_merged_words
from ..parallel import parallel_map, start_parallel, stop_parallel, worker_count
from ..synthetic import build_labeled_dataset
from .hybrid_features import HybridFeatureVector, extract_hybrid_features
from .hybrid_trainer import HybridEnsembleModel, train_hybrid_ensemble
from .ngram_lm import CharacterNgramLM
from .pmi_model import WordPMIModel

HYBRID_VERSION = 1
HYBRID_LM_NAME = "hybrid_lm.pkl"
HYBRID_PMI_NAME = "hybrid_pmi.pkl"
HYBRID_ENSEMBLE_NAME = "hybrid_ensemble.pkl"
HYBRID_METADATA_NAME = "hybrid_metadata.json"


def is_hybrid_bootstrapped(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    if not (
        (cache_dir / HYBRID_LM_NAME).exists()
        and (cache_dir / HYBRID_PMI_NAME).exists()
        and (cache_dir / HYBRID_ENSEMBLE_NAME).exists()
        and (cache_dir / HYBRID_METADATA_NAME).exists()
    ):
        return False
    metadata = json.loads((cache_dir / HYBRID_METADATA_NAME).read_text(encoding="utf-8"))
    return metadata.get("version", 0) >= HYBRID_VERSION


def _extract_hybrid_chunk(args: tuple[list[str], str]) -> list[HybridFeatureVector]:
    texts, cache_dir = args
    from ..parallel import configure_worker_env

    configure_worker_env()
    freq = load_freq_counter(cache_dir)
    lm = CharacterNgramLM.load(Path(cache_dir) / HYBRID_LM_NAME)
    pmi = WordPMIModel.load(Path(cache_dir) / HYBRID_PMI_NAME)
    return [extract_hybrid_features(text, freq, lm, pmi) for text in texts]


def extract_hybrid_features_parallel(
    texts: list[str],
    cache_dir: str | Path,
    *,
    cpu_fraction: float = CPU_FRACTION,
    chunksize: int = 64,
) -> list[HybridFeatureVector]:
    cache_dir = Path(cache_dir)
    workers = worker_count(cpu_fraction)
    if workers <= 1 or len(texts) <= 1:
        freq = load_freq_counter(cache_dir)
        lm = CharacterNgramLM.load(cache_dir / HYBRID_LM_NAME)
        pmi = WordPMIModel.load(cache_dir / HYBRID_PMI_NAME)
        return [extract_hybrid_features(text, freq, lm, pmi) for text in texts]

    chunk_size = max(chunksize, max(1, len(texts) // (workers * 4)))
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    cache_str = str(cache_dir)
    jobs = [(chunk, cache_str) for chunk in chunks]
    parts = parallel_map(_extract_hybrid_chunk, jobs, cpu_fraction=cpu_fraction, chunksize=1)
    return [row for part in parts for row in part]


def bootstrap_hybrid(
    cache_dir: str | Path,
    *,
    samples_per_class: int = TRAIN_SAMPLES_PER_CLASS,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Train the full LRD-Hybrid stack:
      1. Ensure production freq table exists
      2. Fit character n-gram LM on natural corpus words
      3. Fit word PMI from natural training compounds
      4. Train calibrated gradient-boosting ensemble on hybrid features
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if is_hybrid_bootstrapped(cache_dir) and not force:
        return json.loads((cache_dir / HYBRID_METADATA_NAME).read_text(encoding="utf-8"))

    if not is_bootstrapped(cache_dir) or force:
        if verbose:
            print("[hybrid] Bootstrapping production freq table first…")
        bootstrap_production(cache_dir, samples_per_class=samples_per_class, force=force)

    words = load_merged_words(cache_dir)
    training_words = filter_words_for_training(words, min_length=3)

    if verbose:
        print(f"[hybrid] Training character LM on {len(training_words):,} words…")
    language_model = CharacterNgramLM(order=5, discount=0.75)
    language_model.train(training_words)
    language_model.save(cache_dir / HYBRID_LM_NAME)

    if verbose:
        print("[hybrid] Building labeled dataset for PMI + ensemble…")
    parallel_info = start_parallel(CPU_FRACTION)
    try:
        texts, labels = build_labeled_dataset(
            training_words,
            samples_per_class,
            cpu_fraction=CPU_FRACTION,
        )
        natural_texts = [text for text, label in zip(texts, labels) if label == 0]

        freq_counter = load_freq_counter(cache_dir)
        lexicon = freq_counter.lexicon

        if verbose:
            print("[hybrid] Fitting word PMI from natural compounds…")
        pmi_model = WordPMIModel()
        pmi_model.train_from_words(training_words)
        pmi_model.train_bigrams_from_texts(natural_texts, lexicon)
        pmi_model.save(cache_dir / HYBRID_PMI_NAME)

        if verbose:
            print(f"[hybrid] Extracting hybrid features for {len(texts):,} samples…")
        feature_rows = extract_hybrid_features_parallel(
            texts,
            cache_dir,
            cpu_fraction=CPU_FRACTION,
        )

        if verbose:
            print("[hybrid] Training gradient-boosted ensemble (GridSearchCV + calibration)…")
        model, metrics = train_hybrid_ensemble(feature_rows, labels, cpu_fraction=CPU_FRACTION)
        model.save(cache_dir / HYBRID_ENSEMBLE_NAME)
    finally:
        stop_parallel()

    metadata = {
        "version": HYBRID_VERSION,
        "model": "lrd-hybrid",
        "description": (
            "Linguistic Randomness Detector — Hybrid: statistical + lexical + "
            "character LM perplexity + segmentation PMI + HistGradientBoosting"
        ),
        "cache_dir": str(cache_dir),
        "freq_table": str(cache_dir / FREQ_TABLE_NAME),
        "samples_per_class": samples_per_class,
        "parallel_backend": parallel_info.get("backend", "process"),
        "feature_groups": list(model.active_groups),
        "metrics": metrics,
    }
    (cache_dir / HYBRID_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(
            f"[hybrid] Done — F1={metrics.get('f1', 0):.3f}  "
            f"AUC={metrics.get('auc', 0):.3f}  "
            f"artifacts in {cache_dir}"
        )
    return metadata


def load_hybrid_scorer(cache_dir: str | Path):
    from .hybrid_scorer import HybridScorer

    return HybridScorer(cache_dir, auto_bootstrap=False)
