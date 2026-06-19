# Benchmark Methodology

How the **quality** of the randomness-detection algorithm is measured, and why each
benchmark is designed the way it is. Every benchmark uses **real data, real tools,
and no mocks**.

## Why four separate benchmarks

A single accuracy number is easy to fool. A model can score perfectly on data drawn
from its own training generator and still fail on real input. To prevent that
self-deception, quality is measured at four independent layers, each one harder to
game than the last:

| Layer | Benchmark | Data origin | Main question |
|-------|-----------|-------------|---------------|
| 1 | Quality benchmark | Synthetic (held-out) + external tools | Is the algorithm better than established baselines on a controlled split? |
| 2 | Robustness stress test | Hand-curated real-world strings | Does it generalize to strings it could never have memorized? |
| 3 | Real-world data test | Large public datasets (Tranco, malware DGA) | Does it work on the actual production job? |
| 4 | Throughput + CPU-scaling benchmarks | Real corpus + live API | Is it fast enough, and does it actually use all the cores? |

Layers 1–3 measure **detection quality**; layer 4 measures **speed**. The key design
principle: a result only counts if it holds on data that did **not** come from the
training pipeline. Layers 2 and 3 are the ones that catch overfitting.

---

## 1. Quality benchmark — `test_quality_benchmark.py`

**Goal:** prove the production scorer beats every established baseline under a fair,
identical evaluation protocol.

### Competitors

| Method | Source | What it is |
|--------|--------|------------|
| `randomness_detection` | product | The production logistic-regression ensemble |
| `freqpy` | external (git clone) | [Mark Baggett freq.py](https://github.com/MarkBaggett/freq), trained on the same `words_alpha.txt` corpus |
| `ent` | external (`apt install ent`) | Fourmilab Shannon-entropy CLI |
| `deflate_cli` | external (subprocess) | Standalone raw-DEFLATE compression-ratio script (`benchmark_tools/deflate_score.py`) |
| `freq` | internal | Bigram-frequency signal alone |
| `entropy` | internal | Normalized Shannon entropy alone |
| `compression` | internal | Raw DEFLATE ratio alone |
| `avg3` / `max3` | internal | Mean / max of the three internal signals |

External tools are installed/downloaded automatically and **smoke-tested** before use
(each must score a known random string higher than a known natural one, or the
benchmark aborts). This guarantees the baselines are genuinely working, not stubs.

### Dataset construction

The hold-out set is built from the corpus, then split so the corpus words used for
evaluation are disjoint from the first half:

- The eligible corpus (alpha words, length 5–32) is sorted and **only the second
  half** is used, so evaluation words are a fixed, reproducible slice.
- 6 categories are generated with a fixed seed (`2026`):
  - **natural (label 0):** 800 single corpus words + 400 two-word compounds (joined by `""`, `-`, or `_`)
  - **random (label 1):** 400 hex, 400 url-safe tokens, 400 lowercase alnum, 400 consonant-only strings

### Fair-comparison protocol (the important part)

Every method is evaluated **identically** so no method has a home-field advantage:

1. **Calibration / test split** — 35% of samples are a calibration split, 65% a test split.
2. **Per-method threshold tuning** — each method's decision threshold is swept over
   `1..99` on the **calibration split** and the threshold that maximizes F1 is locked
   in. This is critical: methods live on different score scales (entropy ≈ 90–100 for
   everything; the ensemble ≈ 1 vs 99). Giving each its own best threshold removes
   scale bias and compares the methods on their *discriminative power*, not on whether
   50 happens to be a good cut for them.
3. **Metrics on the held-out test split only** — accuracy, precision, recall, F1,
   ROC-AUC, false-positive rate (on naturals), false-negative rate (on randoms), and
   the mean score per class.

ROC-AUC is threshold-independent, so it shows raw separation power regardless of where
the cut lands.

### Pass criteria

```
randomness_detection F1 >= every baseline's F1   AND   ROC-AUC >= 0.95
```

Output: a ranked table + `quality_benchmark_results.json`. Expected `QUALITY CHECK: PASS`.

### How to read it

- **F1 / ROC-AUC** → overall quality. The product should rank `1/9`.
- **FPR** → how often real words are wrongly called random.
- **FNR** → how often random tokens slip through as natural.
- **Natμ / Rndμ** → mean score per class; a large gap means clean separation.

### Limitation (acknowledged honestly)

This benchmark's samples share a *generator family* with training (corpus words +
crypto tokens), so a high score here is **necessary but not sufficient**. That is
exactly why layers 2 and 3 exist.

---

## 2. Robustness stress test — `test_robustness.py`

**Goal:** catch root-cause regressions and overfitting that layer 1 is structurally
blind to. This is the **anti-band-aid guardrail**.

### Methodology

Every string is **hand-curated real-world data** that is **never** produced by
`synthetic.py`. Because the model could not have memorized this distribution, passing
here means the model learned real linguistic structure rather than a shortcut.

Strings are grouped into buckets, scored with the production scorer, and a string is
called "random" when `score >= 50`:

| Bucket | Expectation | Role |
|--------|-------------|------|
| `core_natural` | low score (natural) | **pass/fail** — words, brands (`stackoverflow`, `wikipedia`), separator/camelCase identifiers (`getUserById`), digit-suffixed names (`report2024`) |
| `clear_random` | high score (random) | **pass/fail** — real UUIDs, git SHAs, base64, API keys, DGA-style labels |
| `hard_natural` | low score | diagnostic — short out-of-dictionary brands (`nvidia`, `figma`) |
| `adversarial_random` | high score | diagnostic — concatenated words / leetspeak (`boxcarmittenglow`, `p4ssw0rdz`) |
| `adversarial_natural` | low score | diagnostic — real-word passphrases (`correcthorsebatterystaple`) |

### Pass criteria

```
core_natural false-positive rate <= 10%   AND   clear_random false-negative rate <= 5%
```

The three "hard / adversarial" buckets are reported as **diagnostics only** — they are
inherently ambiguous (a short out-of-dictionary brand and a short pronounceable random
string are not reliably separable from structure alone), so gating on them would
invite band-aid fixes. They are printed so the structural ceiling is visible, not
hidden.

### Why it matters

This test is what caught the historical regression where adding hard-negative training
made the model learn "mixed-case + separator ⇒ random" and start flagging legitimate
identifiers. The fix had to make this test pass on data it could not memorize, which
forced a genuine root-cause solution rather than a patch.

---

## 3. Real-world data test — `test_real_world_data.py`

**Goal:** measure the actual production job on large public datasets that have nothing
to do with the training pipeline. This is the **strongest, least-gameable** layer.

### Datasets (downloaded + cached under `.benchmark_tools/realworld/`)

| Class | Source | Notes |
|-------|--------|-------|
| Legitimate | [Tranco top-1M](https://tranco-list.eu/) | popular, ranking-curated domains |
| DGA (malware) | [andrewaeva/DGA](https://github.com/andrewaeva/DGA) | 800K+ real malware domains; families 1/2/3/5/6/8 = random-character, families 4/7 = dictionary DGA |

### Methodology

1. **Score the registrable label.** Only the second-level domain is scored (TLD
   stripped, with a small two-level-suffix table so `foo.co.uk → foo`). The label is
   the unit a DGA actually randomizes.
2. **Sample** up to 6,000 labels per class with a fixed seed; DGA is sampled evenly
   across families so no single family dominates.
3. **Per-family recall** is reported separately, so the hard dictionary-DGA families
   are shown honestly instead of being averaged away.

### The noise problem and how it is handled

Traffic-ranked "legit" lists are intrinsically noisy: they contain many
machine-generated labels (CDN nodes, hashes, telemetry) that genuinely *are* random.
A naive false-positive rate would unfairly punish the model for being **correct** on
those. Two safeguards address this:

- **AUC on the solvable task** — separation between legit and *random-character* DGA
  only (dictionary DGA is the documented structural ceiling and is excluded from the
  gate).
- **Independent de-noising with a different codebase** — each "legit" label is also
  scored by the external `freqpy` tool. Labels that freqpy *also* judges random
  (`>= 95`) are dropped as contamination, and a **clean** false-positive rate and
  **clean AUC** are computed on the remainder. Using an independent tool to clean the
  ground truth avoids circular reasoning (the model is not used to excuse its own
  errors).

### Pass criteria

```
random-character DGA recall >= 95%   AND   clean solvable ROC-AUC >= 0.95
```

Raw and clean legit FP rates are printed for transparency but are **not** hard gates,
precisely because the "legit" label is contaminated.

### How to read it

- **per-family recall** → which DGA styles are caught; dictionary families (4, 7) are
  expected to be lower and that is reported, not hidden.
- **clean solvable AUC** → the honest, noise-robust quality number.
- **sample legit false positives** → printed so contamination is auditable by eye
  (most are obviously machine domains like `zx5pu9`).

---

## 4. Throughput benchmark — `test_benchmark.py`

**Goal:** verify the system is fast enough in every execution mode. Quality is
worthless if scoring is too slow to deploy.

### Phases (real model, real corpus, live API subprocess)

| Phase | What it measures | Unit |
|-------|------------------|------|
| CLI batch scoring | single-process `Scorer.score_batch` | texts/s |
| Inference pool | 10s sustained load on the multi-worker pool | texts/s |
| Exclude pre-filter | 50,000 domain rules, 10,000 lookups (no ML) | checks/s |
| API `/score` | single-score HTTP throughput + p50/p95 latency | req/s |
| API `/score/batch` | batched HTTP throughput + p50/p95 latency | items/s |

The API server is started as a real subprocess on a dedicated port, health-checked,
load-tested with a thread pool, and torn down. Latency percentiles (p50/p95) are
recorded alongside throughput.

### Pass criteria

```
every phase completes with non-zero throughput and sample count
```

Output: a report table + `benchmark_results.json`. Expected `OVERALL: PASS`. Reference
numbers for a 48-core machine are in [the README benchmarks](README.md#benchmarks).

### 4b. Parallel and CPU-scaling verification

Script: `test_real_parallel.py`.

**Goal:** prove that under heavy concurrent load the work is actually distributed
across the configured number of CPU cores, instead of being serialized behind a lock.
Raw throughput (4) can look fine even when only one core is busy; this test inspects
the machine, not just the response times.

**Methodology (no mocks):**

1. Start a real API server as a subprocess with `RANDOMNESS_INFERENCE_WORKERS=N`.
2. Drive **15 s of sustained, mixed traffic** — single `/score` and `/score/batch`
   (41 items) requests built from 20,000 real corpus words — with a client thread pool
   sized `min(32, N × 2)`.
3. **Count live worker processes** as descendants of the server PID (`pgrep -P`,
   handling `forkserver` nesting), sampled repeatedly during load.
4. **Sample system-wide CPU** from `/proc/stat` every 0.25 s and report peak / avg.

**Pass criteria:**

```
worker processes >= 50% of configured   AND
(system CPU peak >= 15%  OR  avg >= 8%)  AND
>= 100 real requests completed
```

(The worker-process check is skipped for the `thread` backend, which scales on threads
rather than processes.)

**Scaling evidence.** Running the test across worker counts shows both signals rise
with the setting — this is the actual proof of core scaling:

| Configured workers | Live worker procs | System CPU peak |
|--------------------|-------------------|-----------------|
| 6  | 8  | 17.0% |
| 24 | 26 | 53.9% |
| 48 | 50 | 100%  |

A flat CPU curve or a worker count that does not track the setting would mean the
parallelism is broken — that is the failure this layer is designed to catch.

---

## Running everything

```bash
cd randomness_detection
# quality layers
PYTHONPATH=. .venv/bin/python test_quality_benchmark.py     # layer 1
PYTHONPATH=. .venv/bin/python test_robustness.py            # layer 2
PYTHONPATH=. .venv/bin/python test_real_world_data.py       # layer 3
# speed
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python test_benchmark.py             # layer 4
RANDOMNESS_INFERENCE_WORKERS=24 \
PYTHONPATH=. .venv/bin/python test_real_parallel.py         # layer 4b (CPU scaling)
```

All four require a trained model (`PYTHONPATH=. .venv/bin/python -m randomness_detection --bootstrap`).

## Design principles summary

1. **No mocks** — real corpus, real crypto tokens, real external tools, real datasets,
   real HTTP server.
2. **Independent ground truth** — layers 2 and 3 use data the model could not have
   memorized; contamination is cleaned with a *different* tool.
3. **Fair comparison** — every competing method gets its own calibrated threshold.
4. **Honest ceilings** — inherently ambiguous cases (dictionary DGA, short OOV brands,
   passphrases) are reported as diagnostics, never gamed away to inflate a number.
