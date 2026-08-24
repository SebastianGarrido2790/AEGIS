# Phase 2 — Implementation Plan & Decisions

**Status: DRAFT — awaiting Sebastián's review and approval.**
This is a living document. Decisions marked *No User Input Required* are recorded here for completeness and ADR discipline, but do not block progress. All other decisions require explicit approval before any Phase 2 code is written. Once a decision is approved, its outcome is recorded as a new ADR in `system_design.md`, and this entry is marked `RESOLVED` in place rather than deleted, so the deliberation trail is preserved — same convention as `phase_1_implementation_plan.md`.

Author: Sebastián Garrido Arévalo (decisions), Claude (drafting) | Date: August 22, 2026

---

## Part A — Current State Audit

Honest inventory of Phase 2's scope, based on what's been built, reviewed, and reported in this project to date. I have no live access to the actual repository — this reflects Phase 1's gate reports and remediation record, corrected where I've verified specifics against uploaded files. Flag anything this misses.

### A.1 — Phase 2 deliverables: what exists vs. what doesn't

| Artifact (per Roadmap Phase 2) | Status | Detail |
|---|---|---|
| Elasticity dataset (real, e.g. freMTPL2) | **Missing — by design** | Phase 1 deliberately deferred real dataset acquisition to Phase 2 (P1-D4c). Only hand-crafted fixtures exist. |
| Feature engineering pipeline | **Missing** | `src/aegis/pipelines/feature/` doesn't exist yet as working code — only named in AGENTS.md §5's target tree. |
| GLM baseline model | **Missing** | `src/aegis/pipelines/training/` doesn't exist yet. |
| Causal elasticity model (CausalForestDML) | **Missing** | Same. |
| Confounding sensitivity analysis | **Missing** | Not started. |
| MLflow tracking/registry integration | **Missing — and currently undeclared as a dependency** | See A.2, finding 1. |
| Evaluation report | **Missing** | `reports/docs/evaluations/` — one of the Six Pillars — has been empty since it was named. This phase is its first real occupant. |
| Showcase-interface slice (ADR-009) | **Missing — and its prerequisite doesn't exist either** | See A.2, finding 3: no FastAPI serving app has been built at all yet. |
| `tier1_ml` config section | **Exists, Phase 1 placeholder** | Present in `params.yaml`/`schema.py` with defaults per ADR-005; see A.2, finding 2 for a scope note. |
| `src/aegis/schemas/` | **Exists, empty** | `__init__.py` only — no model I/O schemas defined yet. |
| `tests/unit`, `tests/integration`, `tests/evals` | **Exist, mostly empty** | Only the one Phase 1 config-loading test is populated. |

**Bottom line:** unlike Phase 1, this isn't a pure from-scratch scaffold — it inherits a working config/data-contract/CI foundation. But every Phase 2-specific deliverable is genuinely new work, and one of them (the serving layer) is new *infrastructure*, not just new logic sitting in existing structure.

### A.2 — What has a latent bug, or a gap the Roadmap's own wording doesn't quite cover

1. **MLflow is referenced but never declared.** AGENTS.md §4 lists `uv run mlflow ui` as a standing command, and the Roadmap names MLflow as a Phase 2 deliverable — but `pyproject.toml`'s dependency list (`pydantic`, `pyyaml`, `great-expectations`, `dvc`) doesn't include `mlflow`, `scikit-learn`, `econml`, or `dowhy`. None of these were ever actually added; they were named in planning docs, not installed. Not a bug in the sense of broken code — a gap between what's documented as the stack and what's actually a declared dependency.

2. **`pandas` is used in Phase 1 code but never explicitly declared.** `data_contracts.py` does `import pandas as pd` directly. It currently works only because `great-expectations` pulls in `pandas` as a transitive dependency. This is fragile: if GX ever drops or re-pins its own pandas dependency, AEGIS's own explicit `import pandas` breaks with no warning from `pyproject.toml` — the project is relying on someone else's dependency graph for a library it uses directly. Worth fixing regardless of Phase 2, since Phase 2 will lean on pandas heavily.

3. **The FastAPI serving layer doesn't exist yet, at all.** Phase 1 built a CLI entrypoint (`aegis.pipelines.cli`) for DVC stages — it never stood up the "Serving" component named in `system_design.md`'s architecture diagram. ADR-009 commits to a Phase 2 showcase-interface slice built on FastAPI + Jinja2, but there's no `src/aegis/api/` package, no app entrypoint, no templates directory — none of it is in AGENTS.md's current §5 Project Structure tree either. This phase has to stand up the serving layer for the first time, not just add a feature to it.

4. **The Roadmap's own Phase 2 deliverable line has a scope inconsistency.** It reads: *"a first showcase-interface slice (ADR-009) exposing elasticity/bandit output for the phase's preset scenarios."* The bandit doesn't exist until Phase 3. A Phase 2 slice cannot show bandit output. This is addressed directly in Decision P2-D7b below, not silently corrected.

5. **`tier1_ml.exploration_corridor_width`** sits inside the config section this phase actually needs (`Tier1MLConfig`), but it's a bandit-exploration parameter — Phase 3 scope, not Phase 2. Not a bug (nothing depends on it incorrectly today), but worth a conscious call in Decision P2-D8 rather than leaving it looking like Phase 2 scope by proximity.

---

## Part B — Decision Log Summary

| ID | Decision | Constraint(s) it resolves | Status |
|---|---|---|---|
| P2-D1 | Elasticity dataset — source & ingestion | Cost, licensing | **Needs approval** (D1a: no input needed) |
| P2-D2 | Feature engineering pipeline design | Modularity | **Needs approval** |
| P2-D3 | GLM baseline design | Modularity, defensibility | **Needs approval** |
| P2-D4 | Causal elasticity model design | Modularity, methodological validity | **Needs approval** (D4a: no input needed) |
| P2-D5 | MLflow tracking & registry | Cost, modularity | **Needs approval** (D5b: no input needed) |
| P2-D6 | Evaluation report artifact | Modularity | **Needs approval** |
| P2-D7 | Showcase interface foundation | Latency, modularity | **Needs approval** |
| P2-D8 | New dependencies | Cost, modularity | **Needs approval** (mostly mechanical — see below) |

**On latency, explicitly:** Phase 1 had no live request path, so latency wasn't a real constraint. Phase 2 changes that — the showcase interface (P2-D7) is the first component in this entire project that serves a live HTTP response to a human waiting on it. Every decision touching that interface weighs latency for the first time; decisions confined to offline training/evaluation still don't need to.

**On the harness, explicitly:** several decisions below check a proposed approach against AGENTS.md directly — its §5 Project Structure (what's already named as the target tree), its Non-Negotiable Invariants (INV-8 modularity, INV-10 no live external dependency), and its §9 Evaluation, Calibration & Monitoring (which already anticipates this phase's evaluation report needing a home). Where a decision would require editing AGENTS.md itself, that's flagged explicitly rather than assumed.

---

## Part C — Decisions

### P2-D1 — Elasticity Dataset: Source & Ingestion

#### P2-D1a — Which dataset? *(No user input required)*

Already decided in `canvas.md` §6: the French Motor Third-Party Liability claims dataset (freMTPL2). This phase implements that choice; it doesn't reopen it. Recorded here for completeness of the decision record, not because there's a live alternative to weigh.

#### P2-D1b — Acquisition mechanism

- **Option A — `sklearn.datasets.fetch_openml`.** freMTPL2's frequency and severity tables are published on OpenML and fetchable directly through scikit-learn's OpenML loader, no separate package or API key required.
- **Option B — The dedicated `openml` Python package.** More control over caching and versioning, but adds a dependency whose only job scikit-learn's loader already does.
- **Option C — Manual one-time download**, instructions in a README, raw file DVC-tracked afterward (mirroring the Phase 1 `ingest` stage pattern exactly).

**Trade-off:** Option A is the lowest-dependency path and reuses a library already in the stack (scikit-learn, needed for the GLM baseline regardless). Option B's extra control isn't worth a whole additional dependency for a one-time fetch. Option C is more manual but removes any runtime network dependency from the pipeline itself — closer in spirit to INV-10's no-live-external-dependency posture, since `fetch_openml` does reach out to a live external service at fetch time (once, not per-run, but still a network call the current pipeline has never needed before).

**Recommendation:** Option A for the initial fetch, immediately followed by Option C's pattern — fetch once, then DVC-track the resulting local file exactly like the Phase 1 fixtures, so every subsequent `dvc repro` run is fully offline and reproducible without depending on OpenML's availability. This gets Option A's low friction without leaving a live network dependency baked into the ongoing pipeline. **Confidence note:** I'm reasonably but not fully confident on the exact current OpenML dataset name/version for freMTPL2 — verify against OpenML directly at implementation time rather than hardcoding an assumed ID.

#### P2-D1c — License and attribution check

Not a design choice — a due-diligence step that has to happen before ingestion, not after. freMTPL2 is a widely used academic/actuarial benchmark dataset, and my working assumption is that its license permits this kind of educational/portfolio use, but I'm not at the confidence level this project holds itself to for stating that as fact. **Recommendation:** confirm the exact license terms on OpenML's dataset page before the `ingest` stage is built, and record the confirmed license in the evaluation report's data-sources section (P2-D6).

---

### P2-D2 — Feature Engineering Pipeline Design

#### P2-D2a — Module granularity

- **Option A — One monolithic feature-builder function/module.**
- **Option B — Decomposed per feature-group** (e.g., separate transformers for exposure normalization, driver risk factors, vehicle factors), following AGENTS.md's established modularity posture (INV-8) even though a single feature file wouldn't come close to the 1,000-line ceiling at this phase's scope.

**Trade-off:** A monolithic module is faster to write today and genuinely wouldn't violate INV-8 at this size. But this project's own harness treats decomposition as a design-quality stance, not just a line-count workaround (§6 Coding Conventions: "decompose before extending, not after"), and feature logic is exactly the kind of thing that grows fastest once the causal model's confounding checks start demanding new derived features.

**Recommendation:** Option B, decomposed by feature group from the start — consistent with the project's own stated philosophy rather than deferred until a file gets large.

#### P2-D2b — Feature-store abstraction vs. shared transformation function

- **Option A — A formal feature-store abstraction** (versioned feature definitions, a registry, point-in-time correctness guarantees).
- **Option B — A single, deterministic, versioned transformation function** invoked identically at training and inference time — no separate store, just one shared code path.

**Trade-off:** A feature store is a legitimate pattern at real production scale with many models sharing many features and multiple consumers querying point-in-time-correct values. At this project's actual scale — one dataset, one feature set, two models trained together in one phase — it's meaningful infrastructure overhead solving a problem this project doesn't have yet, and risks exactly the kind of scope inflation the project has flagged as a standing risk since Charter §9.

**Recommendation:** Option B. This also directly satisfies AGENTS.md §10's training-serving parity requirement more simply than a feature store would — one function, imported by both the training pipeline and (eventually) the inference path, is a stronger parity guarantee than two systems that are supposed to agree.

#### P2-D2c — Train/test split methodology

- **Option A — Naive random row split** (`test_size` from `params.yaml`, currently 0.20).
- **Option B — Grouped/policy-level split**, ensuring no single policy's records appear in both train and test sets if the dataset has repeated policy-year observations.

**Trade-off:** freMTPL2, like most multi-year motor claims datasets, is structured with one row per policy-year — meaning the same policyholder can appear multiple times across years. A naive random split risks leaking information about a specific policyholder's risk profile across the train/test boundary, inflating apparent model performance in a way that wouldn't hold on genuinely unseen policyholders. This is a real methodological risk, not a style preference — it's exactly the kind of leakage the project's own INV-3 data-contract discipline exists to catch on the training side, just showing up here on the evaluation side instead.

**Recommendation:** Option B, grouped by policy ID, once the dataset's actual structure is confirmed at ingestion time. **Confidence note:** I'm inferring freMTPL2's policy-year structure from how this dataset family is typically published, not from having the file in hand — confirm the actual column structure before committing to the grouping key.

---

### P2-D3 — GLM Baseline Design

#### P2-D3a — Library choice, driven by the "defensible confidence intervals" exit criterion

- **Option A — `statsmodels` GLM** (Poisson/Gamma/Tweedie family). Native, mature statistical inference — `summary()`, `conf_int()` — built for exactly this kind of defensible-CI requirement.
- **Option B — scikit-learn's `PoissonRegressor`/`GammaRegressor`/`TweedieRegressor`**, with confidence intervals obtained via a separate bootstrap resampling procedure written on top.

**Trade-off:** AGENTS.md §3 Stack currently just says "scikit-learn (GLM baseline)" without specifying how confidence intervals get produced — that line was written before this exit criterion's exact demands were pinned down. Option B keeps everything in one library family already named in the stack, but bootstrapping adds real implementation and runtime cost (hundreds of refits) to get what Option A produces natively as a matter of course. Option A means AGENTS.md's Stack line needs a one-line addition (`statsmodels` for the GLM baseline specifically), not a wholesale change.

**Recommendation:** Option A, `statsmodels`. The exit criterion explicitly demands defensible confidence intervals — reach for the tool built for that, rather than bolting inferential statistics onto a library that wasn't designed to produce them. **Flagged:** if approved, AGENTS.md §3 needs `statsmodels` added alongside scikit-learn.

#### P2-D3b — Frequency-severity decomposition vs. single pure-premium model

- **Option A — Separate frequency and severity GLMs** (claim frequency via Poisson, claim severity via Gamma), combined as pure premium = frequency × severity — the traditional actuarial pricing decomposition.
- **Option B — A single Tweedie GLM** directly on pure premium (loss cost), skipping the two-model decomposition.

**Trade-off:** Option B is faster to build and evaluate — one model, one calibration check. Option A is more work but is the actual "actuarial-standard reference point" the Roadmap's own Phase 2 goal names explicitly, and gives more interpretable diagnostics (a calibration problem in frequency vs. severity are different problems with different fixes, indistinguishable in a single blended Tweedie output — the same reasoning that drove the four-metric split in ADR-010, applied here to actuarial modeling instead of RAG evaluation).

**Recommendation:** Option A, frequency-severity decomposition, specifically because the Roadmap goal names "actuarial-standard reference point" as the target, and a blended single-model approach doesn't fully earn that description.

---

### P2-D4 — Causal Elasticity Model Design

#### P2-D4a — Estimator choice *(No user input required)*

Already decided: `params.yaml`'s `tier1_ml.causal_estimator` field is already set to `"CausalForestDML"`, and Canvas/Charter name it explicitly as the estimator for segment-level heterogeneous treatment effects — which is the actual point of using a forest-based estimator over a simpler linear one, given the project's core value proposition is elasticity *varying by segment*, not a single population-average number. Recorded for completeness; not reopened here.

One clarification worth stating explicitly: DoWhy is not an alternative to EconML's `CausalForestDML` — it's a complementary tool for causal graph specification and refutation testing, used alongside the estimator, not instead of it. See P2-D4c.

#### P2-D4b — Treatment-variable construction (the hardest decision in this document)

This is the one place where I'd push back hardest on treating this as a routine implementation detail. freMTPL2 is an **observational** dataset — it has no built-in, ground-truth "we changed this policy's rate and observed the effect" experiment. The `treatment_rate_change` column already required by Phase 1's `elasticity_training_suite.json` GX suite has to come from somewhere, and how it's constructed determines whether the entire causal estimation exercise is measuring something real or something circular.

- **Option A — Synthetic treatment injection with known ground-truth elasticity.** Simulate a rate-change variable with a deliberately chosen, known elasticity function (e.g., retention probability shifts by a defined, segment-varying amount per unit of simulated rate change), then check whether `CausalForestDML` recovers that known function from the data. This is a standard causal-ML validation technique — you can't validate correctness against real-world ground truth you don't have, but you can validate that the estimator recovers a ground truth you constructed yourself.
- **Option B — Natural/quasi-experimental variation.** Use whatever premium variation already exists across policy-years in the historical data (driven by real-world rating-factor changes over time) as a proxy treatment, and estimate elasticity from that naturally occurring variation.

**Trade-off:** Option B sounds more "real" but is substantially harder to validate — there's no ground truth to check the estimate against, and any confounding in *why* premiums varied historically (which is likely extensive and unobserved in a public dataset never designed for this purpose) directly contaminates the estimate, which is precisely the risk Charter §9 already named as this system's biggest causal-inference risk. Option A can't claim to estimate the *real* historical elasticity of this specific dataset — it's a controlled validation exercise, not a real-world finding — but it produces something genuinely checkable: does the estimator recover a known answer.

**Recommendation:** Option A for this phase, explicitly framed as *estimator validation*, not *real elasticity discovery* — the evaluation report (P2-D6) should say exactly that, not imply this produces a real-world elasticity finding. Option B's natural-variation approach is the right approach for an actual production deployment with real historical rate-change data, which this public-dataset-constrained portfolio project doesn't have and isn't claiming to have (consistent with existing non-goals). Framing this correctly in every downstream artifact matters — overclaiming what a semi-synthetic validation exercise proves would undercut the project's own credibility more than the limitation itself would.

#### P2-D4c — Confounding sensitivity analysis method

- **Option A — DoWhy's standard refutation suite**: placebo treatment refuter, random common cause refuter, and a data-subset refuter, at minimum.
- **Option B — A custom, hand-written sensitivity analysis** (e.g., manual Rosenbaum-bounds-style checks).

**Trade-off:** Option B is more implementation work to build correctly and is exactly the kind of custom logic this project's own conventions push away from when a well-supported library already does the job (AGENTS.md §6: wrap deterministic logic in a Tool rather than reinventing it inline — the same instinct applies to reinventing a causal refutation method DoWhy already implements and has tested).

**Recommendation:** Option A. Run at minimum the placebo-treatment and random-common-cause refuters, log both to MLflow as run artifacts (see P2-D5c) so every registered causal model version carries its own sensitivity analysis, not a separately floating report.

---

### P2-D5 — MLflow Tracking & Registry Integration

#### P2-D5a — Tracking backend store

- **Option A — Default local file store** (`./mlruns`).
- **Option B — Local SQLite-backed tracking URI** (`sqlite:///mlflow.db`), no server process required.
- **Option C — A running `mlflow server` process** (local or containerized), backed by SQLite or another database.

**Trade-off, and a real technical constraint, not just a preference:** MLflow's Model Registry — explicitly named as a Phase 2 requirement — is **not supported on the default file store**. Option A would silently fail to deliver half of "MLflow experiment tracking and model registry integration," discovered only once someone tries to register a model and it doesn't work. Option C adds a persistent server process to operate for a solo-practitioner project with no other consumer of a live tracking UI — overhead with no corresponding benefit here. Option B gets full Registry support with zero added operational surface: no server to start, stop, or keep running, fully consistent with the Local-First posture already established for DVC (ADR-007).

**Recommendation:** Option B, `sqlite:///mlflow.db`, local file, gitignored.

#### P2-D5b — Artifact storage location *(No user input required)*

Local filesystem (e.g., `./artifacts/mlflow/`). Given INV-10 and the zero-cost, local-first posture already established project-wide, there's no live alternative worth weighing here — this is the same reasoning already applied to DVC's remote in Phase 1 (P1-D5a), just for a different tool.

#### P2-D5c — Registration and logging conventions

Two small conventions worth deciding explicitly rather than improvising ad hoc once code is being written:

- **Model naming:** register the two models under distinct names — e.g. `aegis-glm-baseline` and `aegis-causal-elasticity` — rather than one shared registered model with version tags trying to distinguish two structurally different model types.
- **Sensitivity analysis as a run artifact:** the DoWhy refutation results from P2-D4c are logged as a structured artifact (JSON) attached to the same MLflow run that registers the causal model version — so a future reader pulling up any registered causal model version can find its sensitivity analysis attached directly, not cross-referenced to a separate document that could drift out of sync.

**Recommendation:** both as stated above — this is mostly a naming/logging-discipline call, not a major trade-off, included for completeness before code gets written rather than decided implicitly mid-implementation.

---

### P2-D6 — Evaluation Report Artifact

#### P2-D6a — Format and location

- **Option A — A Markdown report** under `reports/docs/evaluations/`, following the same documentation conventions as every other planning artifact in this project.
- **Option B — A Jupyter notebook.**
- **Option C — Rely solely on what's attached to the MLflow run**, with no separate durable document.

**Trade-off:** Option B is well-suited to interactive exploration but is a weaker durable artifact for the "Evals" pillar — notebooks diff badly in git and mix code with narrative in a way this project's documentation conventions have deliberately avoided everywhere else. Option C ties the report tightly to a specific run but gives it no independent, versioned home outside MLflow — and `reports/docs/evaluations/` has been an empty, named pillar since the Six Pillars structure was adopted, with nothing yet written to justify its existence.

**Recommendation:** Option A as the durable, versioned record — and also attach a copy to the corresponding MLflow run as an artifact, so both audiences are served: the docs pillar for anyone reading the project's history, the MLflow-attached copy for anyone auditing a specific run's provenance. This is the natural first occupant of the "Evals" pillar named back in the Six Pillars structure.

---

### P2-D7 — Showcase Interface Foundation (ADR-009, First Slice)

#### P2-D7a — Where does the FastAPI app live?

- **Option A — A new `src/aegis/api/` package** (app entrypoint, routes, Jinja2 templates), matching how `system_design.md`'s architecture diagram has always named "Serving" as its own distinct component, separate from `tools/`, `agents/`, and `governance/`.
- **Option B — Fold it into `src/aegis/tools/`.**

**Trade-off:** Option B avoids adding a new top-level package, but `tools/` is specifically defined (AGENTS.md §5) as "standalone deterministic FastAPI microservices" in the tool-serving sense — internal capabilities exposed to agents, not a human-facing demo interface. Conflating the two would blur a distinction the project has held cleanly since Phase 0.

**Recommendation:** Option A. **Flagged:** this requires adding `src/aegis/api/` to AGENTS.md §5's Project Structure — it isn't in the current tree at all, since the tree was drawn before ADR-009 existed.

#### P2-D7b — Correcting the Roadmap's premature "bandit" mention

The current Roadmap Phase 2 deliverable line names "elasticity/bandit output" — but the bandit is Phase 3 scope (Canvas §9's own build order confirms this: elasticity model, *then* the bandit layer wraps it). A Phase 2 slice can only show what Phase 2 actually built.

**Recommendation:** correct `technical_roadmap.md`'s Phase 2 deliverable line to read "elasticity output" only, once this decision is approved — not silently, and not left inconsistent in the interim.

#### P2-D7c — What the preset scenarios actually show

- **Option A — Arbitrary variety** (a few unrelated example policies, chosen for coverage rather than for making a specific point).
- **Option B — Segment-varying presets specifically chosen to demonstrate heterogeneous treatment effects** — e.g., "urban young driver," "rural experienced driver," "high-mileage commuter" — each showing a *different* elasticity estimate and confidence interval.

**Trade-off:** Option A is easier to assemble but doesn't actually demonstrate anything specific to this project's modeling choice. The entire reason `CausalForestDML` was chosen over a simpler linear estimator (P2-D4a) is that elasticity is expected to vary meaningfully by segment — Option B is the only choice that puts that specific, real differentiator on screen, consistent with UI-D1c's own reasoning (the showcase interface exists to make the system's actual differentiator visible, not to show generic output).

**Recommendation:** Option B.

---

### P2-D8 — New Dependencies

Mostly downstream of the decisions above, not independent judgment calls — listed here for completeness so nothing gets installed ad hoc mid-implementation.

- **ML/causal/tracking (mechanical, given P2-D3a/P2-D4a/P2-D5a):** `mlflow`, `scikit-learn`, `statsmodels`, `econml`, `dowhy`, `pandas` (declared explicitly now, closing audit finding A.2.2 — no longer relying on it as a transitive dependency of `great-expectations`).
- **Serving (mechanical, given P2-D7a):** `fastapi`, `jinja2`, `uvicorn`.
- **Charting — a genuine decision, not mechanical:**
  - **Option A — `matplotlib`** for both the static Markdown evaluation report and the showcase interface's charts (server-rendered PNGs embedded in Jinja2 templates).
  - **Option B — split tooling:** `matplotlib` for the static evaluation report; a lightweight client-side JS charting library (e.g., Chart.js via CDN, no new Python dependency) for interactive charts inside the showcase interface itself.

  **Trade-off:** Option A is simpler — one charting stack, no JavaScript. Option B costs a little template complexity but makes the showcase interface genuinely interactive rather than a page of static images, which matters specifically because the entire point of building a showcase interface (UI-D1) was to be more engaging than a document — a page of static plots doesn't clear that bar much better than the evaluation report markdown already does on its own.

  **Recommendation:** Option B. Keep `matplotlib` for the durable evaluation report (no client dependency needed for a document read in git), and use Chart.js for the showcase interface specifically, where interactivity is the actual point.
- **`tier1_ml.exploration_corridor_width` scope note:** leave the field in place in `params.yaml` — it's harmless as an unused placeholder and removing it now just to re-add it in Phase 3 is churn without benefit. Flagged here so it's a conscious non-action, not an oversight.

---

## Part D — What Happens After Approval

1. Approved decisions are recorded as new ADRs in `system_design.md` (continuing from ADR-010).
2. AGENTS.md §3 (Stack), §5 (Project Structure) are updated for `statsmodels`, `src/aegis/api/`, and the new `evaluation`-adjacent dependencies, per the decisions flagged above.
3. `technical_roadmap.md`'s Phase 2 deliverable line is corrected per P2-D7b.
4. This document's entries stay marked `RESOLVED` in place, not deleted.
5. Only then does implementation begin — no code has been written as part of this document, per your instruction.
