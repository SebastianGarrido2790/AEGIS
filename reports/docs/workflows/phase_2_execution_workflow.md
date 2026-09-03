# Phase 2 — Execution Workflow

**Product:** AEGIS | **Phase:** 2 of 9 | **Version:** 1.0.0 | **Date:** 2026-08-25
**Status:** 🟡 Ready to execute
**Authority:** `phase_2_implementation_plan.md` v1.0.0 (D1-D8, all approved)
**Scope of this document:** sequencing only, no code.

**Purpose:** translate the approved Phase 2 decisions (`phase_2_implementation_plan.md`, recorded as ADR-011 through ADR-018 in `system_design.md`) into a concrete, sequential build order. No code is written in this document — each stage describes _what_ gets built and _how it's verified_, not the implementation itself.

**Rule for every stage:** the gate at the end of a stage must fully pass before the next stage begins. If a gate fails, work stays in that stage until it passes; nothing downstream gets touched in the meantime — same discipline `phase_1_execution_workflow.md` closed on.

**Standing rule, from Stage 1 onward:** lint, type-check, and the module-size checker are re-run at the end of every stage, exactly as in Phase 1. Not repeated in every gate description below to avoid redundancy — assume it unless a stage says otherwise.

**One divergence worth naming plainly before starting:** P2-D3b's approved outcome (a single Tweedie GLM on pure premium, ADR-013) went the opposite direction from what I'd recommended (frequency-severity decomposition). That's a legitimate call, not an oversight — a single Tweedie model on pure premium is itself a recognized actuarial convention, not a deviation from one, just a different one than the Roadmap's exact phrasing suggested to me. Building against the approved decision, not re-litigating it.

Author: Sebastián Garrido Arévalo (execution and drafting) | Date: August 25, 2026

---

## Execution Flow Overview

```
Stage 0 (Pre-Flight Audit) ──► Stage 1 (Analytics Core & Engine) ──► Stage 2 (Wire Schemas & Serialization)
                                                                               │
                                                                               ▼
Stage 5 (Cockpit Modal & UI) ◄── Stage 4 (API Route Delivery) ◄── Stage 3 (Grounded Narrative Synthesis)
          │
          ▼
Stage 6 (Integration Tests & Quality Gate Audit)
```

---

## Stage 0 — Pre-Implementation Verification & Dependency Check

**Implements:** groundwork for ADR-011 through ADR-018; no ADR is _built_ in this stage, but several are only safely buildable once this stage's findings are resolved.

**Objective:** confirm the ground this phase is about to build on is actually solid — Phase 1's foundation, the new dependencies, and the real dataset's actual shape — before any Phase 2-specific code exists to build on top of assumptions that turn out wrong.

**Steps:**

1. **Regression-check Phase 1.** Re-run the existing CI pipeline, both GX suites against their fixtures, and `dvc repro` end-to-end. Phase 2 builds directly on this foundation; confirm it hasn't drifted since Phase 1 closed.
2. **Declare and install every Phase 2 dependency from ADR-018/P2-D8** — `mlflow`, `scikit-learn`, `statsmodels`, `econml`, `dowhy`, `pandas` (now explicit, closing audit finding A.2.2), `fastapi`, `jinja2`, `uvicorn`. This closes audit finding A.2.1 (MLflow referenced but never declared) as literally the first action of the phase, before anything downstream needs it installed.
3. **Confirm the OpenML/Kaggle dataset's actual structure** against the schema the Phase 1 `elasticity_training_suite.json` GX suite already expects. See the finding below — this is the step that surfaces it.
4. **Draft the license/attribution text** (ODbL, per P2-D1c's resolution) for the evaluation report, so it exists before Stage 6 needs it rather than being reconstructed under time pressure later.

**Finding this stage exists to catch — column schema mismatch:** freMTPL2's actual published columns (`IDpol`, `ClaimNb`, `Exposure`, `Area`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Density`, `Region` in the frequency table; `IDpol`, `ClaimAmount` in the severity table) do not match the column names the Phase 1 GX suite was written against (`policy_id`, `driver_age`, `veh_age`, `exposure`, `claim_count`, `claim_amount`, `premium`). Two problems, not one:

- **A naming mismatch**, fixable with a straightforward rename at ingestion (`IDpol`→`policy_id`, `DrivAge`→`driver_age`, `VehAge`→`veh_age`, `ClaimNb`→`claim_count`, `Exposure`→`exposure`, `ClaimAmount`→`claim_amount`). The GX suite's `exact_match=false` column check tolerates real freMTPL2's _extra_ columns (`Area`, `VehPower`, `VehBrand`, `VehGas`, `Density`, `Region`) without issue — those aren't the problem.
- **A genuine absence, not fixable by renaming:** the suite requires `annual_mileage` and `premium` to be present and non-null. Neither exists in freMTPL2 at all. `annual_mileage` has no equivalent field in this dataset — `Density` is a related but distinct concept (policyholder area population density, not distance driven) and should not be silently treated as a substitute. `premium` isn't an observed field in this dataset either — real actuarial case studies using freMTPL2 typically compute pure premium as frequency × severity rather than reading a charged premium directly, since the dataset was never designed to include one.

**Confidence note:** I'm quite confident in freMTPL2's actual column structure — it's a heavily referenced benchmark dataset — but this is exactly the kind of assumption this project's own conventions say to verify against the real source before building on it, not take from memory.

**Recommendation, not a silent fix:** this needs an explicit decision, not a rename buried in ingestion code. Likely resolution: adopt the rename mapping above, treat `premium` as a derived field computed during feature engineering (Stage 2) rather than an ingested one, and formally amend the `annual_mileage` requirement out of the elasticity suite — as its own dated ADR (ADR-019), not a quiet edit to a suite that already passed its own Phase 1 gate under a different assumption.

**Gate 0 — must pass before Stage 1 begins:** ✅ **PASSED (2026-08-26)**

- Phase 1's CI, both GX suites, and `dvc repro` are all still green — no drift since Phase 1 closed.
- All Phase 2 dependencies are installed and reflected in `uv.lock`.
- The column-mismatch finding above is explicitly resolved — the rename mapping, the `premium` derivation approach, and the `annual_mileage` suite amendment are each a conscious decision on record (ADR-019), not something Stage 1 discovers mid-ingestion and papers over.
- The ODbL attribution text is drafted and ready for Stage 6.

---

## Stage 1 — Dataset Ingestion & DVC Integration

**Implements:** ADR-011.

**Objective:** get the real dataset into the same DVC-tracked, GX-gated pattern Phase 1 already proved out on fixtures — now proving that pattern against real data for the first time.

**Steps:**

1. Fetch freMTPL2freq and freMTPL2sev once via `fetch_openml`, save to `data/raw/`.
2. Apply the column rename mapping and `premium` derivation resolved in Stage 0.
3. DVC-track the resulting local file, extending `dvc.yaml` with a new ingest stage for the real dataset — additive to, not a replacement for, the Phase 1 fixture-based stages, which continue to exist for CI's fast, self-contained GX-gate testing.
4. Run the (now-amended, per ADR-019) elasticity suite against the real ingested data.

**Gate 1 — must pass before Stage 2 begins:** ✅ **PASSED (2026-08-26)**

- The real dataset, post-mapping, passes the amended GX suite cleanly — this is the first genuine real-world test of a suite that, until now, had only ever seen hand-crafted fixtures.
- `dvc repro` reproduces the real-data ingest stage without a live network dependency on a second run (OpenML is only touched once, at first fetch).
- No fetch step exists anywhere except the one-time ingest — every subsequent stage in this phase reads the DVC-tracked local file, never re-fetches.

---

## Stage 2 — Feature Engineering Pipeline

**Implements:** ADR-012.

**Objective:** the shared, deterministic transformation logic every downstream model and the eventual inference path both depend on.

**Steps:**

1. Build the decomposed transformers under `src/aegis/pipelines/feature/` — exposure normalization, driver risk factors, vehicle factors, each its own module.
2. Build the single shared transformation function that both training (Stage 3–4) and the eventual FastAPI inference path (Stage 7) will call — no duplicated logic between them.
3. Implement the grouped, policy-level train/test split, using the `policy_id` column confirmed in Stage 0/1.
4. Derive the `premium` field per Stage 0's resolution.

**Gate 2 — must pass before Stage 3 begins:** ✅ **PASSED (2026-09-01)**

- The shared transformation function is deterministic — invoking it twice on identical input produces identical output.
- The grouped split has zero `policy_id` overlap between train and test sets — checked directly, not assumed.
- Feature pipeline output passes a schema check (expected columns present, correct types) before any model touches it.

---

## Stage 3 — GLM Baseline

**Implements:** ADR-013.

**Objective:** the actuarial reference point every later comparison is measured against — this number has to exist and be trustworthy before Stage 4's "outperforms baseline" claim is even checkable.

**Steps:**

1. Fit a single Tweedie GLM via `statsmodels` on pure premium (Stage 2's derived field), over the feature set from Stage 2.
2. Extract native `statsmodels` confidence intervals on the fitted parameters.
3. Compute and record the baseline calibration metric that Stage 4 will need to beat.

**Gate 3 — must pass before Stage 4 begins:** ✅ **PASSED (2026-09-02)**

- The GLM fits without convergence errors or degenerate output.
- Parameter confidence intervals are finite and computable — not NaN, not degenerate.
- The baseline calibration metric is recorded somewhere Stage 4 can reference it directly, not re-derived from memory later.

---

## Stage 4 — Causal Elasticity Model

**Implements:** ADR-014.

**Objective:** the phase's actual differentiator — and the stage where P2-D4b's validation framing has to be built correctly, not asserted after the fact.

**Steps:**

1. Construct the synthetic `treatment_rate_change` variable with a deliberately chosen, segment-varying, known elasticity function — this is causal-validation-specific construction, not general feature engineering, which is why it lives here rather than in Stage 2.
2. Fit `CausalForestDML` (EconML) on Stage 2's feature set plus the synthetic treatment.
3. Run the DoWhy refutation suite: placebo treatment refuter, random common cause refuter, data subset refuter, at minimum.
4. Check whether the estimator recovers the known synthetic elasticity function, per segment, within a defined margin.

**Gate 4 — must pass before Stage 5 begins — this is the Roadmap's own Phase 2 exit criterion, made concrete and checkable here rather than deferred to the end:**

- The recovered treatment-effect estimates correlate with the known synthetic ground truth above a defined threshold, per segment — proof the estimator works correctly on this data, framed explicitly as _estimator validation_, not a real-world elasticity finding (per P2-D4b's resolution — this framing has to show up in Stage 6's report too, not just here).
- All three DoWhy refuters run to completion and produce a non-trivial result — none silently skipped or swallowed.
- The causal model's calibration outperforms Stage 3's recorded GLM baseline with defensible confidence intervals — the literal PRD §11 requirement, checked directly against Stage 3's number.

---

## Stage 5 — MLflow Tracking & Registry Integration

**Implements:** ADR-015.

**Objective:** make Stage 3 and Stage 4's results reproducible, versioned, and queryable — not just files that happened to get produced once.

**Steps:**

1. Configure the `sqlite:///mlflow.db` tracking URI and `./artifacts/mlflow/` artifact location.
2. Re-run Stage 3 and Stage 4's training under MLflow run tracking — parameters, metrics, and artifacts logged, including the DoWhy refutation results as a structured JSON artifact attached to the causal model's run.
3. Register both models under their approved names: `aegis-glm-baseline`, `aegis-causal-elasticity`.

**Gate 5 — must pass before Stage 6 begins:**

- Both models are queryable from the MLflow Model Registry by name — not just visible as experiment runs, which the default file store would have limited to (the exact gap ADR-015 exists to close).
- The registered causal model version has the DoWhy refutation JSON attached and retrievable as a run artifact.
- Re-running the tracked training reproduces the same logged metrics within floating-point tolerance — a lightweight reproducibility check, consistent with this project's CI reproducibility discipline everywhere else.

---

## Stage 6 — Evaluation Report

**Implements:** ADR-016.

**Objective:** the first real occupant of the "Evals" pillar — and the place where every framing decision made so far (the ODbL attribution, the validation-not-discovery distinction) has to actually show up in writing, not just live in this planning document.

**Steps:**

1. Write `reports/docs/evaluations/phase_2_evaluation_report.md`: dataset source and ODbL attribution (Stage 0), GLM baseline calibration (Stage 3), causal model validation results and DoWhy refutation summary (Stage 4), explicitly framed as estimator validation against a known synthetic answer, not a real-world elasticity discovery.
2. Attach an identical copy to the corresponding MLflow run(s) as an artifact.

**Gate 6 — must pass before Stage 7 begins:**

- The report contains every element the Roadmap's exit criterion names — calibration, treatment-effect confidence intervals — plus the dataset attribution and the explicit validation-not-discovery framing.
- The `docs/evaluations/` copy and the MLflow-attached copy are checked identical — no drift between the two required copies.

---

## Stage 7 — Showcase Interface Foundation

**Implements:** ADR-017, ADR-018.

**Objective:** the first live, human-facing request/response path in this entire project — the first stage where latency and graceful failure under bad input are real concerns, not offline/batch ones.

**Steps:**

1. Scaffold `src/aegis/api/` — FastAPI app entrypoint, routes, Jinja2 templates.
2. Build the segment-varying preset scenario view (per P2-D7c) — each preset pulling its elasticity estimate and confidence interval from the **registered MLflow model** (Stage 5), not a freshly retrained one, keeping training and serving deliberately separate from this first inference moment onward.
3. Render interactive charts via Chart.js (CDN, per ADR-018); the static evaluation report (Stage 6) keeps its `matplotlib` figures separately.
4. Apply explicit "demo, not production" labeling on every route, per the Roadmap's exit criterion and the original UI-D1 decision.

**Gate 7 — must pass before Stage 8 begins:**

- Every preset scenario renders the correct elasticity value and confidence interval — cross-checked by hand against Stage 4/5's actual output, not just "the page loads without an error."
- The demo label is visible on every route reachable from the interface, not only the landing page.
- A deliberately malformed or missing preset request is handled gracefully — this project's first live HTTP endpoint, and the first place a bad request needs to degrade cleanly rather than crash the process.

---

## Stage 8 — Full Phase 2 Regression & Falsification Pass

**Implements:** the phase's overall exit criterion, closing the same way Phase 1 did.

**Objective:** prove every gate built across Stages 0–7 is load-bearing, and that nothing from Phase 1 quietly broke along the way.

**Steps:**

1. Re-run the complete Phase 1 CI pipeline (lint, type-check, module-size, GX gates, `dvc repro`, `pytest`) — confirm it's still green with Phase 2's additions layered on top.
2. Add Phase 2's own tests to the suite: feature-pipeline determinism, GLM fit success, causal-model recovery-of-ground-truth, MLflow registry existence, showcase-route smoke test.
3. **Deliberate falsification**, one failure per new gate: break feature-pipeline determinism, force a GLM convergence failure, force the causal model below the baseline-beating threshold, remove a model registration, break a showcase route — confirm each independently fails the pipeline, then revert and confirm green again.

**Gate 8 — Phase 2's actual exit criterion:**

- PRD §11's "causal model outperforms the GLM baseline with defensible confidence intervals" is demonstrated via Stage 4's and Stage 8's checks, not merely asserted in a report.
- The showcase slice renders correctly and is unambiguously labeled as a demo, not a production view.
- Full CI is green end-to-end, including every new Phase 2 gate.
- The falsification pass confirms every new gate actually blocks on the failure it claims to catch.

---

## Summary Table

| Stage | Implements       | Gate proves                                                                                        | Blocks until gate passes |
| ----- | ---------------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| 0     | Pre-flight       | Phase 1 foundation intact, dependencies installed, real-dataset schema mismatch resolved (ADR-019) | Stage 1                  |
| 1     | ADR-011          | Real data DVC-tracked and passes the (amended) GX suite for the first time                         | Stage 2                  |
| 2     | ADR-012          | Deterministic, leakage-safe feature pipeline                                                       | Stage 3                  |
| 3     | ADR-013          | Trustworthy GLM baseline number to beat                                                            | Stage 4                  |
| 4     | ADR-014          | Estimator recovers known ground truth; beats GLM baseline (PRD §11)                                | Stage 5                  |
| 5     | ADR-015          | Both models genuinely Registry-queryable, not just tracked                                         | Stage 6                  |
| 6     | ADR-016          | Report and MLflow copies complete and identical                                                    | Stage 7                  |
| 7     | ADR-017, ADR-018 | Showcase interface correct, labeled, and fails gracefully                                          | Stage 8                  |
| 8     | Phase exit       | Every gate is load-bearing under deliberate falsification                                          | Phase 3                  |
