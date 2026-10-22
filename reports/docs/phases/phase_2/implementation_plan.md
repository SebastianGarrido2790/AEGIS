# Phase 2 — Tier 1: Deterministic ML Baseline
## Implementation Plan & Architectural Decisions — AEGIS

Author: Sebastián Garrido Arévalo | Version 0.3 | Date: October 21, 2026
**Status: DECISIONS APPROVED — October 21, 2026.** The recommended decisions were approved in the four batches in §5. Qualification-test results Q1–Q3 remain open where specified below. Feature implementation remains subject to the approved SDD workflow and entry gates.

**What this document is:** the phase-level decision record. It answers: before any Phase 2 code is written, what must be decided, why, and what is recommended.
**Why it exists:** to bridge project intent, the actual current state, architectural decisions, and executable implementation, so that implementation starts from approved constraints and not from assumptions.
**How to use it:** §2 and §3 record evidence and conflicts; §5 records the approval batches; §6 records each approved decision and remaining test-gated outcomes. The Phase Execution Plan consumes these constraints and sequences implementation stages (§7).

Each decision carries one of three tags:

- **Approval required:** a real choice exists; the human decides.
- **No approval required:** *only one option exists under the applicable constraints*; recorded for completeness.
- **Unresolved — qualification test:** the answer depends on evidence a short, bounded test can produce; the test and its pass criteria are defined here, and the decision closes when the result is recorded.

### Revision log

| Version | Change | Reason |
|---|---|---|
| 0.1 | Initial draft | — |
| 0.2 | Added evidence from the EDA run (§2.2). Corrected dataset identity and column set (R11). Added exposure-semantics finding (R12). Replaced assumed cap candidates, segment factors and Q2 sizes with EDA-grounded ones. Removed manufactured alternatives; aligned decision fields, feature IDs and SDD depth with the phase-plan conventions; moved stage sequencing out to the Execution Plan. | EDA output received; planning-convention review |
| 0.3 | Recorded approval of recommended decisions in the four §5 batches; added approval highlights without changing rejected alternatives or trade-off text; aligned dependent documents; verified V9 against the locked MLflow version and current registry documentation. Q1–Q3 result-dependent choices remain open. | Human approval, October 21, 2026 |

### Basis and limits

Sources: project documents (scoping, PRD, user story, roadmap, system design ADRs, capability-profile gates, invariants register), the phase-plan template, repository inspection, and `artifacts/eda_summary.json`. The artifact is present and its dataset identity and principal counts match the plan; full regeneration and consistency verification remain part of V6. V1–V8 remain entry checks; V9 was verified on October 21, 2026 against `uv.lock` (MLflow 3.15.2) and current MLflow registry documentation.

---

## 1. Phase Context

| Field | Value |
|---|---|
| Phase | Phase 2 — Tier 1: Deterministic ML Baseline (reset; EDA-first redo) |
| Objective | Produce the deterministic foundation every later tier consumes: a calibrated GLM loss-cost baseline, a causal elasticity estimator validated against a known semi-synthetic ground truth, a promotable per-segment Tier 1 output, and a first showcase slice. |
| Governing requirements | PRD FR-1 (semi-synthetic causal benchmark), FR-9 (GX-gated ingestion); PRD §7 (reproducibility, modularity), §9 (data), §11 (elasticity KPI as amended); scoping §0 (amended success metric), §2 (goal item 1), §3. |
| Binding invariants | **INV-3** data contracts block versioning; **INV-8** 1,000-line module ceiling; **INV-5** binds recommendations (Phase 2 showcase carries a diagnostic-only exemption: ε and recovery metrics only, never a proposed Δ); **INV-10** no live data. Not exercised: INV-1, INV-2, INV-4, INV-6, INV-7, INV-9. |
| Applicable ADRs | ADR-003, 005, 006, 007, 008 (data contracts, config, DVC, CI); ADR-009 (showcase); ADR-011 (ε definition, Impact assumptions, Phase 2 exemption). |
| Entry conditions | (1) §2.7 checks recorded. (2) Open EDA gaps (§2.4) closed or explicitly accepted. (3) Batch 1 decisions approved, since they amend existing documents. (4) Phase 1 gates green on `main`. |
| Exit criteria (roadmap) | (E1) the causal estimator recovers the known synthetic treatment effect with defensible per-segment and rank-based metrics; (E2) the showcase slice renders correctly, labeled a demo, with the INV-5 exemption honored. |
| Exit criteria (operationalized here) | E1 closes only through the D6 gate with thresholds frozen under D6a before the final run. Added: E3 baseline calibrated with capped-loss treatment documented (D1, D1a); E4 a promotable segment-level artifact under the `approved` alias with a recorded human review (D11, D11a); E5 INV-3 ordering demonstrable in the DAG with failing fixtures for both elasticity-side suites (D8); E6 CI-smoke tier green and full run reproducible within the stated tolerance (D10). |
| Depends on | Phase 1 (scaffold, data contracts, CI). |
| Required by later phases | **Phase 3:** per-segment ε̂ with intervals, baseline retention, corridor parameter. **Phase 6:** the ε definition and conversion, per-segment loss cost and base premium consistent with each other, and a guaranteed bound on |ε|·Δ. **Phase 8:** the validation report as regression and drift baseline. |

### How the project constraints bound the decisions

| Constraint | Effect in Phase 2 | Decisions |
|---|---|---|
| Latency | Tier 1 is offline. The showcase reads a precomputed segment table, so serving latency is trivial. No SLA is set. | D11, D12 |
| Cost | No LLM calls; cost is local compute time on a single development machine. A runtime budget is therefore the binding cost. | D5, D10 |
| Modularity (INV-8) | Features split by responsibility (intake, features, GLM, DGP, estimator, validation, registry, showcase), each module under the ceiling. | §4 |
| Auditability | Truth isolation, frozen thresholds, lineage tags and a recorded review make results reconstructable. | D2, D6a, D11a |
| Solo capacity | Qualification tests replace speculative design; no manufactured alternatives; excluded scope is explicit. | §4, §5 |

---

## 2. Current State Audit

### 2.1 What exists

Evidence tags: **D** documented; **E** observed in the supplied EDA output; **R** reported in change reports, not independently verified; **U** unverified, confirm at entry (§2.7).

| Item | Evidence | Tag |
|---|---|---|
| Phase 1 foundations: namespaced `src/aegis/` package, `uv` toolchain, Python 3.12, domain-nested `params.yaml` validated by Pydantic at load, GX file-based suites with hand-crafted fixtures, DVC DAG (`ingest`, `validate_gx`, `version`) on a local remote, unified GitHub Actions workflow | ADR-004 to ADR-008 recorded as Validated | D |
| Test and lint state: 42 passed, 2 skipped; `ruff` clean; module-size check green across 26 Python files | Change reports | R |
| Phase 2 attempt 1 removed from `main`; preserved with its former ADR-011 to ADR-019 in branch `archive/phase2-attempt-1` | `system_design.md` header | D |
| `scripts/explore_raw_data.py` runs against `data/raw/freMTPLfreq.csv.zip` and `data/raw/freMTPLsev.csv` | EDA output | E |
| `params.yaml` `tier1_ml`: `test_size` 0.20, `random_state` 42, `causal_estimator: CausalForestDML`, `cv_folds` 5, `exploration_corridor_width` 0.15; `governance.auto_approval_corridor_pct` 0.05 | `params.yaml` as shared | R |
| Elasticity GX suite and fixtures (`elasticity_valid.csv`, `elasticity_invalid_leakage.csv`, `elasticity_invalid_range.csv`) | `data_contracts` block | D (names) / U (schema) |

**Archived attempt 1 is evidence, not a code source.** Any function it contains re-enters only through a feature in §4 and must pass new falsifiable tests. Its composite risk feature, single aggregate Pearson gate and silent-fallback refuters are not carried over.

### 2.2 What the EDA established

| Finding (E) | Implication | Decision |
|---|---|---|
| **Dataset identity:** 413,169 policies, 10 frequency columns, 16,181 severity records. This is the original French MTPL benchmark (`freMTPLfreq`/`freMTPLsev`), not freMTPL2. Rating factors present: Power (letter codes d–o), Brand, Gas, Region (10 levels), DriverAge, CarAge, Density. **BonusMalus and Area are absent.** | Dataset naming and rating-factor documentation corrected under D9; Phase 1 GX suite and fixtures may encode a different schema; segmentation cannot use BonusMalus. | D9, D8, D3, R11 |
| **Perfect key reconciliation:** unique PolicyIDs, 0 orphan severity records, 0 claim-count mismatches (15,390 claiming policies; 16,181 claims). | Raw contract can state exact reconciliation expectations; no cleaning rule needed for key mismatches. | D8 |
| **Sparse losses:** 96.28% of policies have zero claims; frequency 6.98% per policy-year; portfolio pure premium €148.67 per car-year. | MAE/RMSE on pure premium are poor calibration metrics. | D1 |
| **Heavy severity tail:** mean €2,130 vs median €1,156; skew 75, kurtosis 6,601; P99 €16,109, P99.5 €35,432, P99.9 €131,834; top claim €2.04M = 5.91% of losses; top 10 = 15.81%; top 1% of claims = 36.75%. Claims at €25k and above: 113 claims (0.70%) carry about 34.0% of losses; the 24 claims above €100k carry 21.59%. Derived from the tier table: capping at €100k removes about €5.04M (≈14.6% of losses; excess load ≈1.17); capping at €25k removes about 25.8%. | Capping is mandatory and the excess load is material, not cosmetic. | D1a |
| **Tail-looking severity relativities:** Power i mean severity €4,530 (pure-premium relativity 2.34×, 722 claims); DriverAge 18–24 mean severity €5,040 (5.26×, with frequency 2.22×). | Uncapped segment loss costs are likely distorted by single large claims; this must be tested, not assumed. | D1a (Q1) |
| **Exposure:** only 29.44% of policies have a full year; 16.16% are below 0.10 years; mean 0.56; 421 policies (0.102%) exceed 1.0, maximum 1.99. The EDA does not say why exposure is short. | Exposure cannot be read as a retention signal; the "annual renewal" outcome needs a stated unit. | D2a, D4, R12 |
| **Factor structure:** Region is highly concentrated (Centre 38.9% of policies; Limousin 1.1%); Density relativities rise monotonically (frequency 0.056 rural to 0.095 dense urban). | Region is too thin for a segment grid; log(Density) is the sensible transform. | D3, D4 |
| **Tautological check:** frequency × severity = pure premium "exact equality". This is arithmetic identity, not evidence of model validity. | Must not be used as a test. | D6 (discipline) |
| EDA labels the Poisson dispersion ratio 1.06 as "overdispersed (Negative Binomial indicated)". | Mild; method not shown; no Phase 2 decision depends on it. | — |

### 2.3 What is missing (known gaps against the Phase 2 outcome)

| Gap | Needed by |
|---|---|
| Phase 2 Tier 1 implementation on `main`: raw intake, feature/loss pipeline, GLM, DGP, estimator, validation, registry artifact | All exits |
| Retained Phase 2 showcase presentation/API and tests exist, but are not evidence that the Tier 1 training pipeline or approved segment artifact exists | E2; V2 |
| `params.yaml` and Pydantic keys: loss cap and excess load, segment bands and size floors, DGP parameters and seeds, thresholds, MLflow settings, premium loading | ADR-005 |
| Raw-table and training-table GX contracts matching the real schema | INV-3 |
| Segment definition and Tier 1 contract are approved in P2-D3/ADR-014 but not implemented in the pipeline | PRD FR-1/FR-2, Phases 3 and 6 |
| Numeric validation thresholds | Scoping §8 |

### 2.4 EDA gaps still open

The console output answers the key-integrity, tail and exposure questions. Still open:

| Gap | Feeds |
|---|---|
| Raw (unbinned) distributions for DriverAge, CarAge, Density and Power: minimum, maximum, percentiles. The output shows only binned views. | D4 tail handling |
| Which segment holds the €2.04M claim, and each segment's share of excess above the candidate caps | D1a (Q1 reports it) |
| Pairwise dependence among rating factors | D2b, D3 |
| Why exposure is short (observation-window truncation versus lapse) | D2a (decision made under uncertainty; see recommendation) |

### 2.5 Latent defects and inconsistent documentation

Routed to §3: R1 severity scope; R2 baseline metric; R3 unfalsifiable E1; R4 slice scope; R5 reproducibility wording; R6 contract coverage; R7 registry mechanics; R11 dataset identity and columns; R12 exposure semantics.

### 2.6 Obsolete artifacts and unresolved decisions

- Obsolete: attempt-1 code and ADR-011 to ADR-019 numbering on the archive branch; any Phase 1 elasticity fixture built for a freMTPL2 schema (U).
- At initial drafting, severity scope, segment definition, gate thresholds, registry contract, and estimator scale were unresolved. Decisions are now approved; Q1–Q3 outcomes and implementation remain open as recorded below.

### 2.7 Entry verification checklist

| # | Check | Informs |
|---|---|---|
| V1 | Phase 1 gates green on `main`; reproduce the 42 + 2 result | Entry condition |
| V2 | Contents of `src/aegis/pipelines`, `tools`, `schemas`, `api`, `bandit`; distinguish retained showcase code from Phase 2 Tier 1 training implementation | §2.3 |
| V3 | Schema encoded by the existing elasticity GX suite and fixtures versus the real column set | D8 |
| V4 | Current `dvc.yaml` stages; what `ingest` fetches and from where; whether `data/raw/` is DVC-tracked; resolve the stray `data/raw/elasticity_fremtpl2.csv` and `data/raw/elasticity_raw.csv` (confirm unused and remove, so the freMTPL files are the sole intake) | D8, D9 |
| V5 | `pyproject.toml`/lock: EconML, scikit-learn, statsmodels, MLflow, FastAPI, Jinja2 resolve on Python 3.12 under Windows | D5, D11, D12 |
| V6 | `eda_summary.json` regenerated and consistent with the console output; note both `artifacts/` and `data/` are gitignored (`.gitignore:42-43`), so either un-ignore the summary via a negation under `artifacts/*` and commit it, or record its checksum as local evidence | §2.2 |
| V7 | `archive/phase2-attempt-1` readable (reference only) | Audit |
| V8 | Development-machine cores and RAM | D5 budget |
| V9 | Pinned MLflow version and its registry/alias behavior — **verified:** `uv.lock` and `uv run` report 3.15.2; SQLite-backed registry check created an `approved` alias, resolved it, reassigned it to another version, and confirmed the new target. Current docs support the same API. D11a selects aliases. | D11a |

---

## 3. Phase Scope and Requirement Assessment

### 3.1 Findings

The table records issues as observed during plan drafting. Their approved resolutions are recorded in §5/§6 and ADR-012–ADR-015; empirical outcomes that still require Q1–Q3 remain open.

| ID | Finding | Sources | Impact if unresolved | Routed to |
|---|---|---|---|---|
| R1 | **Severity scope conflict.** PRD §4/§12: no severity modeling in v1. Scoping §6: frequency-severity decomposition is the baseline precedent. README: "Tweedie GLM". Roadmap: "exposure/frequency/severity feature engineering". Carryover: loss capping mandatory (now quantified, §2.2). Phase 6 Impact needs a per-segment loss cost. | PRD, scoping, README, roadmap | Baseline unspecifiable; Phase 6 has no Losses₀ source | D1, D1a |
| R2 | **Baseline calibration metric.** Scoping §9 names MAE/RMSE and self-consistent intervals. With 96.28% zero-loss policies MAE/RMSE are dominated by the zero mass; self-consistent intervals are an estimator property. | Scoping §9, EDA | Gate passes or fails for the wrong reasons | D1 |
| R3 | **E1 is not falsifiable.** "Defensible per-segment and rank-based confidence intervals" mixes a point statistic with interval language; no thresholds exist. | Roadmap, scoping | Sign-off becomes judgment | D6, D6a |
| R4 | **Slice scope.** ADR-009 says "elasticity/bandit output view"; the bandit is Phase 3; the INV-5 exemption forbids showing Δ. | ADR-009, roadmap, PRD §10 | Scope creep; INV-5 breach | D12 |
| R5 | **Reproducibility wording.** PRD §7 requires bit-for-bit reproducibility. | PRD §7, ADR-007/008 | Unachievable gate silently weakened | D10 |
| R6 | **Contract coverage.** ADR-003 names one elasticity suite; INV-3 covers every versioned dataset, here raw intake and the generated training table. | ADR-003, INV-3 | INV-3 violated at intake, or ADR applied loosely | D8 |
| R7 | **Registry promotion.** Gates Part 4.2 and existing guidance describe a stage-tag flip; current MLflow behavior must be verified. | Gates doc | Promotion built on a possibly deprecated mechanism | D11a |
| R8 | **ε conversion owed by Phase 2.** ADR-011 defines ε as %ΔQ/%ΔP converted from the retention effect τ; conversion needs baseline retention q₀. ADR-011 floors the volume factor at zero and notes ε cancels in the loss ratio (it moves revenue and the underwriting result, Rev − Losses). | ADR-011 | Phase 6 cannot be specified | D2a, D3 |
| R9 | **Premium and renewal absent from the data.** Revenue₀ must be synthetic and consistent with Losses₀. | PRD §9, ADR-011 | Phase 6 financials unanchored | D3 |
| R10 | **"Segment" undefined** in every document that uses it. | PRD, scoping | No shared unit | D3 |
| R11 | **Dataset identity and columns.** Documents say freMTPL2 and list bonus-malus; the data is the original freMTPL with seven rating factors and no BonusMalus or Area. | EDA vs PRD §6/§9, scoping §7, README, system design | Wrong schema in contracts and docs | D9, D8, D3 |
| R12 | **Exposure semantics.** Only 29.44% of policies have a full year; the "annual renewal" outcome is undefined for a policy observed for 0.05 years. | EDA, A1 | The DGP encodes an unexplained artifact as a causal factor | D2a |

### 3.2 Assumptions made explicit

- A1 (revised). The elasticity unit is the policy record; the outcome is renewal at the next term, applied to every record regardless of observed exposure.
- A2. The treatment is a relative price change Δ versus the technical premium, bounded by the exploration corridor (±0.15).
- A3. Competitor prices are not modeled; elasticity is own-price only.
- A4. Claims-experience variables (ClaimNb, ClaimAmount) belong to the baseline target, not to the elasticity covariates.
- A5. Results describe the semi-synthetic benchmark and are never presented as real-world elasticity discovery.

---

## 4. Candidate Feature Register (SDD specs)

**Candidate features are not approved scope.** Each becomes a spec only after its governing decisions are approved. IDs are unique and stable and are never reused or renumbered; the execution plan references them. "SDD depth" is a recommendation: *Full* means spec, plan and tasks; *Light* means the work stays in tasks within the relevant execution stage. Size is relative effort, not a schedule.

| ID (folder) | Feature | Need it serves | Decisions | Depends on | Primary acceptance evidence | Size | SDD depth |
|---|---|---|---|---|---|---|---|
| P2-F01 (`001_raw-intake-contract`) | Raw intake and raw-data contract | INV-3 at the first DAG stage | D8, D9 | — | Malformed fixtures halt the pipeline; exact reconciliation expectations hold; checksum mismatch fails | S | Light |
| P2-F02 (`002_feature-pipeline`) | Deterministic feature pipeline | One cleaning step for both models; heavy-tail handling | D4 | F01 | Identical output on identical input; every clip or exclusion logged with counts; no composite index; claims variables absent from elasticity covariates | M | Full |
| P2-F03 (`003_glm-baseline-loss-treatment`) | GLM baseline, cap policy and calibration report (includes Q1) | Baseline reference; Losses₀ source | D1, D1a | F02 | Q1 verdict; holdout deviance; actual-versus-expected by decile and segment; portfolio A/E within tolerance after excess load | M | Full |
| P2-F04 (`004_semi-synthetic-dgp`) | Semi-synthetic DGP and training-table contract | Known ground truth for E1; premium base | D2, D2a, D2b, D8 | F02, F03 | Self-tests pass (estimand gap, overlap, between-segment dispersion, within-segment homogeneity, |ε|·Δ bound); training-table suite fails a fixture with a truth column | L | Full |
| P2-F05 (`005_segments-tier1-contract`) | Segment definition and Tier 1 output schema | Shared unit for validation, Phases 3 and 6 | D3 | F02, F03 | Size floors met; ratio-of-sums aggregation tested; schema has no proposed-adjustment field | S–M | Full |
| P2-F06 (`006_causal-estimator`) | Estimator training and segment-level inference (includes Q2) | The model under validation | D5 | F04, F05 | Q2 verdict; segment intervals finite, self-consistent, and inference on the segment average | L | Full |
| P2-F07 (`007_validation-gate`) | Validation harness and gate (includes Q3) | E1 | D6, D6a | F06 | Negative controls fail; estimator passes across seeds; thresholds frozen before final seeds | M–L | Full |
| P2-F08 (`008_confounder-sensitivity`) | Hidden-confounder sensitivity analysis | Roadmap sensitivity requirement; PRD §14 Q1 | D7 | F04, F07 | Degradation curve and robustness value; harness raises on any skipped cell | M | Light |
| P2-F09 (`009_registry-promotion`) | MLflow tracking and promotion | Promotable artifact with recorded review | D11, D11a | F03, F06, F07 | `candidate` set by pipeline; `approved` only with a review record and lineage tags | M | Full |
| P2-F10 (`010_pipeline-config-ci-tiers`) | DAG extension, config expansion, CI tiers | INV-3 ordering, ADR-005, reproducibility | D8, D10 | evolves with F01–F09 | CI-smoke green with coverage floor; `dvc repro` reproduces data products identically and model outputs within tolerance | M | Light |
| P2-F11 (`011_showcase-slice`) | Phase 2 showcase slice | E2; secondary audience | D12 | F09 | Banner on every page; response schema has no Δ field; presets render from the `approved` artifact | M | Full |

**Deliberately excluded:** the bandit and any proposed Δ (Phase 3); any Tier 2 node, LLM call or Gateway use; a separate severity model (D1); free-form showcase input (ADR-009, INV-10); drift monitoring and the judge suite (Phase 8); real-world elasticity claims (A5); data-driven feature selection (the covariate set is fixed by the dataset and A4; any later selection uses training and validation splits only); hyperparameter search beyond Q2.

---

## 5. Decision Log Summary

| ID | Title | Parent | Scope / impact | Approval | Status |
|---|---|---|---|---|---|
| P2-D1 | GLM baseline target and severity scope | — | Amends PRD §4/§12, scoping §6/§9, README, roadmap wording | Required | **Approved — Batch 1** |
| P2-D1a | Large-loss capping policy | D1 | Losses₀ accuracy for Phase 6 | Required (policy); level by Q1 | **Policy and Q1 criterion approved — Batch 3; cap level unresolved — Q1** |
| P2-D2 | DGP data products and truth isolation | — | Leakage prevention | Not required | **Accepted as the sole viable approach** |
| P2-D2a | Treatment/outcome form, unit, truth and ε definition | D2 | Phase 6 contract; E1 validity | Required | **Approved — Batch 2** |
| P2-D2b | Confounding, overlap and heterogeneity design | D2 | E1 validity | Required | **Approved — Batch 2** |
| P2-D3 | Segment definition and Tier 1 output contract | — | Phase 3 and 6 interface | Required | **Approved — Batch 2; provisional band edges subject to §2.4 close-out** |
| P2-D4 | Feature pipeline and tail handling | — | Heavy-tail carryover | Required | **Approved — Batch 2; bounds follow raw-distribution check** |
| P2-D5 | Estimator configuration and training scale | — | Runtime budget; validation cost | Required (budget); config by Q2 | **Q2 method and proposed budget criteria approved — Batch 3; scale/configuration unresolved — Q2** |
| P2-D6 | Validation protocol and gate design | — | E1 | Required | **Approved — Batch 2** |
| P2-D6a | Gate thresholds | D6 | E1 | Required (method); values by Q3 | **Q3 calibration method approved — Batch 3; numeric thresholds unresolved — Q3** |
| P2-D7 | Confounding sensitivity method | — | PRD §14 Q1 | Required | **Approved — Batch 2** |
| P2-D8 | Elasticity-side data contracts and DAG ordering | — | Amends ADR-003 (INV-3) | Required | **Approved — Batch 1** |
| P2-D9 | Raw data identity, source and pinning | — | Document corrections; reproducibility | Not required | **Accepted as the factual correction — Batch 1** |
| P2-D10 | Reproducibility contract and execution tiers | — | Amends PRD §7 wording | Required | **Approved — Batch 1** |
| P2-D11 | What Phase 2 registers and promotes | — | Phase 3/6/showcase interface | Required | **Approved — Batch 4** |
| P2-D11a | Tracking backend and promotion mechanism | D11 | Amends gates Part 4.2 wording | Required | **Approved — Batch 1, V9 verified** |
| P2-D12 | Showcase slice scope and INV-5 enforcement | — | E2; ADR-009 clarification | Required (scenario set); enforcement not required | **Approved — Batch 4** |

### Recommended approval batches

| Batch | Decisions | Why this order |
|---|---|---|
| 1 — Requirement conflicts and document corrections | D1, D8, D10, D11a (plus D9 corrections) | They amend existing documents; nothing downstream can be specified until wording is settled |
| 2 — Benchmark design | D2a, D2b, D3, D4, D6, D7 | One coherent block defining what E1 means |
| 3 — Test-gated (approve method now, values later) | D1a, D5, D6a | Close when Q1, Q2, Q3 report |
| 4 — Delivery shape | D11, D12 | Depend on the artifact contract in D3 |

### Approval record

All four batches were approved on **October 21, 2026**, in the order above. The approval adopts each recommendation and its stated constraints; it does not predetermine empirical results.

| Order | Approved decisions | Recorded outcome |
|---|---|---|
| 1 — Requirement conflicts and document corrections | D1, D8, D10, D11a, D9 | Approved/accepted; V9 verified; dependent documents corrected |
| 2 — Benchmark design | D2a, D2b, D3, D4, D6, D7; D2 accepted as sole viable design | Approved/accepted; provisional bands and raw-data-dependent bounds retained |
| 3 — Test-gated | D1a, D5, D6a | Methods and stated criteria approved; Q1 cap level, Q2 scale/configuration, and Q3 numeric thresholds remain evidence-gated |
| 4 — Delivery shape | D11, D12 | Approved |

The approved default criteria and methods are retained as written. Options and trade-off text above remain unchanged for traceability.

### Qualification tests

| Test | Closes | Needs | Output |
|---|---|---|---|
| Q1 — Cap sweep | D1a | P2-F02, P2-F03 skeleton | Cap levels versus loss share removed, holdout deviance, decile calibration, segment loss-cost ranking stability, and excess share by segment |
| Q2 — Estimator scale | D5 | P2-F04, P2-F05, estimator prototype | Runtime, peak memory and recovery at four training sizes; nuisance-model comparison |
| Q3 — Threshold calibration | D6a | D6 harness skeleton, Q2 | Control and estimator metric distributions on pilot seeds; recommended thresholds to freeze |

---

## 6. Decisions

### P2-D1 — GLM baseline target and severity scope

**Question.** What does the baseline GLM predict, given that four documents disagree on severity and Phase 6 needs a per-segment loss cost?

**Context.** See R1 and R2. Of the 413,169 policies, 96.28% have no claim; 16,181 claims carry heavy-tailed severities (§2.2). The Impact node needs Losses₀ and a premium base consistent with it (R9). Attempt 1 reportedly moved toward a single Tweedie model; that is evidence of difficulty with the two-model route, not a requirement.

**Governing requirements / invariants.** PRD §4/§12 (non-goal wording), scoping §6/§9, ADR-011 (Impact inputs), carryover (capping mandatory).

**Options.**

| | A. Poisson frequency GLM only | B. Single Tweedie GLM on capped pure premium | C. Poisson frequency × Gamma severity |
|---|---|---|---|
| Matches PRD non-goal as written | Yes | No (wording amendment) | No |
| Yields per-segment loss cost | Only with a non-modeled severity constant | Directly | By product |
| Models to calibrate and review | 1 | 1 | 2; severity fitted on 16,181 claims, tail-dominated |
| Tail handling | Severity tail ignored | Cap plus excess load (D1a) | Cap inside severity model |

**Trade-offs.** A leaves Losses₀ dependent on a constant that ignores severity differences across segments. C is the textbook route but doubles the calibration surface for a baseline whose roles are "reference" and "Losses₀ source". B is the smallest model set that serves both roles and the carryover.

**Recommendation.** **B.** Reword PRD §4/§12 to: *no separate severity model and no severity-driven pricing in v1; loss cost enters only as the baseline GLM output.* The amendment must state plainly that the Tweedie GLM *is* the sanctioned combined frequency-severity model (compound Poisson-Gamma), so the reworded non-goal reads as *no separate severity model / no severity-driven pricing* — not as excluding severity from the loss-cost model. Align scoping §6, the README row and the roadmap capability text ("exposure and capped-loss feature engineering"). The Tweedie power is estimated or fixed with stated justification in the spec, not left as a default. Calibration is reported as holdout mean Tweedie deviance, actual-versus-expected by predicted-loss decile and by segment; MAE/RMSE are secondary. "Finite, self-consistent intervals" moves to the causal estimator.

> **APPROVED — Batch 1 (October 21, 2026):** Option B. A single capped-pure-premium Tweedie GLM provides the baseline loss cost; there is no separate severity model or severity-driven pricing in v1. Required document corrections are approved.

**Approval.** Approved, October 21, 2026 (Batch 1). **Decision status.** Approved.

**Consequences.** The baseline exposes loss cost per policy-year by segment, aggregated from GLM fitted values (D3). The frequency-times-severity identity is not used as a test.

---

### P2-D1a — Large-loss capping policy *(nested under D1)*

**Question.** At what level are losses capped, and how is the excess handled?

**Context.** Capping at €100k removes about 14.6% of losses (excess load ≈1.17); at about €25k, roughly 25.8%. A uniform load is only adequate if the excess is spread similarly across segments; the EDA suggests it may not be (Power i, DriverAge 18–24).

**Governing.** D1, carryover, Phase 6 dependency on Losses₀.

**Options (policy).**

| | A. Cap plus portfolio-level excess load | B. Cap without load | C. No cap |
|---|---|---|---|
| Stability against the tail | Good | Good | Poor (excluded by carryover) |
| Unbiased portfolio loss | Yes by construction | No (understates by 15–26%) | Yes, unstably |
| Added complexity | One scalar from the training split | None | None |

**Trade-offs.** B would silently understate Losses₀ by the excess share and bias every Phase 6 underwriting result; C is excluded. A costs one scalar and one report.

**Recommendation.** **A.** The cap level and load are computed on the training split only, stored in config and the artifact manifest, and reported with the loss share removed and the excess share by segment.

**Cap level: Unresolved — qualification test Q1.** Candidates from the EDA: P99 (€16,109), P99.5 (€35,432), €100,000, P99.9 (€131,834). For each, report loss share removed, claims affected, Tweedie fit stability, holdout deviance, decile calibration, and segment loss-cost ranking against the next-higher cap. **Pass:** the lowest-truncation level beyond which segment loss-cost Spearman against the next level is ≥ 0.98 and holdout deviance changes by ≤ 1% (proposed defaults, adjustable at approval). If the excess share by segment departs materially from the exposure share, the uniform load is revisited and returns to the decision workflow.

> **APPROVED — Batch 3 (October 21, 2026):** Option A and the stated Q1 comparison and pass criteria. Q1 still determines the cap level; material excess-share concentration by segment returns to decision review.

**Approval.** Policy and Q1 criteria approved, October 21, 2026 (Batch 3). **Decision status.** Approved; cap level unresolved until Q1 reports.

**Consequences.** Segment loss costs used downstream are capped and loaded, and labeled as such.

---

### P2-D2 — DGP data products and truth isolation

**Question.** How are the synthetic treatment, outcome and ground truth stored relative to what the estimator can read?

**Context.** The benchmark is valid only if the estimator cannot see the truth. ADR-003 already includes a post-treatment leakage check.

**Governing.** INV-3, ADR-003, ADR-006, PRD FR-1.

**Options.** One. Three physically separate products: (1) the estimator-visible training table (pre-treatment covariates X, treatment T, outcome Y, segment id); (2) a truth sidecar keyed by PolicyID (τᵢ, q₀ᵢ, εᵢ), read only by the validation harness; (3) a DGP manifest (parameters, seeds, version hash).

**Trade-offs.** Any layout in which truth is reachable from estimator inputs invalidates E1; no trade-off exists to weigh.

**Recommendation.** As stated. Enforcement: the training-table suite fails any table containing a truth column (D8), and a unit test asserts the estimator package cannot import the truth loader.

> **ACCEPTED — Batch 2 (October 21, 2026):** This is the sole viable approach under the applicable constraints. Keep training data, truth sidecar, and DGP manifest as separate products; enforce truth isolation as specified.

**Approval.** No approval required — sole viable approach accepted, October 21, 2026. **Decision status.** Accepted.

**Consequences.** D2a and D2b inherit this separation.

---

### P2-D2a — Treatment/outcome form, unit, truth and ε definition *(nested under D2)*

**Question.** What generates retention, what is the unit of observation, and how are τ, q₀ and ε defined so Phase 6 can consume them?

**Context.** Price and renewal are absent from the data (R9), so T and Y are synthetic. `CausalForestDML` fits a partially linear model. Phase 6 uses the first-order volume Q = Q₀·(1 + ε·Δ) with ε = τ/q₀ (floored at zero per ADR-011). Exposure is short for most policies (R12) and the EDA does not explain why.

**Governing.** ADR-011, PRD FR-1, D2, A1.

**Options (retention form).**

| | A. Linear-probability, truth exact | B. Logistic; truth = local slope at Δ = 0 | C. Logistic; truth = estimator's averaged-slope target |
|---|---|---|---|
| Realism (bounded probabilities) | Weak | Good | Good |
| Alignment with Phase 6 | Fair | Direct (first order at small Δ) | Indirect |
| Truth definition | Trivial | Closed form: τ = q₀(1 − q₀)·b(X) | Numerical integration per policy |
| Risk | Clipping distorts tails | Small gap between truth and estimator target | Diverges from what Phase 6 reads |

**Trade-offs.** B pays one self-test to bound the gap between its truth and the estimator's target; A is simpler but unrealistic; C measures something Phase 6 does not use.

**Recommendation.** **B**, with a self-test that quantifies the per-segment gap between truth and estimator target, kept below a tolerance stated in the P2-F04 spec. Fixed by this decision:

- **Unit:** the policy record; all 413,169 are included, with equal weight. Observed exposure is not a DGP covariate and does not define retention, because it would encode an unexplained observation artifact as a causal factor. Exposure is used only as baseline weight/offset and to express loss cost per policy-year.
- **Outcome:** renewal indicator at the next term. **Treatment:** relative price change Δ ∈ [−0.15, +0.15] (corridor parameter).
- τᵢ = ∂P(renew)/∂Δ at Δ = 0; q₀ᵢ = P(renew) at Δ = 0; εᵢ = τᵢ / q₀ᵢ.
- Segment aggregates use ratio of sums: ε_seg = Σ τᵢ / Σ q₀ᵢ, matching Phase 6, which applies the response to segment-level Q₀.
- **Bound:** max |εᵢ|·0.15 < 1, so the ADR-011 zero floor never binds in the benchmark. The ε range is a stated design choice in the spec, not a claim about real markets.

> **APPROVED — Batch 2 (October 21, 2026):** Option B and the listed unit, treatment, outcome, truth, ε conversion, aggregation, and bound. The self-test tolerance is specified in P2-F04.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved.

**Consequences.** If the EDA gap on exposure (§2.4) shows short exposure reflects lapse, the unit decision is revisited through change management.

---

### P2-D2b — Confounding, overlap and heterogeneity design *(nested under D2)*

**Question.** How is treatment assignment confounded, and how heterogeneous is the truth, so that recovery is a meaningful test?

**Context.** A randomized T makes recovery trivial; excessive confounding destroys overlap. ADR-011 assumes homogeneous lapse risk within a segment; a DGP that violates it would make Phase 6's loss projection systematically wrong. The EDA shows strong risk heterogeneity (novice frequency 2.22×), which makes a risk-based rate-change rule realistic.

**Governing.** D2, ADR-011, scoping §11, PRD §14 Q1.

**Options.**

| | A. Observed confounding through a risk-based rate-change rule plus residual noise, strength adjustable | B. Randomized T |
|---|---|---|
| Tests adjustment | Yes | No |
| Role | Main benchmark | Positive control only |

A third variant (hidden confounder) is reserved for D7.

**Recommendation.** **A** as the main benchmark; **B** retained only as a positive control inside the harness. Design requirements, each verified by a DGP self-test rather than asserted:

1. **Overlap:** residual variability of T given X stays above a stated floor.
2. **Heterogeneity:** truth is driven by at most 3–4 interpretable, banded or capped covariates with at least one interaction; no raw heavy-tailed variable enters the truth.
3. **Dispersion:** between-segment spread of τ exceeds the expected estimator standard error by a stated margin, so Spearman across segments is informative.
4. **Single confounding dial** in config.
5. **Homogeneity (ADR-011):** within a segment, εᵢ is not materially correlated with expected loss cost; the correlation tolerance is stated in the spec.

> **APPROVED — Batch 2 (October 21, 2026):** Option A is the main confounded benchmark; randomized treatment remains the positive control. The five DGP requirements and their self-test enforcement are approved.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved.

**Consequences.** Requirement 5 may constrain which covariates drive heterogeneity inside a band.

---

### P2-D3 — Segment definition and Tier 1 output contract

**Question.** What is a segment, and what does Tier 1 hand to Phases 3, 6 and the showcase?

**Context.** R10. The dataset has seven rating factors and no BonusMalus or Area (R11). Region has 10 levels with a 38.9%/1.1% spread; Power has 12; both would create thin cells. Only 16,181 claims exist, so raw segment loss sums are noisy.

**Governing.** PRD FR-1/FR-2, ADR-011, INV-5, D1, D2a.

**Options.**

| | A. Grid of actuarial bands on DriverAge × CarAge × Density with small-cell merging | B. One-dimensional bands | C. Estimator-derived leaves | D. Policy-level only |
|---|---|---|---|---|
| Interpretable and auditable | Yes | Yes | Partly | Yes |
| Independent of the estimator under test | Yes | Yes | **No (circular for validation)** | Yes |
| Resolves truth heterogeneity | Adequate | Too coarse | Adaptive | Fully |
| Usable by Phases 3/6 | Yes | Weak | Unstable | No |

**Trade-offs.** C is excluded because the validation unit cannot depend on the model being validated. D leaves Phase 3 and 6 without a review unit. B loses the interaction structure D2b requires.

**Recommendation.** **A.** Proposed grid (adjustable once §2.4 closes): DriverAge 4 bands (18–24, 25–44, 45–64, 65+), CarAge 3 (0–4, 5–9, 10+), Density 3 (≤500, 501–2,000, >2,000), giving 36 cells; Region, Power, Brand and Gas enter the GLM only. Band edges are frozen in config, not recomputed per run. Merging applies only where a cell falls below a 1,000-policy floor (the spec states the merge rule); with 36 near-equal cells the realized count is expected near the cell count, so 20-60 is a sanity range, not a separate target. **Segment loss cost is aggregated from GLM fitted values (capped, loaded), not from raw segment sums.**

**Tier 1 output contract (Pydantic).** Per segment: `segment_id`, policy count, exposure in policy-years, baseline retention q₀, τ̂, ε̂, 95% interval on ε̂ (inference on the segment average), loss cost per policy-year, base premium per policy-year, `model_version`, `data_version`, `semi_synthetic = true`. Base premium = loss cost × (1 + loading), loading in config, so Revenue₀ and Losses₀ are consistent by construction; Phase 6 scales by segment policy count. There is **no** proposed-adjustment field.

> **APPROVED — Batch 2 (October 21, 2026):** Option A, the 36-cell actuarial grid and merge floor, and the Pydantic segment contract. The proposed band edges remain subject to §2.4 close-out; schema contains no proposed-adjustment field.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved; proposed band edges subject to §2.4 close-out.

**Consequences.** Scoping §7 and PRD wording on inputs are corrected (R11). Phase 3 and 6 specs inherit these units.

---

### P2-D4 — Feature pipeline and tail handling

**Question.** Which pre-treatment covariates enter the models and how are tails handled, in one shared pipeline?

**Context.** Attempt 1's engineered composite feature reached kurtosis 222 with the top 0.1% of rows carrying 19% of its variance. The real data has seven rating factors, an exposure anomaly (421 policies above 1.0, maximum 1.99) and heavy-tailed severity. The raw numeric tails of DriverAge, CarAge and Density are not yet shown (§2.4).

**Governing.** Scoping §11, carryover, INV-3, INV-8, A4.

**Options.**

| | A. One pipeline: documented clipping, banding and log transforms of raw factors; no composite indices; claims variables excluded from elasticity covariates | B. A plus an engineered composite risk index | C. Separate pipelines per model |
|---|---|---|---|
| Leverage risk | Low | The documented failure | Low |
| Train/serve consistency | Single source | Single source | Skew risk |
| Complexity | Lowest | Higher | Highest |

**Trade-offs.** B reintroduces the demonstrated leverage problem for no stated gain; C duplicates logic that must stay identical.

**Recommendation.** **A.** One cleaning step emits two views: banded/categorical for the GLM and numeric/ordinal for the forest. Rules: exposure capped at 1.0 for the 421 affected policies with a logged count and a tolerance expectation in the raw contract; log(Density); DriverAge and CarAge bounds set after the raw-distribution check. Every clip or exclusion is a stated rule with logged counts. The fitted pipeline, band edges and feature list are versioned artifacts. A baseline feature-distribution summary is persisted with them (cheap, avoids a moving reference later; no monitoring logic in Phase 2).

> **APPROVED — Batch 2 (October 21, 2026):** Option A and the shared dual-view pipeline, no composite index, claims-feature exclusion, clipping/logging, and artifact/versioning requirements. DriverAge and CarAge bounds follow the raw-distribution check.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved; raw-distribution-dependent bounds remain to be set.

**Consequences.** Any later composite feature requires a new decision with a leverage test.

---

### P2-D5 — Estimator configuration and training scale

**Question.** Can `CausalForestDML` run on the full 413,169 policies within an acceptable budget, or must training use a subsample, and which nuisance models are used?

**Context.** The forest with cross-fitting is the most expensive step; D6 and D7 multiply its cost by seeds and sensitivity cells. Machine resources are unrecorded (V8). `cv_folds: 5` exists in config.

**Governing.** PRD §7, solo capacity, D6/D7 cost.

**Options.**

| | A. Full-data fit | B. Stratified subsample | C. Unresolved — qualification test Q2 |
|---|---|---|---|
| Fidelity | Highest | Depends on n\* | Evidence-based |
| Cost | Highest | Lower | One test |

**Trade-offs.** Choosing A or B now would be a guess about runtime and about recovery sensitivity to n.

**Recommendation.** **C**, then A or B by the result. **Q2:** at n ∈ {25k, 100k, 250k, 413k}, record wall-clock time, peak memory and recovery (Spearman across segments, interval coverage) on a pilot seed family; compare nuisance families (gradient-boosted trees from the existing scikit-learn dependency versus a regularized linear model). No new dependency unless Q2 shows it is needed.

**Pass criteria (proposed, adjustable).** Full registered fit ≤ 30 minutes and ≤ 80% of RAM; otherwise adopt the smallest n whose Spearman is within 0.02 of the largest tested n for validation runs, with the registered fit on full data if the budget allows, and the choice stated in the artifact manifest.

> **APPROVED — Batch 3 (October 21, 2026):** Run Q2 at the four stated sample sizes and compare both nuisance-model families. The proposed 30-minute/80%-RAM budget and 0.02 recovery tolerance are approved as decision criteria. Q2 still selects the scale and configuration.

**Approval.** Q2 method and budget criteria approved, October 21, 2026 (Batch 3); configuration is fixed by Q2. **Decision status.** Approved method; outcome unresolved — Q2.

**Consequences.** The Q2 report becomes the basis for K (seeds) in D6 and the grid in D7.

---

### P2-D6 — Validation protocol and gate design

**Question.** What evidence is required for the estimator to "recover the known effect" (E1)?

**Context.** R3. A single aggregate Pearson on one seed was shown unstable (carryover). With about 36 segments, Spearman across segments has limited resolution.

**Governing.** Scoping §0/§2, roadmap exit, falsifiable-testing discipline (`system_design.md` §6).

**Options.** One approach is viable under the carryover (a single-seed aggregate correlation is excluded by it). Sub-choices are fixed below.

**Trade-offs.** Cost is K × fit; K is set by Q2. Tautological checks (for example the frequency × severity identity, or asserting a value equals itself) are prohibited.

**Recommendation.**

- Evaluation on held-out policies (the `test_size` split), never in-sample.
- K seeds (minimum 5; final K by Q2 cost). Each seed varies DGP noise and estimator seed. Pilot seeds (used to calibrate thresholds under D6a) are disjoint from final seeds (used for the gate).
- Metrics, each aggregated across seeds: (m1) Spearman between τ̂ and τ across segments; (m1b) policy-level Spearman on held-out policies, reported for resolution; (m2) per-segment bias on ε; (m3) 95% interval coverage of the truth; (m4) interval self-consistency (finite, contains its own point estimate); (m5) sign agreement.
- Segment intervals are inference on the segment-average effect, not the average of policy-level interval endpoints; the method is fixed in the P2-F06 spec.
- Negative controls that must fail the gate: (c1) an unadjusted estimator on the same confounded data; (c2) shuffled τ̂; (c3) a zero-effect DGP, where the fraction of segments declared significant stays near the nominal rate. Positive control: the randomized-T variant must pass.
- Gate statistic: worst seed and median across seeds, never one seed.

> **APPROVED — Batch 2 (October 21, 2026):** The held-out, multi-seed protocol; listed metrics; negative and positive controls; segment-average interval method; and worst-seed plus median gate statistic are approved.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved.

**Consequences.** The harness is built first against the controls, so a gate that cannot fail is caught before the estimator is judged.

---

### P2-D6a — Gate thresholds *(nested under D6)*

**Question.** What numeric thresholds define a pass?

**Context.** None exist (scoping §8). Fixing them after seeing the estimator's final result is indefensible; fixing them blind risks an unachievable or vacuous gate.

**Options.** (A) Fix numbers now from judgment. (B) Unresolved — Q3.

**Trade-offs.** A has no evidence base; B costs one test and keeps the gate honest.

**Recommendation.** **B — Q3.** On pilot seeds, run the estimator and all negative controls; place each threshold in the gap between the upper tail of the control distributions and the lower tail of the estimator distribution, with a stated minimum margin. If no gap exists, the benchmark or estimator design has failed, and that finding returns to D2b/D5; thresholds are not lowered to fit. Values are frozen in config before the final-seed run; later changes need approval.

> **APPROVED — Batch 3 (October 21, 2026):** Use Q3 to calibrate thresholds on pilot seeds, require separation with a stated margin, and freeze values before final-seed evaluation. Numeric thresholds remain unresolved until Q3; a failed separation reopens benchmark/estimator design rather than lowering the gate.

**Approval.** Calibration method approved, October 21, 2026 (Batch 3); values fixed by Q3. **Decision status.** Approved method; thresholds unresolved — Q3.

**Consequences.** The frozen values and the Q3 report are part of the registered artifact's validation summary.

---

### P2-D7 — Confounding sensitivity method

**Question.** How is the roadmap's "confounding sensitivity analysis" performed so that it survives adversarial review (PRD §14 Q1)?

**Context.** Because the DGP is known, unobserved-confounding strength can be injected and its effect measured exactly. Attempt 1's refuters belonged to the silent-fallback problem behind the "zero silent fallbacks" rule.

**Governing.** PRD §14, scoping §11, `system_design.md` §6.

**Options.**

| | A. DoWhy-style refuters | B. Hidden-confounder sweep on the known DGP | C. Both |
|---|---|---|---|
| Uses ground truth | No | Yes | Partly |
| Failure seen before | Silent fallback | None | Inherits A's |
| Output | Pass/fail checks | Degradation curve and robustness value | Both |

**Trade-offs.** A is redundant when truth is known and carries a demonstrated failure mode; C doubles cost.

**Recommendation.** **B.** Add an unobserved confounder U to T and Y at a small grid of strengths (proposed 3 × 3 plus a zero cell), hide it from the estimator, and report degradation of m1–m3. The robustness value is the smallest strength at which median Spearman breaches the D6a threshold, labeled a property of the synthetic benchmark. Grid size follows Q2 cost; the harness raises on any skipped cell. A reopens only if a reviewer requires it, with a fail-loud wrapper.

> **APPROVED — Batch 2 (October 21, 2026):** Option B and the hidden-confounder degradation sweep, robustness-value definition, Q2-bounded grid, fail-loud behavior, and stated condition for reconsidering Option A.

**Approval.** Approved, October 21, 2026 (Batch 2). **Decision status.** Approved.

**Consequences.** Scenario S4 in D12 depends on this output.

---

### P2-D8 — Elasticity-side data contracts and DAG ordering

**Question.** Which datasets need a GX suite, and where do the gates sit in the DAG?

**Context.** INV-3 requires every versioned dataset to pass its suite. Phase 2 creates two: the raw policy table (frequency joined with per-policy severity) and the generated training table. ADR-003 names one elasticity suite. The existing suite's schema is unverified and may reflect a different column set (R11, V3).

**Governing.** INV-3, ADR-003, ADR-006, ADR-007.

**Options.**

| | A. Two suites: raw policy table; training table | B. Training-table suite only; raw checked by ad hoc code | C. One suite spanning both tables |
|---|---|---|---|
| INV-3 literal compliance | Yes | **No, if raw is versioned** | Yes |
| Failure attribution | Clear | Clear for derived only | Blurred |

**Trade-offs.** B saves one suite at the cost of violating INV-3; C blurs which dataset failed.

**Recommendation.** **A**, as an amendment to ADR-003 (the regulatory-corpus suite is unchanged). Raw suite, grounded in the EDA: unique PolicyID; ClaimNb in 0–4; exposure above 0 and at most 2 with a tolerance band for the 0.102% above 1.0; ClaimAmount above 0; every severity record maps to a policy; per-policy severity count equals ClaimNb; no nulls. Training-table suite: pre-treatment covariates only, no truth column, T within the corridor, overlap floor, Y binary. Existing fixtures are re-based to the real schema, with new fixtures for each failure above. The per-policy loss field carried forward is the **sum** of ClaimAmount across a policy's claims (the EDA's 757 policies with 2-4 claims must aggregate, not select a single claim); this summed field is what feeds segment loss cost.

Provisional DAG (names finalized in P2-F10): `ingest → validate_raw → build_features → fit_baseline → generate_dgp → validate_training → train_estimator → validate_recovery → register_candidate → version`. No stage versions a dataset before its suite passes.

> **APPROVED — Batch 1 (October 21, 2026):** Option A; two elasticity-side suites; the specified raw/training checks, fixtures, summed per-policy claim loss, and fail-before-version DAG ordering. ADR-003 is amended accordingly.

**Approval.** Approved, October 21, 2026 (Batch 1; ADR-003 amendment). **Decision status.** Approved.

**Consequences.** ADR-003 text changes from one to two elasticity-side suites.

---

### P2-D9 — Raw data identity, source and pinning

**Question.** Which dataset is this, where does it come from, and how is it pinned?

**Context.** The EDA ran on `data/raw/freMTPLfreq.csv.zip` and `data/raw/freMTPLsev.csv` (413,169 policies; 16,181 claims). The PRD, scoping document, README and system design say freMTPL2 and list bonus-malus (R11).

**Governing.** INV-3, INV-10, ADR-007, PRD FR-1/§9.

**Options.** One. The data is what it is: pin the analyzed files by checksum in a DVC-tracked intake manifest (failing on mismatch), and correct the documents to name the dataset as the original French MTPL benchmark with seven rating factors and no BonusMalus or Area. An unpinned or misnamed source would make downstream artifacts irreproducible and the documents false.

> **ACCEPTED — Batch 1 (October 21, 2026):** Correct the dataset identity and factor list, and require checksum-pinned input through a DVC-tracked intake manifest.

**Trade-offs.** None; the correction is factual.

**Approval.** No approval required — factual correction accepted, October 21, 2026 (Batch 1). **Decision status.** Accepted.

**Consequences.** Amendments to PRD FR-1/§9, scoping §7, README and system design naming are applied under change management.

---

### P2-D10 — Reproducibility contract and execution tiers

**Question.** What reproducibility can Phase 2 honestly guarantee, and what runs where?

**Context.** R5. Forest fitting is multi-threaded and floating-point results differ across operating systems. CI runs on hosted runners that cannot reach the local DVC remote (ADR-007), and full-data training is too heavy for CI.

**Governing.** PRD §7, ADR-007, ADR-008.

**Options.**

| | A. Bit-for-bit everywhere (PRD literal) | B. Tiered contract |
|---|---|---|
| Achievable | No (cross-OS, threading) | Yes |
| Honest | Only by silent weakening | Explicit |

**Trade-offs.** A cannot be met for model outputs; B states exactly what is guaranteed.

**Recommendation.** **B.** Data products (cleaned data, DGP table, truth sidecar, manifest) are bit-identical under the same lockfile and operating system; model outputs reproduce within stated numeric tolerances; cross-OS identity is not claimed. Tiers:

- **T1 CI-smoke:** committed small fixture (a few thousand rows); pipeline integrity, schemas, falsifiable unit tests, tolerance-level model checks, and a coverage floor on Tier 1 modules (proposed 65% for this phase, raised later).
- **T2 Full run:** local `dvc repro` on full data; registers the candidate artifact; logs evidence to MLflow.
- **T3 Validation campaign:** K seeds, controls and sensitivity grid, on demand and before sign-off; its summary report is committed.

PRD §7 is reworded accordingly.

> **APPROVED — Batch 1 (October 21, 2026):** Option B; tiered reproducibility contract and T1/T2/T3 execution tiers, including the proposed 65% T1 coverage floor.

**Approval.** Approved, October 21, 2026 (Batch 1). **Decision status.** Approved.

**Consequences.** The CI fixture must preserve the data's key shape (zero-inflation and a heavy tail) well enough to exercise the code paths, not to reproduce results.

---

### P2-D11 — What Phase 2 registers and promotes

**Question.** What is the promotable unit: the fitted estimator, the segment-level output, or both?

**Context.** Phases 3, 6 and the showcase consume segments, not policy-level predictions. The registered model name `aegis-causal-elasticity` appears in the README. Showcase input is curated and INV-10 excludes free-form input. Serialized estimators carry pickle-loading and library-version fragility.

**Governing.** INV-5, INV-10, D3, gates Part 4.2.

**Options.**

| | A. Fitted estimator | B. Segment-level output set | C. B promoted; estimator stored as non-promotable run artifact |
|---|---|---|---|
| Serves Phases 3/6/showcase | Needs inference code | Directly | Directly |
| Reproducibility evidence | Yes | Indirect | Yes |
| Load-time risk | Pickle, version-bound | Plain data | Plain data in serving |

**Trade-offs.** A forces serving to deserialize a library-bound object for no consumer need; B loses the evidence trail; C keeps both at the cost of one extra logged artifact.

**Recommendation.** **C.** The registered model packages the segment table (D3 contract), the baseline and causal calibration reports, the validation-gate summary, the DGP manifest id, the DVC data hash and the git commit. The serving path never deserializes the estimator.

> **APPROVED — Batch 4 (October 21, 2026):** Option C. Promote the segment-level output package with its calibration, validation, DGP, data, and source lineage; retain the fitted estimator as a non-promotable run artifact and never deserialize it in serving.

**Approval.** Approved, October 21, 2026 (Batch 4). **Decision status.** Approved.

**Consequences.** No feature computation happens at serving time, so training-serving skew cannot arise in Phase 2.

---

### P2-D11a — Tracking backend and promotion mechanism *(nested under D11)*

**Question.** Which MLflow backend and promotion mechanism does Phase 2 use?

**Context.** Gates Part 4.2 and existing guidance describe promotion as a stage transition with rollback as a stage-tag flip. As far as I know, current MLflow versions deprecate registry stages in favor of aliases and are moving away from the plain file store. This must be verified against the pinned version (V9) before approval.

**Verification update (October 21, 2026):** V9 found MLflow 3.15.2 pinned in `uv.lock`; current MLflow registry documentation describes alias creation and reassignment, and confirms aliases replace stages as the recommended promotion reference.

**Governing.** Gates Part 4.2, D11, ADR-005.

**Options.**

| | A. Local SQLite backend (path under gitignored `artifacts/`) with alias promotion | B. File store with tags |
|---|---|---|
| Registry support | Supported | Version-dependent |
| Human-review gate | Review record required to set `approved` | Tag convention only |
| Fit with local-first posture | Yes | Yes |

**Trade-offs.** B depends on behavior that may be deprecated; A is equally local.

**Recommendation.** **A.** The pipeline sets `candidate`; `approved` is set only with a recorded reviewer identity and the hash of the reviewed calibration report. A candidate cannot be promoted without lineage tags: git commit, DVC data hash, run id and a gate-pass tag. Rollback is re-pointing `approved` to the previous version. Phase 3 and the showcase read only `approved`. Gates Part 4.2 wording is updated. V9 verified MLflow 3.15.2 alias support; revisit only if implementation evidence contradicts this verified behavior.

> **APPROVED — Batch 1 (October 21, 2026):** Option A, verified against V9. Use local SQLite-backed MLflow tracking and mutable `candidate`/`approved` aliases; gate promotion on human review and complete lineage. Rollback re-points `approved` to the previous version.

**Approval.** Approved, October 21, 2026 (Batch 1; V9 verified). **Decision status.** Approved.

**Consequences.** Tracking URI comes from environment configuration, not code.

---

### P2-D12 — Showcase slice scope and INV-5 enforcement

**Question.** Which scenarios does the Phase 2 slice show, and how is the INV-5 exemption enforced and not merely labeled?

**Context.** R4. ADR-009 requires curated presets, FastAPI + Jinja2, and at least one scenario that exposes a limit. In Phase 2 the analog of "escalation" is the estimator's own failure modes.

**Governing.** ADR-009, ADR-011, INV-5, PRD §12.

**Options.** One approach satisfies ADR-009; the scenario set is the choice requiring approval.

**Trade-offs.** A happy-path-only set would hide exactly the limits ADR-009 exists to show.

**Recommendation.** Four presets reading the `approved` artifact: (S1) a large, well-recovered segment; (S2) a small segment with a wide interval; (S3) a segment whose uncapped loss cost is tail-inflated, chosen from the Q1 output, showing the effect of capping; (S4) a hidden-confounding degradation case from D7. The slice shows ε̂ against the known truth with intervals and the validation-gate summary, excludes the bandit and any proposed Δ, and serves Chart.js as a local static asset.

**Enforcement (no approval required — only one viable option exists under the applicable constraints).** The response schema has no proposed-adjustment field, a test asserts its absence, and every page carries the banner *model validation demo on semi-synthetic data — not a pricing recommendation*. A label alone would not stop a future field from appearing.

> **APPROVED — Batch 4 (October 21, 2026):** Four curated presets from the `approved` artifact; ε̂/truth, intervals, and gate summary only; no bandit or proposed Δ. Optionally dependent scenarios S3/S4 remain blocked until their Q1/D7 evidence exists. The schema-level prohibition and persistent demo banner are accepted as required enforcement.

**Approval.** Scenario set approved, October 21, 2026 (Batch 4); enforcement accepted as the sole viable approach. **Decision status.** Approved.

**Consequences.** S3 and S4 depend on Q1 and P2-F08 outputs; their absence blocks E2.

---

## 7. What Happens After Approval

1. **Approved decisions become implementation constraints.** Unresolved decisions (D1a level, D5, D6a) stay open until their qualification tests report; each result is recorded back in §6 as a dated entry.
2. **ADR consolidation complete:** approved decisions are recorded in ADR-012–ADR-015 in `system_design.md`: (i) baseline target, loss treatment, and feature preparation (D1, D1a, D4); (ii) semi-synthetic causal benchmark and validation protocol (D2, D2a, D2b, D5, D6, D6a, D7); (iii) Tier 1 segments, output contract, and registry (D3, D11, D11a); (iv) dataset identity, elasticity-side contracts, and reproducibility tiers (D8, D9, D10). D12 is recorded as an amendment to ADR-009.
3. **Documents amended in this approval change set:** PRD FR-1/§4/§7/§9/§11/§12 (D1, D9, D10); scoping §0/§6/§7/§9 (D1, D9); README Phase 2 row (D1, D9); roadmap Phase 2 capability and showcase text (D1, D12); ADR-003 (D8); ADR-009 clarification that the Phase 2 slice excludes the bandit (D12); gates Part 4.2 (D11a). Dataset identity is corrected to the original French MTPL benchmark, with seven rating factors and no BonusMalus or Area.
4. **The Phase Execution Plan is the next planning artifact.** It, not this document, sequences stages and gates. Inputs handed to it include independent ADR/decision review and sign-off checkpoints; the ADRs are consolidated and approved but implementation is not yet verified:
   - Stage 0 pre-implementation verification: V1–V9 and the open EDA gaps (§2.4).
   - Qualification tests Q1–Q3 as stage gates, with the pass criteria above.
   - Candidates for the one-time falsification pass, each a failure that must be detected: a truth column placed in the training table (D2, D8); a deliberately broken estimator against the gate (D6, via controls); a malformed raw file halting the DAG (D8); a proposed-adjustment field added to the showcase schema (D12); a threshold changed after the final run (D6a).
5. **Feature-level work enters the SDD lifecycle.** P2-F01 to P2-F11 each need their spec approved before implementation, at the depth stated in §4. Specs may not reopen an approved decision.
6. **Material deviations return through the decision and change-management workflow.** Approval of a decision here does not authorize unrelated work.

**Downstream obligations.** Phase 3 reads only the `approved` artifact and treats ε̂ intervals as exploration inputs, not proposals. Phase 6's Impact specification inherits the ε definition, ratio-of-sums aggregation, per-policy-year units, base-premium rule and |ε|·Δ bound from D2a/D3; its absolute underwriting result (Rev − Losses) is where ε matters, per ADR-011. Phase 8 inherits the validation report and the baseline feature-distribution summary.

---

## 8. Post-Implementation Review & Remediation

This section stays open until implementation has been independently reviewed against the approved decisions, requirements, invariants, contracts, fixtures, tests, evaluations and runtime evidence. It records what was built, not a rewrite of intent. Expectations are pre-seeded; the remaining columns are completed during review.

| Item | Expectation | Actual | Finding | Severity | Remediation | Status |
|---|---|---|---|---|---|---|
| Dataset identity | Files pinned by checksum; documents name the original MTPL benchmark and its seven factors (D9) | | | | | |
| Loss treatment | Capped Tweedie baseline with excess load; level supported by Q1; portfolio A/E within tolerance; excess share by segment reported (D1, D1a) | | | | | |
| Truth isolation | No truth column reachable from estimator inputs; fixture with a truth column fails; import-boundary test passes (D2) | | | | | |
| DGP self-tests | Estimand gap, overlap, dispersion, within-segment homogeneity, max \|ε\|·0.15 < 1 verified; exposure not used in the DGP (D2a, D2b) | | | | | |
| Segments and contract | Size floors met; GLM-aggregated loss cost; ratio-of-sums; schema has no Δ field (D3) | | | | | |
| Feature pipeline | Deterministic; exclusions logged; no composite index; claims variables absent from elasticity covariates; artifacts versioned (D4) | | | | | |
| Estimator scale | Q2 verdict recorded; budget met or subsample choice documented (D5) | | | | | |
| Validation gate | Thresholds frozen before final seeds; controls fail; estimator passes on worst seed and median (D6, D6a) | | | | | |
| Sensitivity | Degradation curve and robustness value; no skipped cells (D7) | | | | | |
| Data contracts | INV-3 ordering visible in the DAG; failing fixture per suite; ADR-003 amended (D8) | | | | | |
| Reproducibility | Data products bit-identical; model outputs within tolerance; CI-smoke green with coverage floor (D10) | | | | | |
| Registry | `approved` set only with review record and lineage tags; serving never loads the pickled estimator (D11, D11a) | | | | | |
| Showcase | Banner on every page; no Δ field; four presets render from `approved` (D12) | | | | | |
| Engineering standards | INV-8 held; tests falsifiable; no silent fallbacks; `ruff` and `pyright` clean | | | | | |
| Documentation | All amendments in §7.3 applied; no stale severity, dataset-name or bit-for-bit wording remains | | | | | |
| Exit criteria | E1–E6 independently verified | | | | | |

### Phase closure

The phase cannot close until: every approval-required decision has a final status; implementation stages have passed their gates; required tests and evaluations provide evidence; material deviations are resolved or explicitly accepted; the four ADRs are consolidated; this review is complete; exit criteria are independently verified; and Phase 3 entry conditions hold. Final status is stated per item and the four states are never interchangeable:

| State | Meaning |
|---|---|
| Decided | Approved in this document |
| Implemented | Built and merged |
| Verified | Evidence shows it meets the decision and exit criteria |
| Signed Off | Independent review complete and accepted |
