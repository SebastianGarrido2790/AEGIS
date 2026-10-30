# Phase 2 — Staged Execution Plan

## Tier 1: Deterministic ML Baseline (EDA-First Redo)

**Author:** Sebastián Garrido Arévalo | **Date:** October 21, 2026
**Status:** ✅ **APPROVED (October 21, 2026)**
**Implementation authorization:** ✅ Granted 2026-10-21 (operator authorized: "begin Stage 0")
**Planning baseline:** Phase 1 Validated; Phase 2 Implementation Plan v0.3 approved across all four decision batches (2026-10-21); ADR-012 … ADR-015 consolidated in `architecture/system_design.md`.

---

## Purpose and Execution Discipline

This plan translates the approved Phase 2 Implementation Plan (`reports/docs/phases/phase_2/implementation_plan.md`) into a sequential, evidence-gated implementation sequence. It introduces **no new architectural decisions**. The decisions already accepted in the architecture record (ADR-012 through ADR-015, with amended ADR-003 and ADR-009) are binding constraints to implement, not decisions to reopen.

**Core execution rules:**

1. **Sequential execution:** Stages execute strictly in order. Each stage terminates in a **hard gate**. No work in the next stage begins while the previous stage is failed, partial, awaiting required approval, or awaiting unresolved remediation.
2. **Separation of phase and feature execution:** The Phase Execution Plan controls the phase-level sequence and dependencies. The Feature SDDs control the feature-level specifications, plans, tasks, and code. Neither may silently replace the other.
3. **Selective falsification:** Falsification is a deliberate attempt to prove that a specific, load-bearing control actually detects a meaningful failure mode. It is not applied mechanically to every stage, but only where injecting a controlled failure provides positive evidence of detection capability.
4. **Change management:** If repository evidence or runtime qualification exposes a material decision not covered by approved architecture, execution stops immediately and returns to the decision workflow. No silent workarounds.

---

## 1. Execution Preconditions

The following conditions must be satisfied before Stage 0 implementation begins:

- [x] Phase 2 Implementation Plan v0.3 explicitly approved across all four batches (2026-10-21; verified in `implementation_plan.md` §5).
- [x] Consolidated ADRs (ADR-012, ADR-013, ADR-014, ADR-015) verified in `reports/docs/architecture/system_design.md` §4.
- [x] Amendments to ADR-003 (two elasticity-side GX suites) and ADR-009 (Phase 2 slice scope excludes bandit/Δ) verified in `system_design.md`.
- [x] V9 MLflow version and registry alias behavior verified live against SQLite (MLflow 3.15.2 supports mutable `candidate`/`approved` aliases; tested 2026-10-21).
- [x] Project documents (PRD FR-1/§4/§7/§9/§11/§12, Scoping Doc §0/§6/§7/§9, Roadmap, README, Capability Profile Gates) reconciled with approved dataset identity (original French MTPL, 7 factors, no BonusMalus/Area) and Tweedie baseline scope.
- [x] Working tree clean and ready for Stage 0 baseline capture.
- [x] Stage 0 baseline audit and V1–V8 verification executed and recorded.
- [x] Stage 0 blocking clarification (Q2 sample sizes vs. 20% test-split) resolved with operator before Stage 7.
- [x] Formal implementation authorization granted by operator ("begin Stage 0").

---

## 2. Approved Feature Set & Stage Mapping

Derived from the approved Feature Register (`implementation_plan.md` §4). All 11 candidate features are approved into the execution scope.

| Feature ID | Folder / Slug                     | Feature Name                                                  | SDD Depth | Primary Stage(s)                 | Dependencies  | Status         |
| ---------- | --------------------------------- | ------------------------------------------------------------- | --------- | -------------------------------- | ------------- | -------------- |
| **P2-F01** | `001_raw-intake-contract`         | Raw intake, checksum manifest & raw-table contract            | Light     | Stage 1                          | —             | ✅ Implemented |
| **P2-F02** | `002_feature-pipeline`            | Deterministic dual-view feature pipeline & tail handling      | Full      | Stage 2                          | F01           | ✅ Implemented |
| **P2-F03** | `003_glm-baseline-loss-treatment` | Cap-parameterized Tweedie GLM & calibration report            | Full      | Stage 3 (finalized Stage 5)      | F02           | ✅ Implemented |
| **P2-F04** | `004_semi-synthetic-dgp`          | Semi-synthetic DGP, truth sidecar & training contract         | Full      | Stage 6                          | F02, F03      | ✅ Implemented |
| **P2-F05** | `005_segments-tier1-contract`     | 36-cell segment definition & Tier 1 Pydantic schema           | Full      | Stage 4                          | F02, F03      | ✅ Implemented |
| **P2-F06** | `006_causal-estimator`            | `CausalForestDML` training & segment-level inference          | Full      | Stages 7–8                       | F04, F05      | ✅ Implemented |
| **P2-F07** | `007_validation-gate`             | Multi-seed validation harness & qualification gate            | Full      | Stages 9–10                      | F06           | ✅ Implemented |
| **P2-F08** | `008_confounder-sensitivity`      | Hidden-confounder sensitivity sweep & robustness value        | Light     | Stage 11                         | F04, F07      | ⬜ Not started |
| **P2-F09** | `009_registry-promotion`          | MLflow artifact packaging & human-gated alias promotion       | Full      | Stage 12                         | F03, F06, F07 | ⬜ Not started |
| **P2-F10** | `010_pipeline-config-ci-tiers`    | DVC DAG extension, config expansion & CI tiers                | Light     | Stage 14 (evolves across stages) | F01–F09       | ⬜ Not started |
| **P2-F11** | `011_showcase-slice`              | Diagnostic glass-box showcase slice (presets from `approved`) | Full      | Stage 13                         | F09           | ⬜ Not started |

**Sequencing dependency note:** Q1 (Cap Sweep) is listed in the implementation plan's §5 qualification table as needing the F02/F03 skeleton, but its approved pass criterion is _segment loss-cost ranking stability_ and it must report _excess-share by segment_ (P2-D1a, ADR-012). Therefore, Q1 strictly requires the Stage 4 segment grid (F05). The stage order (Stage 3 GLM skeleton → Stage 4 Segments → Stage 5 Q1 Cap Sweep) honors this dependency naturally without modifying any approved decision.

**Stage 0 blocking clarification:** D2a fixes the DGP unit as all 413,169 policies; D6 fixes evaluation on held-out policies (`test_size: 0.20`); Q2 specifies testing sample sizes n ∈ {25k, 100k, 250k, 413k}. If 20% is held out, the estimator's training split is ~330k, so 413k cannot be pure training rows. Stage 0 must confirm whether Q2 sample sizes represent total dataset slices (with the 80/20 split applied inside each) or pure training samples. This must be resolved before Stage 7 begins.

---

## 3. Stage Index

| Stage  | Name                                     | Primary Deliverable                                       | ADR Implements   | Falsification                                                           | Hard Gate (One-Line Summary)                                                                             |
| ------ | ---------------------------------------- | --------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **0**  | Pre-Implementation Verification          | Baseline audit, V1–V9 checks, EDA resolution              | None new         | Evidence reconciliation against live repository                         | ✅ **PASSED** — All V-checks recorded; EDA gaps closed; Q2 ambiguity resolved; baseline commit `04903e6` |
| **1**  | Raw Intake & Raw Data Contract           | Checksum manifest, raw GX suite, fixtures                 | ADR-015, ADR-003 | Orphan severity record halts the DVC DAG before versioning              | ✅ **PASSED** — Valid data passes; malformed fixtures fail named rules; tamper fails                     |
| **2**  | Deterministic Feature Pipeline           | Dual-view pipeline, clip logging, versioned artifacts     | ADR-012          | None (re-run determinism verified by assertion)                         | ✅ **PASSED** — Identical output on identical input; clips counted; no composite/claims leakage          |
| **3**  | GLM Baseline & Calibration Reporting     | Cap-parameterized Tweedie GLM & deviance report           | ADR-012          | None                                                                    | ✅ **PASSED** — Converged `p=1.84`; deviance + A/E deciles produced; power justified                     |
| **4**  | Segment Definition & Tier 1 Contract     | 36-cell grid, merge rule, Pydantic schema                 | ADR-014          | Injected `proposed_adjustment` field is rejected by schema              | ✅ **PASSED** — 35 segments, floor met, schema rejects Δ; ratio-of-sums tested                           |
| **5**  | Qualification Q1 — Cap Sweep & Selection | Frozen loss cap, excess load & segment report             | ADR-012          | None (evidence-gathering qualification)                                 | ✅ **PASSED** — Cap €100,000; Spearman ≥0.98 / deviance ≤1%; excess concentration escalated              |
| **6**  | Semi-Synthetic DGP & Training Contract   | Training table, truth sidecar, manifest, GX suite         | ADR-013, ADR-015 | Planted truth column fails GX suite; import-boundary test blocks loader | All DGP self-tests pass; suite fails truth column; import boundary holds                                 |
| **7**  | Qualification Q2 — Estimator Scale       | Scale, nuisance family & runtime report                   | ADR-013          | None (evidence-gathering qualification)                                 | All grid cells run; budget criteria met; scale and nuisance family selected                              |
| **8**  | Causal Estimator & Segment Inference     | Trained estimator & segment-average intervals             | ADR-013          | None                                                                    | ✅ **PASSED** — 35 segments; 100% finite/self-consistent; repeated CSV byte-identical                    |
| **9**  | Validation Harness & Q3 Calibration      | Metric harness, negative/positive controls, thresholds    | ADR-013          | Negative controls (unadjusted, shuffled, zero-effect) fail harness      | ✅ **PASSED** — controls fail; positive passes (`m1=0.9152`); thresholds frozen with margin              |
| **10** | Final-Seed Validation Gate               | Multi-seed recovery evaluation (E1 close)                 | ADR-013          | Attempted modification of frozen threshold is detected & blocked        | ✅ **PASSED** — Closed under Option F Amendment (October 30, 2026); E1 met under governed recovery envelope |
| **11** | Hidden-Confounder Sensitivity            | Degradation curve & robustness value                      | ADR-013          | None (skipped-cell guard asserted in gate)                              | ✅ **PASSED** — 10/10 cells; zero skips; Rv=0.5004; Scenario S4 locked (gt_0.05_gy_0.50) |
| **12** | Registry & Human-Gated Promotion         | Packaged artifact, SQLite tracking, alias promotion       | ADR-014          | Promotion to `approved` without human review/lineage is rejected        | Pipeline sets `candidate`; `approved` requires review + lineage; serving is plain data                   |
| **13** | Showcase Slice                           | Four curated presets rendering from `approved` (E2 close) | ADR-009, ADR-014 | Response schema rejects Δ field; banner presence asserted               | 4 presets render from `approved`; no Δ field; model validation banner present; E2 met                    |
| **14** | DAG, Config & CI Tiers / Reproducibility | 10-stage DVC DAG, `params.yaml`, CI-smoke tier            | ADR-015, ADR-003 | None (reproducibility verified by re-run)                               | CI-smoke green with ≥65% Tier 1 coverage; `dvc repro` reproduces bit-identical                           |
| **15** | ADR Consolidation & Phase Sign-Off       | Complete PIR, validated ADRs, Phase 3 entry               | ADR-012…015      | Documentation-to-reality audit of all exit claims                       | E1–E6 independently verified; PIR complete; human sign-off recorded                                      |

---

## 4. Feature-Level SDD Integration

When a phase stage contains substantial feature work, the applicable feature must follow:

$$\text{specification} \longrightarrow \text{approval} \longrightarrow \text{plan} \longrightarrow \text{tasks} \longrightarrow \text{implementation} \longrightarrow \text{validation}$$

The Phase Execution Plan controls the **phase-level sequence**. The feature SDD controls the **feature-level execution**. **Neither may silently replace the other.**

### Architectural Hierarchy

```text
Project Constitution / Governance
│
reports/docs/
├── implementation_roadmap.md                  # High-level phase dependency map (Phases 0–9)
├── architecture/
│   └── system_design.md                       # Canonical ADR ledger (ADR-001 … ADR-015)
├── evaluations/
│   └── capability_profile_gates.md            # Agent justification & complexity gates
└── phases/
    ├── phase_0/
    │   └── implementation_plan.md             # Planning foundation & scoping
    ├── phase_1/
    │   ├── implementation_plan.md             # Scaffold & data contracts decisions
    │   └── execution_plan.md                  # Phase 1 execution record (Validated)
    └── phase_2/
        ├── implementation_plan.md             # Phase 2 decisions (D1–D12; ADR-012…ADR-015)
        ├── execution_plan.md                  # This document: sequential execution workflow
        ├── 001_raw-intake-contract/           # P2-F01 (Light: tasks.md)
        ├── 002_feature-pipeline/              # P2-F02 (Full: spec.md, plan.md, tasks.md)
        ├── 003_glm-baseline-loss-treatment/   # P2-F03 (Full: spec.md, plan.md, tasks.md; Q1)
        ├── 004_semi-synthetic-dgp/            # P2-F04 (Full: spec.md, plan.md, tasks.md)
        ├── 005_segments-tier1-contract/       # P2-F05 (Full: spec.md, plan.md, tasks.md)
        ├── 006_causal-estimator/              # P2-F06 (Full: spec.md, plan.md, tasks.md; Q2)
        ├── 007_validation-gate/               # P2-F07 (Full: spec.md, plan.md, tasks.md; Q3)
        ├── 008_confounder-sensitivity/        # P2-F08 (Light: tasks.md)
        ├── 009_registry-promotion/            # P2-F09 (Full: spec.md, plan.md, tasks.md)
        ├── 010_pipeline-config-ci-tiers/      # P2-F10 (Light: tasks.md)
        └── 011_showcase-slice/                # P2-F11 (Full: spec.md, plan.md, tasks.md)
```

### SDD Lifecycle Protocol

For each feature folder in `reports/docs/phases/phase_2/`:

1. **`spec.md` (Specification):**
   - Must be written and **approved by the operator** before any technical plan or task is executed.
   - States exact input/output contracts (typed Pydantic schemas), mathematical formulations, boundary conditions, explicit error hierarchy, and falsifiable acceptance criteria.
   - Traces directly to governing ADRs and invariants.
2. **`plan.md` (Technical Plan):**
   - Decomposes the approved specification into internal module designs under `src/aegis/`.
   - Defines class/function signatures, dependency relationships, test strategy (unit, property, integration), and qualification test protocols (for Q1, Q2, Q3).
   - Enforces the 1,000-line module ceiling (INV-8) by construction.
3. **`tasks.md` (Task Decomposition):**
   - Decomposes the plan into fine-grained, atomic, sequential tasks.
   - Each task specifies: files touched, exact implementation actions, and automated verification commands (`pytest`, `ruff`, `pyright`).
   - Tasks execute one at a time. The working tree must be clean and tests green before advancing to the next task.
4. **Implementation & Validation:**
   - Follows Test-Driven Development (TDD) where applicable: test written first, observed failing, then implementation brings it green.
   - Gate verification: when all tasks in `tasks.md` are complete, the stage gate defined in this Execution Plan is evaluated against recorded empirical evidence.

### SDD Depth Standards

- **Full SDD (`spec.md` + `plan.md` + `tasks.md`):** Applied to core modeling, causal inference, data contracts, and public API interfaces (P2-F02, P2-F03, P2-F04, P2-F05, P2-F06, P2-F07, P2-F09, P2-F11). Spec approval is a mandatory human gate.
- **Light SDD (`tasks.md` with inline specification):** Applied to bounded tooling, raw intake scripts, or sensitivity sweep harnesses (P2-F01, P2-F08, P2-F10). The task list contains explicit contracts and acceptance assertions, but does not require a standalone multi-document suite.

---

## 5. Implementation Stages

### Stage 0 — Pre-Implementation Verification & Dependency Check ✅ **COMPLETED**

**Goal:** Establish that the baseline repository state is verified, all preconditions are satisfied, open EDA gaps are closed or formally accepted, and the Q2 sample-size ambiguity is resolved before any code is written.

**Evidence:** Executed outcomes for Stage 0 (V1–V9 outputs, EDA gap closure, Q2 sample-size resolution) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-0--pre-implementation-verification--dependency-check).

**Inputs / Dependencies:**

- `data/raw/freMTPLfreq.csv.zip` and `data/raw/freMTPLsev.csv` present and checksum-verified.
- `uv.lock` and Python 3.12 environment.

**ADR Implements:** None new (verification and baseline audit only).

**Falsification:**
Evidence reconciliation audit: all live command outputs, checksums, package imports, and hardware metrics were independently captured and reconciled against baseline claims. Zero unverified assertions.

**Gate 0 — Hard Close: ✅ PASSED (2026-10-21)**
All V1–V9 checks executed and documented; all 4 EDA gaps resolved with empirical metrics; Q2 sample-size definition locked with operator; baseline test/lint state 100% clean; zero production code modified. Stage 1 is unlocked.

---

### Stage 1 — Raw Intake & Raw Data Contract (P2-F01) ✅ **COMPLETED**

**Goal:** Enforce INV-3 at the very first pipeline node: raw observational data cannot be versioned or processed without passing a Great Expectations contract grounded in the empirical EDA findings.

**Evidence:** Executed outcomes for Stage 1 (checksum verification, policy-level joining, 20 GX expectations pass, 8 malformed fixtures fail, orphan severity falsification) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-1--raw-intake--raw-data-contract).

**Inputs / Dependencies:**

- Stage 0 baseline accepted.
- Raw CSV files in `data/raw/`.

**ADR Implements:** ADR-015 (dataset identity, raw contract, checksum manifest); ADR-003 amendment (two elasticity-side suites).

**Falsification:**
Generate a deliberately malformed raw fixture containing an orphan severity record (a `PolicyID` not present in frequency) and execute `validate_raw_gx`. Confirm that the GX checkpoint fails with a specific referential integrity violation and blocks downstream execution. Revert fixture and confirm green.

**Gate 1 — Hard Close: ✅ PASSED (2026-10-21)**
Raw GX suite passes clean on valid raw data; each malformed fixture fails its specific expectation; checksum verification fails on tampered data; orphan severity falsification halts the pipeline.

---

### Stage 2 — Deterministic Feature Pipeline (P2-F02) ✅ **COMPLETED**

**Goal:** Establish a single, deterministic feature transformation pipeline that emits two distinct views (GLM categorical/banded view and Causal Forest numeric/ordinal view) with zero leverage-inducing composite indices and strict exclusion of claims history from elasticity covariates.

**Evidence:** Executed outcomes for Stage 2 (421 exposure clips, 307/410 tail clips, deterministic hashes, 14/27-column views, metadata round-trip) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-2--deterministic-feature-pipeline).

**Inputs / Dependencies:**

- Stage 1 raw data contract validated and versioned.
- Stage 0 unbinned distribution metrics.

**ADR Implements:** ADR-012 (feature preparation, tail handling, zero composite indices).

**Falsification:** None (determinism is verified by repeated bit-for-bit execution; feature leakage is prevented by schema assertion).

**Gate 2 — Hard Close: ✅ PASSED (2026-10-21)**
Pipeline produces identical outputs on repeated runs; all clips and exclusions are logged with exact counts; composite indices are absent; claims variables are provably absent from elasticity covariates; transformer artifacts are versioned.

---

### Stage 3 — GLM Loss-Cost Baseline & Calibration Reporting (P2-F03) ✅ **COMPLETED**

**Goal:** Build a single, cap-parameterized Tweedie GLM (compound Poisson-Gamma) as the foundational actuarial loss-cost baseline and $Losses_0$ source, with a standardized calibration reporting suite.

**Evidence:** Executed outcomes for Stage 3 (selected `p=1.84`, holdout deviance 16.9016, decile/segment A/E, artifact hashes) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-3--glm-loss-cost-baseline--calibration-reporting).

**Inputs / Dependencies:**

- Stage 2 feature pipeline validated.
- Training and held-out test splits generated with fixed seed (`tier1_ml.random_state: 42`).

**ADR Implements:** ADR-012 (Tweedie baseline, compound Poisson-Gamma, calibration reporting); P2-D1.

**Falsification:** None.

**Gate 3 — Hard Close: ✅ PASSED (2026-10-21)**
Tweedie GLM converges without numerical errors; holdout mean deviance and decile A/E ratios are generated; Tweedie variance power has documented justification; MAE/RMSE are not used as primary pass/fail gates.

---

### Stage 4 — Segment Definition & Tier 1 Output Contract (P2-F05) ✅ **COMPLETED**

**Goal:** Formalize the 36-cell actuarial segment grid as the universal analysis unit across AEGIS, and define the typed Tier 1 Pydantic output contract with strict structural exclusion of proposed price adjustments.

**Evidence:** Executed outcomes for Stage 4 (35 realized segments, 2,234-policy merged cell, floor 1,067, portfolio loss cost €126.53/yr, artifact hashes) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-4--segment-definition--tier-1-output-contract).

**Inputs / Dependencies:**

- Stage 2 feature pipeline and Stage 3 GLM fitted values.
- Segment definition approved in ADR-014 / P2-D3.

**ADR Implements:** ADR-014 (segment definition, Tier 1 output contract, ratio-of-sums aggregation); P2-D3.

**Falsification:** An injected `proposed_adjustment: 0.05` payload is rejected by `Tier1SegmentOutput` (and all adjustment-like field names), so no proposed adjustment can leak into Tier 1 outputs.

**Gate 4 — Hard Close: ✅ PASSED (2026-10-21)**
All policies map to valid segments; every realized segment meets the ≥1,000-policy floor after the approved merge; ratio-of-sums aggregation passes unit tests; Pydantic validation blocks any `proposed_adjustment` field.

---

### Stage 5 — Qualification Q1: Loss Cap Sweep & Final Calibration ✅ **COMPLETED**

**Goal:** Execute qualification test Q1 across EDA-grounded candidate thresholds to select the optimal loss cap and portfolio excess load, closing the open P2-D1a level.

**Evidence:** Executed outcomes for Stage 5 (selected cap €100,000, excess load 1.1988, Spearman 0.9992 / deviance 0.406% at the passing transition, excess concentration 62.9×) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-5--qualification-q1-loss-cap-sweep--final-calibration) and [`q1_cap_sweep_report.md`](q1_cap_sweep_report.md).

**Inputs / Dependencies:**

- Stage 3 parameterized GLM baseline.
- Stage 4 segment definitions.

**ADR Implements:** ADR-012 (cap level selection, excess load calculation); closes P2-D1a.

**Falsification:** None (empirical qualification experiment).

**Gate 5 — Hard Close: ✅ PASSED (2026-10-24)**
Cap sweep report complete; selected cap €100,000 satisfies the Spearman ≥0.98 and deviance ≤1% criteria; excess-load distribution verified across segments (material concentration in one segment escalated to and resolved by the operator: uniform load + documented caveat); `tier1_ml.loss_cap = 100000.0` and `tier1_ml.excess_load = 1.1988` frozen in `params.yaml`.

---

### Stage 6 — Semi-Synthetic DGP & Training Contract (P2-F04) ✅ **COMPLETED**

**Goal:** Construct the isolated semi-synthetic data-generating process (DGP) with known ground truth, enforce strict physical separation of truth from estimator inputs, and gate training data ingestion with a dedicated Great Expectations contract.

**Evidence:** Executed outcomes for Stage 6 (synthetic treatment and outcome generator, truth definitions, bound guarantee, truth isolation) are recorded in [`stage_evidence.md`](stage_evidence.md#stage-6--semi-synthetic-dgp--training-contract).

**Actions:**

1. Create `reports/docs/phases/phase_2/004_semi-synthetic-dgp/`: write `spec.md`, obtain approval, write `plan.md` and `tasks.md`.
2. Implement the synthetic treatment and outcome generator in `src/aegis/pipelines/synthetic_dgp.py`:
   - Unit: all 413,169 policies with equal weight; observed exposure excluded from DGP covariates (A1, R12).
   - Treatment: price change $\Delta \in [-0.15, +0.15]$ generated with observed confounding via a risk-based pricing rule plus residual noise.
   - Outcome: logistic renewal probability $P(\text{renew} \mid X, \Delta) = \sigma(a(X) + b(X) \cdot \Delta)$; binary renewal indicator $Y \sim \text{Bernoulli}(P)$.
   - Truth definitions: baseline renewal $q_{0i} = \sigma(a(X_i))$; treatment effect $\tau_i = q_{0i}(1 - q_{0i}) \cdot b(X_i)$; elasticity $\epsilon_i = \tau_i / q_{0i}$.
   - Bound guarantee: enforce $\max |\epsilon_i| \cdot 0.15 < 1.0$ by construction so volume never floors at zero.
3. Generate three physically separate output artifacts:
   - Estimator training table: `PolicyID`, pre-treatment covariates $X$, treatment $T$, outcome $Y$, `segment_id`.
   - Truth sidecar table: `PolicyID`, $\tau_i$, $q_{0i}$, $\epsilon_i$ (stored in validation path, invisible to estimator).
   - DGP manifest: parameter values, random seeds, input data hash.
4. Implement automated DGP self-tests:
   - Estimand gap: numerical integration confirms closed-form $\tau_i$ matches local derivative within tolerance.
   - Overlap: residual variance $\text{Var}(T \mid X) \ge \text{floor}$.
   - Heterogeneity & dispersion: between-segment spread of $\tau$ exceeds expected estimator standard error.
   - Homogeneity: correlation between $\epsilon_i$ and expected loss cost within segment is $\le 0.05$.
5. Implement training-table GX contract (`data_contracts/elasticity_training_suite.json`):
   - Prohibit any truth columns (`tau`, `epsilon`, `q0`, `true_effect`) in the training table.
   - Enforce treatment bounds $T \in [-0.15, +0.15]$.
   - Enforce binary outcome $Y \in \{0, 1\}$.
6. Implement import-boundary test: assert that modules in `aegis.pipelines.causal` cannot import from `aegis.pipelines.truth_loader`.

**Inputs / Dependencies:**

- Stage 2 cleaned feature pipeline.
- Stage 5 finalized loss-cost baseline and parameters.

**ADR Implements:** ADR-013 (benchmark design, truth isolation, DGP self-tests); ADR-015 (training table GX contract); P2-D2, P2-D2a, P2-D2b, P2-D8.

**Falsification:**
Plant a synthetic truth column (`tau_ground_truth`) into a test training-table fixture and run the training GX checkpoint. Confirm the suite fails loudly with an unapproved-column error and halts the pipeline. Run the import-boundary test and confirm it fails if an import of the truth loader is injected into the causal estimator package.

**Gate 6 — Hard Close:** ✅ **PASSED (2026-10-26)**
All 5 DGP self-tests pass within stated tolerances; training-table GX suite passes valid data and rejects truth-column fixture; truth loader import boundary is verified; three separate physical artifacts are produced.

---

### Stage 7 — Qualification Q2: Estimator Training Scale & Budget ✅ **COMPLETED**

**Goal:** Execute qualification test Q2 across sample sizes and nuisance-model configurations to establish the optimal estimator training scale within runtime and memory budgets, closing the open P2-D5 decision.

**Evidence:**

On 2026-10-26, `uv run python scripts/q2_estimator_scale.py` completed all 8 cells
(seed 42; stratified 80/20 split; 35 held-out segments per cell). Both full-data fits met
the 30-minute/80%-RAM budget. Spearman was 0.9014 (GBT) and 0.9031 (linear); the 0.0017
difference is within the approved 0.02 tolerance, so GBT at N=413,169 was selected.
Frozen config: `training_sample_size: 413169`, `nuisance_model: "gbt"`, `cv_folds: 5`.

Full metrics and input hashes: [`q2_estimator_scale_report.md`](./q2_estimator_scale_report.md).
Validation: pytest **203 passed, 2 skipped**; Pyright **0 errors**; focused Ruff and
module-size checks passed. Repository Ruff has an unrelated existing E501 at
`src/aegis/pipelines/truth_loader.py:78`.

**Actions:**

1. Using the Stage 6 DGP training data and Stage 4 segment definitions, run `CausalForestDML` across the sample sizes resolved in Stage 0 (e.g. $n \in \{25\text{k}, 100\text{k}, 250\text{k}, n_{\text{max}}\}$) on pilot seeds.
2. Compare two nuisance model families:
   - Gradient-Boosted Trees (`GradientBoostingClassifier` / `GradientBoostingRegressor` from scikit-learn).
   - Regularized Linear Models (`LogisticRegressionCV` / `ElasticNetCV`).
3. For each configuration, record:
   - Wall-clock training and cross-fitting time.
   - Peak physical memory utilization.
   - Out-of-sample segment Spearman rank correlation against ground truth.
   - 95% confidence interval coverage of true segment effects.
4. Evaluate results against approved budget criteria:
   - Ceiling: full fit must complete in $\le 30$ minutes and utilize $\le 80\%$ of system RAM.
   - Selection rule: if full-data fit exceeds budget, choose the smallest $n$ whose segment Spearman correlation is within 0.02 of the largest tested $n$.
5. Write the qualification report in `reports/docs/phases/phase_2/q2_estimator_scale_report.md`.
6. Freeze the selected sample size $n^*$, cross-fitting folds ($cv=5$), and nuisance model choices in `params.yaml`.

**Inputs / Dependencies:**

- Stage 6 DGP training data.
- Stage 0 resolution of sample-size semantics.
- Stage 0 hardware baseline (V8).

**ADR Implements:** ADR-013 (estimator scale, nuisance configuration, runtime budget); closes P2-D5.

**Falsification:** None (empirical qualification experiment).

**Gate 7 — Hard Close:** ✅ **PASSED (2026-10-26)**
Q2 report written with zero skipped cells; scale and nuisance models selected per budget criteria; chosen parameters frozen in `params.yaml`; dated result recorded in `implementation_plan.md` §6.

---

### Stage 8 — Causal Estimator & Segment Inference (P2-F06) ✅ **COMPLETED**

**Goal:** Implement the production `CausalForestDML` training pipeline and segment-level inference engine under the frozen Q2 configuration, generating finite, self-consistent confidence intervals on segment-average treatment effects.

**Evidence:**

The full-data run produced 35 segments (82,634 held-out policies) with 100% finite and self-consistent intervals; the segment CSV was byte-identical across two consecutive full runs (`SHA-256: 48ee4db2c9557ec4778b9126a82bd633828c05731ffa818f2f3a401d02d75ad5`); the independent hand calculation matched `epsilon_hat = sum(tau_i) / sum(q0_i)` exactly; the fitted artifact is confined to gitignored `artifacts/` and serving isolation tests pass. Evidence: [`stage_evidence.md`](stage_evidence.md#stage-8--causal-estimator--segment-inference).

**Actions:**

1. Create `reports/docs/phases/phase_2/006_causal-estimator/`: write `spec.md`, obtain approval, write `plan.md` and `tasks.md`.
2. Implement estimator training in `src/aegis/pipelines/causal_estimator.py`:
   - Fit `CausalForestDML` on training split only (strictly zero evaluation on training data).
   - Use frozen Q2 parameters, nuisance models, and cross-fitting folds.
3. Implement segment-level inference:
   - Generate point estimates $\hat{\tau}_{\text{seg}}$ and $\hat{\epsilon}_{\text{seg}}$ aggregated via ratio of sums over segment policies.
   - Compute 95% confidence intervals on the **segment-average effect** (inference on the segment mean, not the average of individual policy interval bounds, per P2-D6).
4. Assert interval properties: verify all reported confidence intervals are strictly finite and contain their own point estimate ($\hat{\epsilon} \in [\text{lower}, \text{upper}]$).
5. Ensure estimator objects remain decoupled from the serving inference path (packaged as non-promotable run artifacts).

**Inputs / Dependencies:**

- Stage 6 DGP training data and truth sidecar.
- Stage 7 frozen estimator scale and nuisance configuration.

**ADR Implements:** ADR-013 (CausalForestDML, segment-average interval inference); P2-D5, P2-D6.

**Falsification:** None.

**Gate 8 — Hard Close:** ✅ **PASSED (2026-10-26)**
Estimator fits cleanly on training data; segment-average point estimates and intervals generated; 100% of segment intervals are finite and self-consistent; serving dependencies do not import the estimator model directly.

---

### Stage 9 — Validation Harness & Q3 Threshold Calibration (P2-F07) ✅ **COMPLETED**

**Goal:** Construct the multi-seed validation harness, prove its sensitivity by executing required negative controls, and calibrate numeric gate thresholds on pilot seeds (Q3) without peeking at final validation seeds.

**Evidence:**

Q3 executed on pilot seeds `[42, 101, 202]`; all three negative controls failed as required; the positive control passed with `m1=0.9152`; separation gaps were `m1=0.1930`, `m1ε=0.1652`, `m1b=0.1568`, `m2=0.0692`; `m5` was fixed at `0.90` under OD-5 and excluded from separation; thresholds were frozen in `params.yaml`. Evidence: [`stage_evidence.md`](stage_evidence.md#stage-9--validation-harness--q3-threshold-calibration).

**Actions:**

1. Create `reports/docs/phases/phase_2/007_validation-gate/`: write `spec.md`, obtain approval, write `plan.md` and `tasks.md`.
2. Implement validation metric calculation in `src/aegis/pipelines/validation_harness.py`:
   - $m_1$: Segment-level Spearman correlation $\rho(\hat{\tau}_{\text{seg}}, \tau_{\text{seg}})$.
   - $m_{1b}$: Policy-level Spearman correlation on held-out test policies.
   - $m_2$: Per-segment mean absolute elasticity bias $|\hat{\epsilon}_{\text{seg}} - \epsilon_{\text{seg}}|$.
   - $m_3$: 95% confidence interval empirical coverage of true segment effect.
   - $m_4$: Interval self-consistency rate (proportion of intervals that are finite and contain point estimate).
   - $m_5$: Directional sign agreement rate ($\text{sign}(\hat{\tau}) == \text{sign}(\tau)$).
3. Implement and execute negative controls (which MUST fail the validation gate):
   - $c_1$: Unadjusted estimator (naive difference in means on confounded data).
   - $c_2$: Shuffled estimator (randomly permuted treatment effect predictions).
   - $c_3$: Zero-effect DGP ($b(X) = 0$; proportion of segments flagged significant must remain near nominal $\alpha = 0.05$).
4. Implement and execute positive control: randomized treatment assignment variant (must pass cleanly).
5. Execute qualification test Q3 on **pilot seeds** (disjoint from final seeds):
   - Record metric distributions for estimator vs. negative controls.
   - Place each numeric pass threshold in the gap between the upper tail of the control distribution and the lower tail of the estimator distribution, enforcing a minimum margin.
   - If no separation exists, halt and return to benchmark/estimator design; do not lower thresholds.
6. Write `reports/docs/phases/phase_2/q3_threshold_calibration_report.md` and freeze final numeric thresholds in `params.yaml` under `tier1_ml.validation_thresholds`.

**Inputs / Dependencies:**

- Stage 8 causal estimator and segment inference.
- Pilot random seeds disjoint from final evaluation seeds.

**ADR Implements:** ADR-013 (validation metrics, negative/positive controls, threshold calibration); closes P2-D6a.

**Falsification:**
Run the three negative controls ($c_1$, $c_2$, $c_3$) through the validation harness. Confirm that each control independently fails the gate metrics. This provides concrete evidence that the gate cannot be passed by a trivial or defective estimator.

**Gate 9 — Hard Close:** ✅ **PASSED (2026-10-27)**
All three negative controls demonstrably fail the harness; the positive control passes (`m1=0.9152`); Q3 proves separation on `m1`, `m1ε`, `m1b`, and `m2`; `m5` is fixed at `0.90` under OD-5; numeric thresholds are frozen in `params.yaml` before final-seed evaluation.

---

### Stage 10 — Final-Seed Validation Gate (Roadmap Exit E1)

**Goal:** Execute the definitive multi-seed validation campaign across disjoint final seeds to prove recovery of the known causal treatment effect, formally satisfying Roadmap Exit Criterion E1.

**Actions:**

1. Execute the validation campaign across $K$ final seeds ($K \ge 5$, varying both DGP noise and estimator random seeds) on held-out test policies.
2. Compute the approved summary gate statistics across seeds:
   - **Worst-seed performance:** must exceed the frozen floor threshold for all metrics ($m_1$ through $m_5$).
   - **Median-seed performance:** must meet or exceed the target calibration threshold.
3. Generate the comprehensive validation report:
   - Summary table of metric values across all seeds.
   - Comparison against frozen Q3 thresholds.
   - Confirmation of zero tautological tests (e.g. asserting values equal themselves).
4. Persist the validation summary artifact (`artifacts/causal_validation_summary.json`) with cryptographic hashes of the frozen config and test data.
5. Apply post-hoc immutability check: verify that any attempt to alter a frozen threshold after the run invalidates the validation artifact checksum.

**Inputs / Dependencies:**

- Stage 8 trained estimator pipeline.
- Stage 9 frozen validation thresholds.
- Disjoint final seed family.

**ADR Implements:** ADR-013 (Exit E1 verification); P2-D6.

**Falsification:**
Attempt to tamper with a frozen threshold in `params.yaml` after the final seed run has completed. Confirm that the validation verification script detects the mismatch between the committed configuration hash and the validation artifact, failing verification.

**Gate 10 — Hard Close:**
Worst-seed and median-seed metrics beat frozen thresholds across all $K$ seeds; zero failed seeds; tamper check blocks post-hoc threshold edits; Roadmap Exit Criterion E1 is fully satisfied with recorded evidence.

> **❌ GATE 10 FAILED (2026-10-28).** Campaign executed as specified (K = 5, seeds 501–505, independent DGP noise per seed). 4 of 5 seeds breached the frozen rank floors; only seed 501 cleared them. `m2`/`m3`/`m4`/`m5` passed on every seed, so the failure is confined to rank recovery ($m_1$, $m_{1b}$). Root cause: rank recovery spans 0.55–0.95 across DGP-noise draws with the estimand held fixed; noise draw 42 — the only draw Q3 calibrated on — is the optimistic tail. Analysis: `reports/docs/phases/phase_2/stage10_failure_analysis.md`. Escalated to P2-D5a.

> **✅ GATE 10 CLOSED UNDER OPTION F AMENDMENT (2026-10-30).** Follow-up to failure analysis and P2-D5a Phase 1 sweep (`p2_d5a_phase1_sweep_report.md`): operator approved Option F formal amendment (`e1_option_f_amendment.md`). Roadmap Exit Criterion E1 is closed under the multi-metric governed recovery envelope: unbiased scale recovery ($m_2 \le 0.1167$, achieved $0.039 - 0.080$), honest interval coverage ($m_3 \ge 0.9000$, achieved $100\%$ under tree CATE), self-consistency ($m_4 = 1.0000$), sign agreement ($m_5 \ge 0.9000$), positive confounder separation ($\Delta(m_1) \ge 0.15$), and documented realization rank variance ($m_1 \in [0.64, 0.93]$, median $\ge 0.70$). Stage 10 PASSED under Option F; Stages 11–15 unblocked.

---

### Stage 11 — Hidden-Confounder Sensitivity Analysis (P2-F08) ✅ **COMPLETED**

**Goal:** Execute an unobserved-confounding sensitivity sweep over the known synthetic benchmark to establish the causal estimate's robustness value and generate the degradation curve feeding showcase scenario S4.

**Actions:**

1. Create `reports/docs/phases/phase_2/008_confounder-sensitivity/tasks.md`.
2. Implement hidden-confounder injection in `src/aegis/pipelines/sensitivity.py`:
   - Inject an unobserved confounder $U \sim \mathcal{N}(0, 1)$ simultaneously into treatment assignment $T$ and outcome $Y$ with varying coupling strengths $(\gamma_T, \gamma_Y)$.
   - Use a $3 \times 3$ grid plus baseline zero cell (10 cells total).
   - Hide $U$ from the causal estimator feature set.
3. Re-fit estimator across grid cells and measure degradation of metrics $m_1$ (Spearman correlation) and $m_2$ (elasticity bias).
4. Compute the **robustness value**: identify the minimum confounding strength at which median segment Spearman correlation falls below the frozen Stage 10 threshold.
5. Enforce fail-loud execution: assert that the harness raises an explicit exception if any cell in the sensitivity grid is skipped or fails to converge.
6. Record the sensitivity degradation curve in `artifacts/sensitivity_summary.json` and select the representative degraded case for showcase Scenario S4.

**Inputs / Dependencies:**

- Stage 6 DGP generation pipeline.
- Stage 10 validated causal estimator and frozen thresholds.

**ADR Implements:** ADR-013 (confounding sensitivity, robustness value, fail-loud execution); P2-D7.

**Falsification:** None (skipped-cell guard is built into the runner and asserted in the gate).

**Gate 11 — Hard Close:**
All 10 sensitivity grid cells execute successfully with zero skips; degradation curve and robustness value are recorded in the artifact; scenario S4 confounding parameters are locked.

> **✅ GATE 11 PASSED (2026-10-30).** All 10 sensitivity grid cells executed successfully with zero skips verified; degradation curve recorded in `artifacts/sensitivity_summary.json` (SHA-256 `e9da0a1b...`); robustness value $R_v = 0.5004$ recorded; Scenario S4 locked to cell `gt_0.05_gy_0.50` ($\gamma_T = 0.05, \gamma_Y = 0.50, r = 0.502, m_1 = 0.6579, m_2 = 0.7419$). Report: `reports/docs/phases/phase_2/confounder_sensitivity_report.md`. Stage 12 unblocked.

---

### Stage 12 — MLflow Artifact Packaging & Human-Gated Promotion (P2-F09)

**Goal:** Package Tier 1 outputs into a promotable, auditable registry artifact and enforce a strict human-review gate that permits `candidate` assignment by automated pipelines, but requires human sign-off for `approved` promotion.

**Actions:**

1. Create `reports/docs/phases/phase_2/009_registry-promotion/`: write `spec.md`, obtain approval, write `plan.md` and `tasks.md`.
2. Implement artifact packaging in `src/aegis/pipelines/registry.py`:
   - Package segment output table (`Tier1SegmentOutput` CSV/Parquet conforming to ADR-014).
   - Package GLM baseline calibration report and loss cap manifest.
   - Package causal validation summary and sensitivity report.
   - Attach complete lineage metadata: Git commit hash, DVC data hash, DGP manifest ID, MLflow run ID, gate pass tag.
   - Retain fitted `CausalForestDML` model as a non-promotable run artifact; do NOT include serialized model in the serving package.
3. Configure MLflow tracking using SQLite backend (`sqlite:///artifacts/mlflow.db`) with URI injected strictly via environment variable (`MLFLOW_TRACKING_URI`).
4. Implement promotion script (`scripts/promote_model.py`):
   - Pipeline execution automatically sets alias `candidate`.
   - Promotion to alias `approved` requires: human reviewer ID, hash of reviewed calibration report, and verification that all lineage tags are present.
   - Rejection rule: promotion fails closed if any lineage tag is missing or if the calibration hash does not match the artifact.
5. Implement rollback mechanism: re-pointing `approved` alias to the previous model version.
6. Verify serving load behavior: confirm that loading the model for inference reads plain segment tables and never deserializes a pickled scikit-learn or EconML estimator.

**Inputs / Dependencies:**

- Stage 4 Tier 1 segment output schema.
- Stage 5 GLM calibration report.
- Stage 10 causal validation summary.
- Stage 11 sensitivity summary.

**ADR Implements:** ADR-014 (artifact packaging, SQLite tracking, human-gated alias promotion, plain data serving); P2-D11, P2-D11a.

**Falsification:**
Attempt to run `promote_model.py` to assign the `approved` alias without providing the required reviewer ID and without a valid calibration report hash. Confirm that the promotion script aborts with a non-zero exit code and logs a typed security/governance violation.

**Gate 12 — Hard Close:**
`candidate` alias is set automatically by pipeline; promotion to `approved` is blocked without valid human review record and complete lineage; rollback successfully re-points alias; serving loads plain data without deserializing estimator objects; Exit E4 satisfied.

---

### Stage 13 — Diagnostic Showcase Slice (P2-F11)

**Goal:** Implement the glass-box developer showcase demo slice reading from the `approved` segment artifact, displaying only $\epsilon$, confidence intervals, and validation metrics, with the INV-5 exemption enforced by construction.

**Actions:**

1. Create `reports/docs/phases/phase_2/011_showcase-slice/`: write `spec.md`, obtain approval, write `plan.md` and `tasks.md`.
2. Configure the FastAPI serving layer (`src/aegis/api/`) and Jinja2 templates:
   - Connect `model_service.py` to resolve model metadata strictly from the MLflow `approved` alias.
   - Serve Chart.js as a local static asset (zero external CDN dependency in production/offline mode).
   - Add persistent warning banner to every page: _"Model validation demo on semi-synthetic data — not a pricing recommendation."_
3. Implement the four approved curated preset scenarios:
   - **Scenario S1 (Large segment):** Well-recovered driver segment with tight confidence intervals.
   - **Scenario S2 (Small segment):** Sparse policy segment exhibiting wide, honest confidence intervals.
   - **Scenario S3 (Cap-sensitive segment):** Segment whose uncapped loss cost was tail-inflated, displaying the impact of Stage 5 capping.
   - **Scenario S4 (Confounded segment):** Hidden-confounding degradation case from Stage 11, demonstrating estimator limits.
4. Enforce structural INV-5 compliance:
   - Showcase response schema contains strictly **zero proposed adjustment ($\Delta$) fields**.
   - Assert in unit tests that no proposed rate delta or published rate state can be emitted.
5. Re-enable and update existing showcase unit tests in `tests/unit/test_showcase.py`, removing the temporary skip marker.

**Inputs / Dependencies:**

- Stage 12 `approved` registered model artifact.
- Stage 5 cap sweep findings (for S3) and Stage 11 sensitivity findings (for S4).

**ADR Implements:** ADR-009 amendment (Phase 2 showcase scope); ADR-014; P2-D12.

**Falsification:**
Inject a dummy `proposed_rate_change` field into the showcase API response model and run the test suite. Confirm that schema tests fail loudly. Temporarily remove the disclaimer banner from `showcase.html` and confirm test `test_showcase_renders_default_registered_model_scenario` fails.

**Gate 13 — Hard Close:**
Showcase launches via `uv run uvicorn aegis.api.app:app`; all four preset scenarios render accurately from the `approved` artifact; response schema has no proposed adjustment field; disclaimer banner is present on all views; all tests pass without skips; Roadmap Exit Criterion E2 is met.

---

### Stage 14 — DAG Assembly, Config & CI Tiers / Reproducibility (P2-F10)

**Goal:** Assemble the complete 10-stage DVC pipeline DAG with strict INV-3 validation ordering, formalize execution tiers (T1 CI-smoke, T2 local repro, T3 validation campaign), and prove reproducible execution.

**Actions:**

1. Assemble `dvc.yaml` into the unified 10-stage DAG:
   $$\text{ingest\_raw} \longrightarrow \text{validate\_raw\_gx} \longrightarrow \text{build\_features} \longrightarrow \text{fit\_baseline} \longrightarrow \text{generate\_dgp} \longrightarrow \text{validate\_training\_gx} \longrightarrow \text{train\_estimator} \longrightarrow \text{validate\_recovery} \longrightarrow \text{register\_candidate} \longrightarrow \text{version\_artifacts}$$
2. Verify INV-3 DAG ordering: confirm that no stage producing versioned data can execute unless its upstream GX validation stage completes successfully.
3. Expand `params.yaml` and Pydantic configuration schemas (`src/aegis/config/schema.py`) to encompass all Phase 2 parameters:
   - Loss cap and excess load scalar.
   - Segment grid boundaries and merge threshold.
   - DGP parameters, treatment corridor width, and random seeds.
   - Validation pass thresholds.
   - MLflow tracking and artifact settings.
4. Implement execution tiers per P2-D10:
   - **Tier 1 (CI-smoke):** Minimal committed fixture exercising code paths, schema validation, and unit tests in GitHub Actions; enforce $\ge 65\%$ test coverage floor across `src/aegis/` Tier 1 modules.
   - **Tier 2 (Full local run):** Complete pipeline execution via `dvc repro` on full dataset, registering candidate artifact.
   - **Tier 3 (Validation campaign):** Multi-seed campaign and sensitivity sweep executed locally or on-demand before phase sign-off.
5. Prove bit-for-bit data reproducibility: run `dvc repro` twice from scratch on the same machine; verify that output hashes for cleaned data, DGP table, and truth sidecar are bit-identical.

**Inputs / Dependencies:**

- All features P2-F01 through P2-F11 completed.

**ADR Implements:** ADR-015 (tiered reproducibility, DAG ordering); ADR-003 amendment; P2-D8, P2-D10.

**Falsification:** None (reproducibility is verified by repeated bit-for-bit hash comparisons).

**Gate 14 — Hard Close:**
`dvc repro` executes cleanly end-to-end; data products reproduce bit-identically across consecutive runs; model metrics match within stated floating-point tolerances; CI-smoke suite passes with $\ge 65\%$ coverage; INV-3 ordering is demonstrable; Roadmap Exits E5 and E6 are met.

---

### Stage 15 — ADR Consolidation & Phase 2 Sign-Off

**Goal:** Close the loop between **decided $\to$ built $\to$ verified $\to$ recorded**, update the authoritative architecture ledger, complete the Post-Implementation Review, and verify that all conditions for advancing to Phase 3 are satisfied.

**Actions:**

1. Record final empirical qualification outcomes in `implementation_plan.md` §6:
   - Final chosen loss cap and excess load (closes Q1 / P2-D1a).
   - Final estimator scale, nuisance configuration, and runtime metrics (closes Q2 / P2-D5).
   - Final frozen validation thresholds and control separation margins (closes Q3 / P2-D6a).
2. Complete every row of the Post-Implementation Review table (`implementation_plan.md` §8): record actual empirical evidence, findings, severity, remediation, and final status for each of the 16 review items. Ensure zero `_pending_` rows remain.
3. Update `reports/docs/architecture/system_design.md`:
   - Mark ADR-012, ADR-013, ADR-014, and ADR-015 as **Validated** (confirming implementation matched the decisions without material deviation).
   - Confirm ADR-003 and ADR-009 amendment status lines reflect validated Phase 2 delivery.
4. Verify all Phase 2 Exit Criteria independently:
   - **E1:** Causal estimator recovers known synthetic effect with defensible per-segment metrics (Stage 10).
   - **E2:** Showcase slice renders correctly with INV-5 exemption honored (Stage 13).
   - **E3:** Baseline GLM calibrated with capped-loss treatment documented (Stage 3 & Stage 5).
   - **E4:** Promotable segment artifact under `approved` alias with recorded human review (Stage 12).
   - **E5:** INV-3 ordering demonstrable in DAG with failing fixtures for both elasticity suites (Stage 1 & Stage 6).
   - **E6:** CI-smoke tier green and full run reproducible within tolerance (Stage 14).
5. Verify Phase 3 Entry Preconditions:
   - Validated segment table under `approved` alias exposes $\hat{\epsilon}_{\text{seg}}$, 95% confidence intervals, baseline retention $q_0$, and base premium.
   - Bounded exploration corridor parameters ($\pm 15\%$) documented in relation to auto-approval threshold ($\pm 5\%$).
6. Update `reports/docs/implementation_roadmap.md` and `README.md` to reflect Phase 2 completion.
7. Present complete audit evidence and obtain formal human sign-off for Phase 2 close-out.

**Inputs / Dependencies:**

- Stages 0 through 14 passed and evidenced.

**ADR Implements:** ADR-012 … ADR-015 status finalization (marks Validated).

**Falsification:**
Documentation-to-reality audit: independently cross-check each exit criterion claim against test logs, git diffs, MLflow run artifacts, and DVC pipeline state. Any unsupported claim fails the gate.

**Gate 15 — Hard Close:**
All 15 stages passed; all 7 falsifications confirmed detected and reverted; Post-Implementation Review table 100% complete; ADR-012 … ADR-015 marked Validated; Roadmap Exits E1–E6 verified; formal human sign-off recorded. **Phase 2 is formally closed; Phase 3 may begin.**

---

## 6. Global Execution Controls & Change Handling

1. **The 80% Rule:** If the implementer is not at least 80% confident in a mathematical formulation, statistical library behavior, or pipeline configuration — **stop and ask**. Do not guess silently.
2. **Stop-the-Line Rule:** When any unexpected test failure, regression, or schema mismatch occurs during a stage, stop immediately. Preserve evidence, diagnose the root cause, fix the defect, verify the fix, and only then resume. Never push past a broken test.
3. **Change Management Protocol:** If implementation surfaces a required change affecting architecture, data contracts, invariants, external interfaces, or phase scope:
   - **Step 1:** Halt implementation on the affected stage.
   - **Step 2:** Formulate a change proposal documenting:
     ```text
     Original decision:
     New evidence:
     Proposed change:
     Impact on downstream stages:
     Recommendation:
     Operator decision:
     ```
   - **Step 3:** Update `implementation_plan.md` and obtain operator approval before resuming.
4. **Credential and Secret Hygiene:** Standing rule across all stages: zero committed secrets. Provider API keys, credentials, and tokens are never committed to version control, embedded in source, or logged.

---

## 7. Acceptance Evidence Checklist

Before Phase 2 can be declared complete at Stage 15, evidence for each item must be recorded:

- [x] **E1 (Causal Recovery):** Closed under Option F Amendment (`e1_option_f_amendment.md`, October 30, 2026). Unbiased scale ($m_2 \le 0.08$), 100% interval coverage ($m_3=1.0000$), 100% sign agreement ($m_5=1.0000$), positive control separation ($\Delta \ge 0.24$), and documented rank envelope ($m_1 \in [0.64, 0.93]$, median $\ge 0.76$).
- [ ] **E2 (Showcase Slice):** FastAPI application renders presets S1–S4 from `approved` artifact; test confirms zero proposed adjustment fields in API schema; persistent disclaimer banner present.
- [ ] **E3 (GLM Baseline):** Holdout Tweedie deviance, decile A/E ratios, and segment loss-cost report persisted; excess load calculation documented.
- [ ] **E4 (Promotable Artifact):** MLflow registry contains packaged segment table with full lineage; alias promotion script enforces human-review gate; serving loads plain data.
- [ ] **E5 (Data Contracts):** DVC DAG demonstrates raw and training GX gates upstream of versioning; both suites fail their respective malformed fixtures.
- [ ] **E6 (Reproducibility):** Data products reproduce bit-identically across repeated local `dvc repro` runs; CI-smoke suite passes in GitHub Actions with $\ge 65\%$ module coverage.
- [ ] **INV-3 Enforcement:** No dataset versioned without passing GX contract.
- [ ] **INV-5 Enforcement:** Tier 1 output contract and showcase schema contain zero proposed adjustment fields; diagnostic exemption banner active.
- [ ] **INV-8 Enforcement:** `uv run python scripts/check_module_size.py` passes clean across all Python modules under `src/` (zero files exceeding 1,000 lines).
- [ ] **INV-10 Enforcement:** Zero live carrier core-system integrations; intake operates strictly on public benchmark data.

---

## 8. Summary Table & Stage Record

| Stage  | Name                      | Primary Deliverable                     | ADRs         | Falsification              | Hard Gate Summary                              | Status         |
| ------ | ------------------------- | --------------------------------------- | ------------ | -------------------------- | ---------------------------------------------- | -------------- |
| **0**  | Baseline & Preconditions  | Verified environment, recorded V-checks | None         | Evidence audit             | All preconditions met; baseline stable         | ✅ **PASSED**  |
| **1**  | Raw Intake & Contract     | Checksum manifest, raw GX suite         | ADR-015      | Orphan severity halts DAG  | Malformed raw data blocked before versioning   | ✅ **PASSED**  |
| **2**  | Feature Pipeline          | Dual-view pipeline, clip logs           | ADR-012      | None                       | Deterministic; clips logged; no leakage        | ✅ **PASSED**  |
| **3**  | GLM Loss-Cost Baseline    | Parameterized Tweedie GLM & report      | ADR-012      | None                       | Parameterized fit; deviance & A/E deciles      | ✅ **PASSED**  |
| **4**  | Segments & Tier 1 Schema  | 36-cell grid, Pydantic schema           | ADR-014      | Injected Δ rejected        | No Δ field; merge floor met; ratio-of-sums     | ✅ **PASSED**  |
| **5**  | Qualification Q1          | Selected cap level & excess load        | ADR-012      | None                       | Spearman $\ge 0.98$; deviance $\le 1\%$        | ✅ **PASSED**  |
| **6**  | Semi-Synthetic DGP        | Training table, truth sidecar, suite    | ADR-013, 015 | Planted truth fails suite  | DGP self-tests pass; truth isolation proven    | ✅ **PASSED**  |
| **7**  | Qualification Q2          | Scale, nuisance family & report         | ADR-013      | None                       | Budget met; scale & nuisance family selected   | ✅ **PASSED** — N=413,169, GBT, 10.16 min |
| **8**  | Causal Estimator          | Trained forest & segment intervals      | ADR-013      | None                       | 35 segments; 100% finite/self-consistent; byte-identical CSV | ✅ **PASSED** |
| **9**  | Validation Harness & Q3   | Metric harness & frozen thresholds      | ADR-013      | Negative controls fail     | Controls fail; threshold separation proven     | ✅ **PASSED** |
| **10** | Final-Seed Validation     | Multi-seed evaluation report (E1)       | ADR-013      | Threshold tamper detected ✅ (unit-level; artifact is schema-v1) | Worst-seed & median beat frozen thresholds     | ✅ **PASSED** — Closed under Option F Amendment (October 30, 2026) |
| **11** | Confounder Sensitivity    | Degradation curve & robustness          | ADR-013      | None                       | Full grid executed; robustness value recorded  | ✅ **PASSED** — 10/10 cells; Rv=0.5004; S4 locked |
| **12** | Registry & Promotion      | Packaged artifact, alias promotion      | ADR-014      | Unapproved promotion fails | Pipeline sets candidate; approved needs review | ⬜ Not started |
| **13** | Showcase Slice            | 4 presets from `approved` (E2)          | ADR-009, 014 | Response schema rejects Δ  | 4 presets render; banner on all views          | ⬜ Not started |
| **14** | DAG, Config & CI Tiers    | 10-stage DAG, CI-smoke (E5/E6)          | ADR-015, 003 | None                       | Bit-identical repro; CI green ($\ge 65\%$ cov) | ⬜ Not started |
| **15** | ADR Consolidation & Close | Validated ADRs, PIR, sign-off           | ADR-012…015  | Documentation audit        | E1–E6 verified; PIR complete; human sign-off   | ⬜ Not started |
