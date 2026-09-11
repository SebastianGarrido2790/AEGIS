# Phase 2 Remediation Plan — Causal Validation, Test Suite & Data Quality

**Purpose:** close the gap between Phase 2's claimed completion (`phase_2_evaluation_report.md`'s "🟢 Complete", `phase_2_execution_workflow.md`'s Gate 4 "PASSED", `system_design.md`'s ADR-014 "Validated") and what the code, artifacts, and test assertions actually demonstrate. Until this plan's final gate passes, **Phase 2 sign-off remains withheld** — the prior "complete" markers reflected tests that passed without testing the right thing, not a false report.

**Rule for every stage:** same discipline as every gated workflow in this project — the gate at the end of a stage must pass before the next stage begins. No code is written as part of *this document*; this is the plan, not the fix.

**Why the causal fixes are grouped the way they are, not in the order listed:** Fix #1 and Fix #2 are not independent — changing how `treatment_rate_change` is generated changes what the mathematically correct "true effect" formula even is. Fixing #2 before #1, or separately, would mean writing a correct ground-truth check against an already-wrong data-generating process. They're one stage, not two, for that reason. Fix #3 and Fix #4 are genuinely independent of #1/#2 and of each other — a code-correctness bug in the estimation module, and a silently-swallowed exception in the refutation module respectively — so they're sequenced separately, but Fix #3 comes first only because it's the simpler, faster, lower-risk change to verify in isolation.

---

## Stage 0 — Baseline Capture

**Objective:** record the current, broken state as evidence — not just a claim — before anything changes, so the eventual "fixed" state has something concrete to be compared against.

**Steps:**
1. Archive the current `data/validated/causal_elasticity.json`, `data/validated/glm_baseline.json`, and the full `test_causal_elasticity.py` file exactly as reviewed, under a dated subfolder (e.g., `reports/docs/evaluations/phase_2_pre_remediation/`) — not deleted, not overwritten, kept as the documented "before."
2. Record, verbatim, the three findings this baseline demonstrates: the point estimate (283.05) falling outside its own reported interval `[-188.00, 220.24]`; the `diagnostic_note` proving the DoWhy fallback fired; the correlation (0.23) against the wrong ground-truth formula.

**Gate 0 — must pass before Stage 1 begins:**
- The pre-remediation artifacts and test file are archived and referenced by path from this plan.
- The three baseline findings are written down as concrete numbers, not paraphrased — so Stage 6's regression pass has an exact "before" to diff against, not a memory of one.

---

## Stage 1 — Redesign the Synthetic Data-Generating Process (Fixes #1 and #2)

**Objective:** give the causal estimator a data-generating process with genuine identifiable signal, and a ground-truth check that's actually validating the effect, not the treatment-assignment formula.

**Steps:**
1. **Fix #1 — independent stochastic variation in treatment assignment.** Add a meaningfully sized, independently drawn random component to `treatment_rate_change` (not just to the outcome, which already has one) — `treatment_rate_change` becomes a function of `risk_index` *plus* independent noise, not `risk_index` alone. The systematic, `risk_index`-driven part of treatment will still be fully explainable by `X` in a Double-ML residualization step — that's expected and correct. The fix is ensuring a non-trivial *residual* exists after that explaining-away, since that residual is the only thing a Double-ML estimator can actually learn from.
2. **Decide, explicitly, what stays in `X`.** `risk_index`, `driver_risk_score`, `vehicle_risk_score`, and `exposure_normalized` all currently sit in `X` together, and `risk_index` is a deterministic combination of the other two plus `exposure_normalized` — collinear by construction. Recommended resolution: keep them in `X` (segment heterogeneity is the actual point of using a forest-based estimator — these are the effect-modifying covariates that should drive *heterogeneous* effects), and rely on Step 1's independent noise to supply the residual variation Double-ML needs. This needs to be a stated decision in the code's docstring, not an implicit assumption a future reader has to reconstruct.
3. **Fix #2 — ground truth must be the actual causal effect, not the assignment formula.** The true effect is the analytic derivative of the outcome-generating function with respect to treatment, not the formula used to assign treatment in the first place — those are two different functions, and the current code copies the wrong one into `ground_truth`. Derive the ground-truth-checking formula programmatically from the *same* function that builds the synthetic outcome — one function computes the outcome, a second function (or the same function's known derivative) computes what the effect *should* be — so the two can never independently drift the way they already have once.
4. Re-derive the DGP's outcome formula alongside Step 1's treatment change, since the two need to stay mathematically consistent with each other (the outcome must actually respond to the now-noisier treatment in a way Step 3's ground-truth formula correctly describes).

**Gate 1 — must pass before Stage 2 begins:**
- `treatment_rate_change` has confirmed non-trivial variance left over after regressing it on `risk_index` alone — checked directly (e.g., residual variance is a meaningful fraction of total variance), not assumed from the code reading correct.
- The ground-truth formula used for validation is derived from, or provably identical to, the analytic effect of the outcome-generating function — not a separately hand-written formula that happens to look plausible.
- The decision on what stays in `X` is documented inline, with the reasoning from Step 2 above, not left implicit.

---

## Stage 2 — Correct Point Estimate and Confidence Interval Computation (Fix #3)

**Objective:** the point estimate and its reported interval need to come from the same, untransformed quantity — right now they don't, which is why the point estimate falls outside its own interval.

**Steps:**
1. Remove the `positive_effect = clip(raw_effect - min(raw_effect), ...)` transformation entirely. Once Stage 1 produces a DGP with a real, well-defined effect (likely uniformly signed by construction if that's the intended design), there should be no need for a defensive post-hoc positivity shift — and if the estimator's raw output *does* legitimately span zero or go negative in some segments, that's information worth seeing, not something to hide behind an artificial clip.
2. Compute `average_treatment_effect` as the direct mean of the model's raw, untransformed effect output.
3. Compute the reported interval from `model.effect_interval()` on that same untransformed array — same source, same transformation (none), for both statistics.
4. Add an explicit internal consistency check: the computed point estimate must fall within its own reported interval bounds before the result is considered valid — this becomes a hard assertion in the code itself, not just something checked later in a test (see Stage 4).

**Gate 2 — must pass before Stage 3 begins:**
- The point estimate and interval are demonstrably computed from the identical array, with no divergent transformation between them.
- On a re-run against Stage 1's corrected DGP, the point estimate falls inside its own reported interval — checked directly, not assumed from the code change alone.

---

## Stage 3 — Repair DoWhy Refutation Execution (Fix #4)

**Objective:** make the refutation suite actually run, and make its failure — if it ever fails again — loud, not silently replaced with plausible-looking placeholder numbers.

**Steps:**
1. Diagnose and fix the root cause of the `"This estimator does not support X=None!"` failure — almost certainly `CausalModel(...)` not declaring `effect_modifiers`, so the wrapped `CausalForestDML` estimator never receives the `X` it requires. Pass the heterogeneity feature columns as `effect_modifiers` explicitly. **Confidence note:** DoWhy's exact current API surface for this parameter should be verified against current documentation at implementation time, not assumed from memory, per the project's standing 80% confidence rule.
2. Remove the bare `except Exception: summary["diagnostic_note"] = str(exc)` fallback pattern. If the refutation genuinely fails to run after the fix in Step 1, that is a hard failure — raise, surfaced loudly, never silently replaced with the hardcoded `{p_value: 0.42, passed: true}`-style placeholder currently sitting in the function's default.
3. Confirm all three refuters (placebo treatment, random common cause, data subset) execute against live DoWhy output, not the fallback dict, on a clean run.
4. Document, in the code, which direction a "pass" means for each refuter type — a high p-value on a placebo/random-common-cause/subset refuter generally means "failed to reject the null of no meaningful change," the opposite convention from a typical significance test. This was flagged as ambiguous in review; it needs to be unambiguous in the code, not just correct by accident.

**Gate 3 — must pass before Stage 4 begins:**
- A clean run produces real DoWhy refutation output with no `diagnostic_note` fallback marker anywhere in the resulting artifact.
- Deliberately breaking the DoWhy call (e.g., temporarily reverting Step 1's fix) causes the pipeline to fail loudly — confirmed directly, not assumed — rather than silently substituting placeholder values.

---

## Stage 4 — Rebuild the Test Suite With Assertions That Can Actually Fail

**Objective:** replace every tautological or effectively-unfalsifiable assertion identified in review with one that would genuinely fail against the pre-remediation (Stage 0) code, and pass only against the Stage 1–3 fixes.

**Steps:**
1. Remove or replace the assertions that can never fail: `lower == pytest.approx(lower)` / `upper == pytest.approx(upper)` (tautological against themselves), `baseline_mae >= 0.0` (true by definition of MAE), and the key-existence-only check on `refutation_summary` (passes identically whether DoWhy ran or the fallback fired).
2. Add a direct regression test for the Stage 2 bug class: assert the point estimate falls within its own reported confidence interval. This is the single test that would have caught the original defect immediately.
3. Add a direct regression test for the Stage 3 bug class: assert `refutation_summary` contains no fallback/diagnostic-note marker, i.e., that DoWhy's real execution path was actually taken.
4. Add a dedicated test for Stage 1's fix: assert `treatment_rate_change` retains meaningful variance after accounting for `risk_index` — guards specifically against someone reverting to a fully-deterministic treatment assignment in the future.
5. Add a "ground-truth consistency" test for Stage 1's fix: assert the ground-truth-checking formula and the outcome-generating formula are the same function (or provably derived from one), not two independently maintained formulas — guards against the exact drift that caused Fix #2's original bug.
6. Raise the correlation threshold from `>= 0.1` to a level actually calibrated against what the corrected estimator produces on a clean run — not chosen in advance, since the right bar depends on how much identifiable signal Stage 1's redesign actually leaves.
7. **Improve logging, not just assertions:** the underlying functions (Stage 1–3's fixes) should log when imputation, fallback, or clipping behavior would have fired, so a future test failure comes with actionable context in the log output, not just a failed assertion with no trail back to cause.
8. **Falsification check on the test suite itself** — the same technique already used at the close of both Phase 1 and Phase 2: temporarily reintroduce the original Stage 0 defects (the positivity shift, the deterministic treatment, the bare exception handler) one at a time, and confirm the rebuilt test suite actually fails on each. A test suite that's never seen a real failure hasn't demonstrated it can catch one.

**Gate 4 — must pass before Stage 5 begins:**
- Every assertion identified as tautological in review has been replaced or removed.
- The falsification check in Step 8 confirms the rebuilt suite fails against each of the three original defects, individually reintroduced, and passes again once reverted.
- The full `test_causal_elasticity.py` suite passes cleanly against the Stage 1–3 fixed code.

---

## Stage 5 — Imputation Logging and Split-Logic Documentation

**Objective:** the two "other issues" named directly — neither requires a behavior change to code that's already correct, but both need to stop being silent.

**Steps:**
1. **Imputation logging (`driver.py`, `vehicle.py`).** The `.fillna(median)` calls are a reasonable choice for a public benchmark dataset, but currently leave no trace that imputation happened. Add an explicit count of imputed values per column, logged at feature-matrix build time — so a future reader can see, e.g., "driver_age: 0 imputed" and trust the number, rather than having to infer it never happened.
2. **Split-logic clarification (`pipeline.py`).** `create_policy_split`'s logic is confirmed correct — the string-sort-then-permute approach doesn't affect which policies land in train vs. test, only an arbitrary (but seed-stable) ordering that has no bearing on split validity. Add an inline comment stating this explicitly, including a note on the `"POL-10"` sorting before `"POL-2"` lexicographic quirk, so a future reader doesn't mistake deliberate, harmless behavior for a bug.

**Gate 5 — must pass before Stage 6 begins:**
- A feature-matrix build against a deliberately-nulled test fixture produces a visible, correct imputation count in the log output.
- The split-logic comment is in place and technically accurate against the actual sampling mechanism, not a simplified paraphrase of it.

---

## Stage 6 — Full Regression Pass and Re-Verification

**Objective:** confirm the fixes work together, not just individually — and produce the real, corrected numbers this phase's sign-off will actually rest on.

**Steps:**
1. Re-run the complete Stage 3/4/5 pipeline from the original Phase 2 execution workflow (GLM baseline, causal elasticity fit, MLflow tracking/registration) against the Stage 1–5 fixes.
2. Re-run the full regression and falsification pass from the original Phase 2 Stage 8 — lint, type-check, module-size, `dvc repro`, full `pytest` — confirming nothing outside this remediation's scope broke.
3. Compare the new `causal_elasticity.json` against Stage 0's archived baseline directly: correlation, whether the point estimate falls inside its interval, and whether the refutation summary is free of fallback markers.

**Gate 6 — must pass before Stage 7 begins:**
- The three Stage 0 baseline findings are each demonstrably resolved, shown side-by-side against the corrected artifact, not just asserted as fixed.
- Full CI, including Phase 2's original gates, is green.

---

## Stage 7 — Documentation Reconciliation

**Objective:** the project's own Update Protocol is explicit that a diverged implementation gets logged as an **Amended** ADR entry, not a silent re-validation — this stage is where that actually happens, closing the loop the same way every prior remediation in this project has.

**Steps:**
1. Amend ADR-014 in `system_design.md` with a new, dated entry describing what was wrong (the three root causes from this plan), not just updating its status label in place.
2. Update `phase_2_evaluation_report.md` §4 with the corrected causal validation numbers, replacing the ones this remediation invalidated — and preserve, rather than delete, a note that the original numbers were found unsound and why, consistent with how this project has always treated its own decision and evaluation record as a trail, not a place to quietly overwrite history.
3. Update `phase_2_execution_workflow.md`'s Gate 4 evidence to reflect the corrected numbers and the fact that the original "PASSED" reflected assertions that didn't test the right thing.
4. Re-attach the corrected evaluation report to the MLflow run per the original byte-for-byte identity requirement (Stage 6/ADR-016), since the report content has changed.

**Gate 7 — Phase 2 sign-off condition:**
- ADR-014 carries a dated Amended entry, not a silently updated Validated status.
- Every document that previously stated or implied the causal validation had succeeded now reflects the corrected numbers and an honest account of what was wrong — nothing is left overwritten without a trace.
- Only once this gate passes does Phase 2 carry genuine sign-off — superseding, not quietly replacing, its prior "Complete" markers.

---

## Summary Table

| Stage | Addresses | Gate proves | Blocks until gate passes |
|---|---|---|---|
| 0 | Baseline capture | The "before" state is documented evidence, not a memory | Stage 1 |
| 1 | Fix #1, Fix #2 | Treatment has real residual variance; ground truth matches the actual effect, not the assignment formula | Stage 2 |
| 2 | Fix #3 | Point estimate and interval come from the same untransformed quantity | Stage 3 |
| 3 | Fix #4 | DoWhy actually runs; a real failure surfaces loudly, never as a fake pass | Stage 4 |
| 4 | Test suite | Every assertion can genuinely fail; falsification check proves it | Stage 5 |
| 5 | Imputation logging, split-logic docs | Silent behavior becomes visible; correct behavior becomes legible | Stage 6 |
| 6 | Full regression | All fixes hold together; Stage 0's findings are demonstrably resolved | Stage 7 |
| 7 | Documentation reconciliation | ADR-014 amended, not silently re-validated; every affected document tells the truth | Phase 2 sign-off |
