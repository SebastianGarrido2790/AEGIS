# Stage 10 — Gate Failure Analysis and E1 Verdict

**Date:** 2026-10-28
**Status:** Exit Criterion **E1 = FAILED** (recorded, not remediated)
**Decision:** Operator approved Option A — record the failure, escalate to P2-D5; no threshold recalibration, no rerun.
**Supersedes:** the withdrawn P2-D6a amendment proposal (recalibrate Q3 across DGP-noise draws).

---

## 1. Verdict

Roadmap Exit Criterion E1 — "the causal estimator recovers the known synthetic treatment effect with defensible per-segment and rank-based metrics" — **is not met.**

The Stage 10 campaign ran as designed: $K = 5$ final seeds `[501, 502, 503, 504, 505]`, disjoint from the Q3 pilot seeds `[42, 101, 202]`, each with independent DGP noise (`seed + 10000`) and an independent estimator random state, evaluated only against thresholds frozen at Stage 9.

**4 of 5 seeds breached the frozen floors.** Only seed 501 cleared them.

| Seed | m1(tau) | m1(eps) | m1b | m2 | Floors |
|---:|---:|---:|---:|---:|:---:|
| 501 | 0.8737 | 0.8910 | 0.8854 | 0.0795 | ✅ |
| 502 | 0.6813 | 0.7234 | 0.5775 | 0.0729 | ❌ |
| 503 | 0.6796 | 0.7308 | 0.7621 | 0.0723 | ❌ |
| 504 | 0.5517 | 0.6345 | 0.7308 | 0.0783 | ❌ |
| 505 | 0.7762 | 0.7878 | 0.7017 | 0.0681 | ❌ |

Frozen floors: m1(tau) 0.8428 · m1(eps) 0.8831 · m1b 0.7784 · m2 ceiling 0.1167.

Only the three **rank-recovery** metrics failed. `m2` (bias), `m3` (interval coverage), `m4` (interval self-consistency) and `m5` (sign agreement) passed on **every** seed — `m3`/`m4`/`m5` at 1.000 throughout.

The verdict does not depend on the gate's aggregation framing. The Stage 10 module was subsequently corrected (separate `passes_floor` / `meets_target`, worst-seed naming, median-target reporting); under either framing, 4/5 seeds sit below the m1(tau) floor.

## 2. Root cause: rank recovery is unstable across DGP-noise draws

A variance-decomposition probe (`scripts/d1_variance_probe.py`) refit the Q2-frozen estimator on the **fixed seed-42 DGP tables** using the failing split/estimator seeds. Recovery returned to 0.9046 / 0.9295. That isolated the variation to the **DGP-noise axis**, not the split or estimator seed.

An independent sweep across fresh noise draws (full $n = 413{,}169$, split seed 42) confirms the magnitude:

| DGP noise | Estimator m1(tau) | C1 unadjusted control m1(tau) |
|---|---:|---:|
| 42 (reference) | 0.8858 | 0.6928 |
| 10042 | 0.8193 | 0.4952 |
| 10101 | 0.6956 | 0.2822 |
| 10202 | 0.8023 | 0.5624 |

With split seeds varied on draw 10101: 0.6956 / 0.5941 / 0.6587.

**The estimand is identical across all draws** — `true_tau` is a function of features only, verified constant to 4 dp (mean −0.3680, sd 0.0758, between-segment sd 0.1156). Only the realized treatment noise and the Bernoulli renewal draw change.

**Noise draw 42 — the draw underlying Stage 8, Q2 and Q3 — is the optimistic tail of this distribution.** The Q3 pilot seeds all shared it, so the frozen floors were calibrated against an unusually favourable realization.

## 3. Why recalibration was not the answer

The proposed amendment (3 noise draws × 3 seeds, unchanged separation-gap method, $\Delta_{min} = 0.15$) cannot produce a pass. Measured:

| Variant | Estimator min | Control max | Gap | Result |
|---|---:|---:|---:|---|
| Pooled, all draws (as proposed) | 0.5941 | 0.6928 | −0.0987 | ❌ no separation |
| Pooled, fresh draws only | 0.5941 | 0.5624 | 0.0317 | ❌ < 0.15 margin |
| Per-draw paired (best case) | — | — | 0.240–0.324 | ✅ separation, but floor ≈ 0.7124 |

The per-draw paired variant is the only one that establishes separation, and its conservative floor (0.7124) still exceeds four of the five observed Stage 10 seeds (0.5517 / 0.6796 / 0.6813 / 0.7762).

Every route to a "pass" therefore required lowering a floor to fit the observed result — explicitly forbidden by P2-D6a ("thresholds are not lowered to fit"). The frozen thresholds were left untouched.

## 4. Interpretation (carried to P2-D5)

Bias is stable and small in absolute terms ($m_2 \in [0.068, 0.080]$ across the five campaign seeds and the probe refits; $0.082$–$0.098$ on the three Q3 pilot seeds), and coverage/sign are perfect. What is fragile is **ranking**, and the ratio explains it:

> mean absolute bias ≈ 0.07–0.08 against a between-segment effect spread (sd) of 0.1156 → bias/signal ≈ 0.6–0.7.

A bias of that size relative to the segment-to-segment signal is large enough to reorder segments from one realization to the next. This is a property of the estimator's precision under this DGP, not a defect in the gate, and not something a threshold change can address.

Candidate directions for the P2-D5 estimator-family decision (not decided here):

1. **Reduce bias** — richer nuisance specification, more cross-fitting folds, or a smoothed/driver-model estimator against the known DGP structure.
2. **Increase the between-segment signal** — revisit the DGP `b(X)` coefficient spread (P2-D2b) so the estimand separates more strongly; this changes the benchmark, so it reopens D2b.
3. **Change the gate metric** — replace per-segment rank recovery with a coarser, better-conditioned statistic (e.g. top-k segment identification or tier-level ordering). This weakens the claim and needs a superseding decision.

Direction 1 preserves the existing benchmark and is the cheapest to test. Direction 2 is the only one that would make the current floors attainable, but it deliberately hardens the benchmark and must not be mistaken for fixing the estimator.

## 5. Evidence

| Item | Value |
|---|---|
| Campaign artifact | `artifacts/causal_validation_summary.json` |
| SHA-256 | `a716acb72971e58959efdf0de168b0d2b18d950931ff473bd939b3181269e789` |
| Size | 18,699 B |
| Generated | 2026-10-27T17:23:36+00:00 |
| Frozen config SHA-256 | `2eb6ff23488041e933d4d5cb00dbc6d481b9f4d2dccbc3e7c1ab0d1618fb5f18` |
| Report | `reports/docs/phases/phase_2/final_seed_validation_report.md` |
| Probe | `scripts/d1_variance_probe.py` → `logs/d1_probe.out` (local, gitignored) |

Tamper falsification (threshold edit → verification exits 1 → restore → exits 0) is covered by `tests/unit/test_final_seed_validation.py::test_falsification_threshold_tamper_is_detected` and `tests/unit/test_cli.py`; it validates the mechanism, and does not affect the E1 verdict.

### 5.1 Artifact schema drift (disclosed)

The campaign artifact carries `artifact_version: 1.0` and was written by the Stage 10 module *before* its gate aggregation was corrected. It therefore does not satisfy the corrected `FinalSeedValidationSummary` schema:

- extra: `campaign_gates.median_seed_all_floors_cleared`
- missing: `campaign_gates.median_seed_targets_met`
- missing: `campaign_gates.campaign_passed`

Running `verify-final-validation` against it fails closed with `malformed_summary` (exit 1). This is a schema-version mismatch, **not** a tamper finding, and it is not a pass.

Consequences, stated plainly:

- The artifact is retained unmodified as the historical record; it has not been rewritten or replaced.
- The E1 verdict does not depend on it: the per-seed metrics above are plain values, independently re-read from the artifact, and 4/5 sit below unmodified frozen floors.
- The threshold-tamper falsification is evidenced at unit level against the corrected schema. Artifact-level tamper verification would require re-emitting the artifact, which means rerunning the campaign (~1 h); it would not change the verdict.
- Aliasing `median_seed_all_floors_cleared` to `median_seed_targets_met` was rejected: the two are semantically different (median-vs-floor gate became median-vs-target report), and aliasing would silently reinterpret evidence.

## 6. What this blocks

Stages 11–15 remain blocked. E1 is a hard precondition for Roadmap Exit E1 and Phase 2 close. Phase 3 may not begin until either the estimator meets the frozen floors or a superseding decision (new ADR) redefines E1.
