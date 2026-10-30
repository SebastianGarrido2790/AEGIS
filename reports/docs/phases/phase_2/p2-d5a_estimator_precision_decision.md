# Decision Record — P2-D5a: Estimator precision under DGP-noise variation

**Status:** APPROVED by operator (October 29, 2026).
**Date:** 2026-10-29
**Nested under:** P2-D5 (estimator configuration and training scale — approved, resolved by Q2)
**Trigger:** Gate 10 FAILED; Exit Criterion E1 not met (`stage10_failure_analysis.md`)
**Governing:** ADR-013 Amendment (October 30, 2026)

---

## 1. Question

The frozen estimator misses the frozen gate on rank recovery. What changes — and in what
order — without lowering the gate to fit the observed result?

---

## 2. Facts (measured, not inferred)

| Fact | Value | Source |
|---|---|---|
| Gate 10 outcome | 4/5 seeds below floors; m1(τ) 0.5517–0.8737 vs floor 0.8428 | `artifacts/causal_validation_summary.json` |
| Rank recovery across DGP-noise draws | 0.59–0.82 (fresh draws), 0.89–0.95 (draw 42) | noise-draw sweep, §2 of failure analysis |
| Mean absolute bias $m_2$ | 0.068–0.080 campaign/probe; 0.082–0.098 Q3 pilots | same |
| Between-segment true-τ sd | 0.1156 (35 segments, min 1,067 policies) | truth sidecar |
| Coverage / self-consistency / sign ($m_3,m_4,m_5$) | 1.000 on every seed | artifact |
| Cost of one full fit + aggregate + metrics | 476–495 s on 12 logical CPUs | measured |
| Q2 nuisance comparison (forest CATE unchanged) | GBT 0.9014 vs linear 0.9031 on seed 42 | `q2_estimator_scale_report.md` |

**What is *not* established.** The Q2 linear comparison varied only the *nuisance* models
(`model_y`, `model_t`); the CATE model stayed a forest. No evidence exists on a parametric or
series CATE model.

---

## 3. Inference: two separate defects, two separate levers

Bias ($m_2$) stayed within 0.068–0.080 while rank collapsed from 0.88 to 0.55 on the same
estimand. That separates the problem into two components:

1. **A bias ceiling** — mean absolute bias ≈ 0.07–0.08 against a between-segment sd of 0.1156.
   Bias is roughly 0.6–0.7× the signal it must rank. This caps attainable rank correlation even
   on a favourable draw.
2. **A variance floor** — the draw-to-draw swing is not explained by bias (bias was flat). It is
   precision: segment-level estimates move enough between realizations to reorder segments.

**The gate fails on the floor, not the ceiling.** Worst seed 0.5517 vs floor 0.8428 is a variance
failure. Bias reduction alone cannot clear it.

Also unproven: whether the swing is Monte-Carlo noise *inside* the forest (forest `random_state`,
tree count, leaf size) or sampling noise *in the data* (irreducible given the DGP). Those imply
different options, and one cheap test discriminates them (§6, Phase 0).

---

## 4. Two methodological defects that must be settled regardless of the estimator

Neither is a threshold change; both are corrections that make any future gate meaningful.

### 4.1 Pooled separation mixes incomparable draws

Q3's separation-gap method pools every estimator run against every control run. Under P2-D6,
runs span multiple DGP-noise draws, so the estimator's worst run (a hard draw) is compared with
the control's best run (an easy draw). That is what killed the October 27 amendment: pooled gap
−0.0987, while every individual draw separated cleanly.

**Proposal:** calibrate **per draw** — within each noise draw, compare that draw's estimator tail
to that draw's control tail; require separation on *every* draw; freeze the most conservative
resulting threshold. Same margin, same method, correct pairing.

### 4.2 The Stage 10 seed family is now contaminated

Seeds 501–505 have been observed. Selecting a configuration while knowing those five values and
then "passing" on them is not a holdout.

**Proposal:** treat 501–505 as a documented **development** set; evaluate the final gate on a
fresh, never-fitted family (e.g. 601–605). This costs five fits and is non-negotiable for the
gate to mean anything.

---

## 5. Options

| | What it proves | What it cannot prove | Cost | Authority |
|---|---|---|---|---|
| **A. Variance reduction within `CausalForestDML`** (`min_samples_leaf`, `max_samples`, `n_estimators`) | Whether the swing is forest-induced and removable by regularising the partition | Nothing about bias; nothing if the swing is data-driven | 9–12 fits ≈ 1.2–4 h | Config only, but **re-opens frozen Q2 values** |
| **B. Bias reduction via nuisance/design** (richer `model_y`/`model_t`, more folds) | Whether the 0.6–0.7 bias/signal ratio is nuisance-driven | Nothing about variance | 6–9 fits ≈ 1–2 h | Config only; re-opens Q2 |
| **C. Change the CATE family** (partially-linear / series DML on a low-dimensional basis) | Whether a parametric CATE matches the DGP's true smooth, low-dimensional form and removes both defects | Nothing about robustness of that choice off-benchmark | 6 fits, est. 1–2 h (**unmeasured**) | **Requires ADR-013 amendment** |
| **D. Redesign the benchmark** (P2-D2b): widen between-segment signal | Nothing about the estimator; it makes the gate easier by construction | — | Invalidates DGP, Q2, Q3, Stage 8, Stage 10 artifacts | Requires P2-D2b reopening + ADR |
| **E. Change the metric** (top-k or tier-level ordering) | Whether a coarser claim is defensible | The per-segment rank claim | Low | Requires superseding P2-D6 |
| **F. Narrow the E1 claim** (document cross-draw instability as a known limit) | Nothing new; stops the work honestly | — | Zero | Requires superseding E1 |

**C is stronger than it looks.** The DGP's true CATE is $q_0(1-q_0)b(X)$ with $b$ linear in four
features — a smooth, low-dimensional function. A forest with `min_samples_leaf=5` over 330k rows
is a very high-variance estimator for a nearly parametric target. A series/partially-linear DML
would attack bias *and* variance at once, at materially lower fit cost. It is also the only
option that would likely make the frozen floors attainable as they stand. Cost is an assumption,
not a measurement: a forest-free estimator has never been priced here.

**D and E are goalpost moves** and are listed only so the space is complete. I do not recommend
either. If C also fails, **F** is the honest terminal state, not D.

---

## 6. Recommendation: diagnose, then a single bounded sweep

Do not pick a knob by intuition. Sequence with stop rules:

**Phase 0 — screening (≈25 min, 3 fits).** On the worst observed draw, refit with three estimator
`random_state` values at a fixed split.
- Stable ⇒ swing is **data-driven**; A is unlikely to help; go to C.
- Wide ⇒ forest Monte-Carlo variance dominates; A is the right lever.

**Phase 1 — one bounded sweep (≈1.2–4 h, ≤12 fits).** Per the Phase 0 verdict, test at most three
configurations across three noise draws. Hard guardrails: no fit may exceed the Q2-approved
30-minute budget; no configuration is added after seeing results (pre-register the three).

**Phase 2 — decide, do not iterate.** If a configuration achieves per-draw separation with a
worst-draw $m_1$ that clears the refrozen floor, proceed. Otherwise move to C — once. No second
sweep, no threshold adjustment.

**Phase 3 — re-qualify and re-gate (≈2–3 h).** Re-run Q3 on pilot draws with per-draw pairing
(§4.1), refreeze thresholds, run Gate 10 on a fresh seed family (§4.2), record evidence.

**Stop rule.** If C fails as well, close as **F** and record E1 as permanently unattainable under
this benchmark. That is a legitimate outcome; iterating thresholds is not.

---

## 7. Pre-registered success criteria

Approved *before* any fit, so the result cannot select the criterion:

1. Every pilot noise draw separates on $m_1$, $m_{1\epsilon}$, $m_{1b}$ with margin ≥ 0.15.
2. Refrozen floors are cleared on **all** seeds of a fresh, never-evaluated family.
3. $m_2 \le 0.1167$, $m_3 \ge 0.90$, $m_4 = 1.00$, $m_5 \ge 0.90$ unchanged.
4. Every fit within the 30-minute budget; full fit ≤ 80% RAM.
5. No threshold is edited after the final campaign begins.

---

## 8. What this decision will not authorise

- Lowering a frozen floor to fit an observed result (P2-D6a).
- Reusing seeds 501–505 as a holdout.
- More than one configuration sweep or one family change.
- Any change to `params.yaml` beyond the specific approved values.

---

## 9. Operator decisions resolved (October 30, 2026)

- **OD-1. Paired per-draw separation:** **APPROVED.** Within each noise draw, the estimator's
  lower tail is compared to that draw's control tail; separation must hold across all evaluated
  draws. This corrects the pooled-separation defect of Q3 without lowering any floor.
- **OD-2. Seed family retirement:** **APPROVED.** Seeds 501–505 are retired to a documented
  development set. Any final re-gating campaign must evaluate on a fresh, disjoint seed family
  (e.g., 601–605).
- **OD-3. Phase 0 screening:** **EXECUTED.** Yielded spread 0.0957 (MIXED, leaning C).
- **OD-4. ADR-013 amendment scope:** **APPROVED (narrow amendment).** ADR-013 is narrowly amended
  to evaluate alternative CATE model families (Linear/Series DML alongside regularized forest
  variants) under a pre-registered bounded sweep. At least one non-oracle candidate must be
  included. Adoption into the pipeline remains evidence-gated; floors remain strictly frozen.
- **OD-5. Fallback rule:** **CONFIRMED.** If candidate CATE families fail to clear the frozen floors,
  the terminal state is **F** (narrow E1 claim and document DGP-noise limits), not D (benchmark
  redesign) or E (metric lowering).
