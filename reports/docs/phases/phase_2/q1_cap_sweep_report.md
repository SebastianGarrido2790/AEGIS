# Q1 — Loss Cap Sweep & Final Calibration

Qualification test Q1 (P2-D1a, ADR-012). All metrics except holdout deviance are computed on the training split (330,535 policies / 12,249 claims).

## Per-candidate metrics

| Cap (€) | Loss removed (€) | Removed % | Claims truncated | Truncated % | Excess load | p̂ | Holdout deviance | Max excess ratio (segment) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,109 | 8,801,359 | 31.34% | 126 | 1.03% | 1.4565 | 1.81 | 16.3187 | 36.7× (D:18-24_C:10+_Dens:<=500) |
| 35,432 | 7,045,693 | 25.09% | 67 | 0.55% | 1.3349 | 1.81 | 16.6300 | 44.3× (D:18-24_C:10+_Dens:<=500) |
| 100,000 | 4,656,129 | 16.58% | 20 | 0.16% | 1.1988 | 1.84 | 16.9016 | 62.9× (D:18-24_C:10+_Dens:<=500) |
| 131,834 | 4,087,724 | 14.56% | 14 | 0.11% | 1.1704 | 1.84 | 16.9702 | 69.6× (D:18-24_C:10+_Dens:<=500) |

## Adjacent-cap transitions (selection criteria)

| Transition | Spearman (segment loss-cost) | Holdout deviance change |
|---|---:|---:|
| 16,109 → 35,432 | 0.9972 | 1.908% |
| 35,432 → 100,000 | 0.9927 | 1.633% |
| 100,000 → 131,834 | 0.9992 | 0.406% |

## Selection

- Selected cap: **€100,000** (lowest truncation level whose transition satisfies Spearman ≥ 0.98 and deviance change ≤ 1%).
- Frozen uniform excess load: **1.1988**.

## Excess concentration (known limitation)

The excess losses are materially concentrated: at the selected cap, segment `D:18-24_C:10+_Dens:<=500` holds 72.0% of excess loss against 1.14% of exposure (ratio 62.9×). This single segment contains the €2.04M claim. A uniform excess load therefore understates `Losses₀` for that segment and overstates it elsewhere; this is recorded as a known benchmark limitation carried to Phase 6 (operator decision 2026-10-24).

