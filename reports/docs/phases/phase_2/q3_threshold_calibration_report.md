# Qualification Q3: Threshold Calibration Report

**Executed:** October 27, 2026  
**Pilot Seeds:** `[42, 101, 202]`  
**Method:** Separation gap with minimum margin (ADR-013, P2-D6a)  
**Outcome:** ✅ PASSED — Requires separation, all negative-control failures, and positive-control pass.

## 1. Estimator Performance on Pilot Seeds

| Seed | m1(tau) | m1(eps) | m1b | m2 | m3(tau) | m3(eps) | m4 | m5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8858 | 0.8983 | 0.7853 | 0.0976 | 1.000 | 1.000 | 1.000 | 1.000 |
| 101 | 0.9479 | 0.9447 | 0.8776 | 0.0823 | 1.000 | 1.000 | 1.000 | 0.971 |
| 202 | 0.9537 | 0.9551 | 0.9342 | 0.0909 | 1.000 | 1.000 | 1.000 | 1.000 |

## 2. Controls Performance

| Control | Name | As Expected | m1 (tau) | m2 (bias) | m5 (sign) | Diagnostic |
|---|---|:---:|---:|---:|---:|---|
| `c1_unadjusted` | Unadjusted naive within-segment OLS on confounded data | ✅ YES | 0.6928 | 0.1667 | 0.9429 | c1 m1_tau=0.6928, m2_bias=0.1667, m5_sign=0.9429 |
| `c2_shuffled` | Randomly permuted segment effect predictions | ✅ YES | -0.0402 | 0.2114 | 1.0000 | c2 m1_tau=-0.0402 (near 0 as expected) |
| `c3_zero_effect` | Zero-effect DGP (b(X)=0) | ✅ YES | -0.3727 | 0.3900 | 0.6857 | c3 false_discovery_rate=0.000 (<= 0.15), m1_tau=-0.3727 |
| `positive_control` | Randomized treatment assignment variant (unconfounded) | ✅ YES | 0.9152 | 0.0727 | 1.0000 | positive_control m1_tau=0.9152 (>= 0.9: True) |

## 3. Separation Gaps and Threshold Calibration

| Metric | Estimator Tail | Control Tail | Gap | Margin | Separated | Threshold |
|---|---:|---:|---:|---:|:---:|---:|
| `m1_tau_spearman` | 0.8858 | 0.6928 | 0.1930 | 0.1500 | ✅ YES | **0.8428** |
| `m1_epsilon_spearman` | 0.8983 | 0.7331 | 0.1652 | 0.1500 | ✅ YES | **0.8831** |
| `m1b_policy_spearman` | 0.7853 | 0.6284 | 0.1568 | 0.1500 | ✅ YES | **0.7784** |
| `m2_mean_abs_bias` | 0.0976 | 0.1667 | 0.0692 | 0.0500 | ✅ YES | **0.1167** |

### Fixed absolute gate

`m5_sign_agreement` is fixed at **0.90** and excluded
from empirical separation calibration. Under homogeneous-negative true effects,
the shuffled and unadjusted controls preserve direction and cannot separate on sign.

## 4. Frozen Thresholds Block (`params.yaml`)

```yaml
tier1_ml:
  validation_thresholds:
    m1_tau_spearman_floor: 0.8428
    m1_tau_spearman_target: 0.8954
    m1_epsilon_spearman_floor: 0.8831
    m1_epsilon_spearman_target: 0.9139
    m1b_policy_spearman_floor: 0.7784
    m1b_policy_spearman_target: 0.8280
    m2_mean_abs_bias_ceiling: 0.1167
    m2_mean_abs_bias_target: 0.1038
    m3_tau_coverage_floor: 0.90
    m3_epsilon_coverage_floor: 0.90
    m4_self_consistency_floor: 1.00
    m5_sign_agreement_floor: 0.9000
```

## 5. Gate Conclusion

Qualification Q3 passed with clear statistical separation across all separation-calibrated metrics. All negative controls demonstrably failed the harness, and the positive control passed cleanly.

Thresholds are frozen in `params.yaml` only when the outcome is PASSED, ahead of
the definitive Stage 10 final-seed evaluation.
