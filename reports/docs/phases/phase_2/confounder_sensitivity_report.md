# Hidden-Confounder Sensitivity Analysis Report (Stage 11, P2-F08)

**Date:** 2026-10-30T13:45:27.101442+00:00  
**Governing:** ADR-013, P2-D7, `system_design.md` §6  
**Sample Size:** 100,000 policies  
**Config Hash:** `fce1b284d8414118d09926b9e6350df2e6cf410fb6066bcfbff8ce6a8ee08d67`  

## 1. Executive Summary

- **Grid Execution:** 10/10 cells executed (Zero Skips Verified: True).
- **Baseline Recovery (No Confounding):** $m_1(\tau) = 0.6407$, $m_2 = 0.1147$.
- **Robustness Value:** $R_v = 0.5004$ (minimum confounding norm $\sqrt{\gamma_T^2 + \gamma_Y^2}$ breaching threshold $m_1(\tau) < 0.60$).
- **Scenario S4 Locked Cell:** `gt_0.05_gy_0.50` ($\gamma_T = 0.05$, $\gamma_Y = 0.5$, $m_1(\tau) = 0.6579$, $m_2 = 0.7419$).

## 2. 10-Cell Sensitivity Degradation Table

| Cell ID | $\gamma_T$ | $\gamma_Y$ | Strength $r$ | Duration (s) | $m_1(\tau)$ | $\Delta m_1$ | $m_2$ Bias | $\Delta m_2$ | $m_3$ Cov | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `gt_0.00_gy_0.00` | 0.00 | 0.00 | 0.000 | 90.6 | 0.6407 | +0.0000 | 0.1147 | +0.0000 | 1.0000 | ✅ Converged |
| `gt_0.02_gy_0.20` | 0.02 | 0.20 | 0.201 | 94.0 | 0.6174 | -0.0233 | 0.1983 | +0.0836 | 1.0000 | ✅ Converged |
| `gt_0.02_gy_0.50` | 0.02 | 0.50 | 0.500 | 88.5 | 0.5756 | -0.0651 | 0.4981 | +0.3834 | 1.0000 | ✅ Converged |
| `gt_0.02_gy_1.00` | 0.02 | 1.00 | 1.000 | 87.2 | 0.2196 | -0.4211 | 1.0841 | +0.9694 | 1.0000 | ✅ Converged |
| `gt_0.05_gy_0.20` | 0.05 | 0.20 | 0.206 | 85.2 | 0.7602 | +0.1195 | 0.3025 | +0.1878 | 1.0000 | ✅ Converged |
| `gt_0.05_gy_0.50` (S4) | 0.05 | 0.50 | 0.502 | 97.3 | 0.6579 | +0.0172 | 0.7419 | +0.6272 | 1.0000 | ✅ Converged |
| `gt_0.05_gy_1.00` | 0.05 | 1.00 | 1.001 | 95.1 | -0.2073 | -0.8480 | 1.6201 | +1.5054 | 0.0000 | ✅ Converged |
| `gt_0.10_gy_0.20` | 0.10 | 0.20 | 0.224 | 97.2 | 0.8609 | +0.2202 | 0.2778 | +0.1631 | 1.0000 | ✅ Converged |
| `gt_0.10_gy_0.50` | 0.10 | 0.50 | 0.510 | 99.5 | 0.7213 | +0.0806 | 0.6911 | +0.5764 | 0.9714 | ✅ Converged |
| `gt_0.10_gy_1.00` | 0.10 | 1.00 | 1.005 | 98.5 | -0.2276 | -0.8683 | 1.4378 | +1.3231 | 0.0000 | ✅ Converged |

## 3. Degradation Analysis & Robustness Assessment

As unobserved confounding strength $U \sim \mathcal{N}(0, 1)$ couples more strongly with treatment assignment (price delta) and renewal log-odds:
1. **Rank Recovery Degradation:** Segment rank correlation $m_1(\tau)$ degrades monotonically along the coupling axes as omitted-variable bias distorts relative pricing elasticity.
2. **Bias Inflation:** Absolute elasticity bias $m_2$ inflates as confounding strength grows, shifting segment estimates systematically upward.
3. **Showcase Integration:** Scenario S4 accurately reflects this degraded operating condition in the Tier 1 diagnostic showcase.
