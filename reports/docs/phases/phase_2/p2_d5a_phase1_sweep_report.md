# Phase 1 Bounded CATE Family Sweep Report

**Governing:** ADR-013 Amendment (October 30, 2026), P2-D5a Decision Record.
**Date:** 2026-10-30

## Summary Table

| Candidate | Draw | Duration (s) | m1(tau) | m1(eps) | m1b | m2 bias | Cleared |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `LinearDML_Linear` | 10042 | 15.5 | 0.9215 | 0.9526 | 0.9459 | 0.0516 | ❌ FAIL |
| `LinearDML_Poly` | 10042 | 27.34 | 0.9117 | 0.9436 | 0.9386 | 0.0440 | ✅ PASS |
| `CF_GBT_Reg` | 10042 | 73.39 | 0.9337 | 0.9415 | 0.8745 | 0.0505 | ✅ PASS |
| `LinearDML_Linear` | 10101 | 17.59 | 0.9013 | 0.9379 | 0.9095 | 0.0565 | ❌ FAIL |
| `LinearDML_Poly` | 10101 | 31.09 | 0.7024 | 0.7430 | 0.7593 | 0.0583 | ❌ FAIL |
| `CF_GBT_Reg` | 10101 | 76.7 | 0.7135 | 0.7469 | 0.6653 | 0.0603 | ❌ FAIL |
| `LinearDML_Linear` | 10504 | 15.23 | 0.9649 | 0.9762 | 0.9552 | 0.0398 | ❌ FAIL |
| `LinearDML_Poly` | 10504 | 41.82 | 0.8473 | 0.8694 | 0.8378 | 0.0393 | ❌ FAIL |
| `CF_GBT_Reg` | 10504 | 75.62 | 0.6409 | 0.7110 | 0.8064 | 0.0685 | ❌ FAIL |

## Per-Candidate Aggregates

| Candidate | Worst m1 | Mean m1 | Worst m2 | Mean Dur (s) | All Cleared |
|---|---:|---:|---:|---:|:---:|
| `LinearDML_Linear` | 0.9013 | 0.9292 | 0.0565 | 16.1 | ❌ NO |
| `LinearDML_Poly` | 0.7024 | 0.8205 | 0.0583 | 33.4 | ❌ NO |
| `CF_GBT_Reg` | 0.6409 | 0.7627 | 0.0685 | 75.2 | ❌ NO |
