# Plan: P2-F11 Diagnostic Showcase Slice
## Phase 2 — Stage 13

**Spec:** APPROVED October 31, 2026 (OD-1: Option A, OD-2: Option A)  
**Status:** APPROVED — October 31, 2026  
**Governing:** ADR-009 (showcase scope, INV-5 exemption), ADR-014 (segment contract, `load_approved_segments`), P2-D12 (scenarios)  

---

## 1. Strategy (Lean)

Stage 13 is a pure reader of the `approved` v3 artifact (`aegis-causal-elasticity`). No new model, no new pipeline. Five tight tasks:

1. Rewire `model_service.py` (legacy payload reader) → `load_approved_segments()`.
2. Lock 4 preset scenarios (S1–S4) with hardcoded segment IDs from v3 bundle.
3. Structural INV-5 assertion: Pydantic `extra="forbid"` + `FORBIDDEN_FIELD_FRAGMENTS` guard.
4. Banner + local Chart.js; falsification (dummy delta, banner removal).
5. Re-enable `test_showcase.py` (remove `skip`) and verify against real backend.

---

## 2. Component Updates

### 2.1 `src/aegis/api/model_service.py` (rewired, 88 lines, must stay < 1,000)
Replaces `load_registered_model_payload()`'s hardcoded `sqlite:///mlflow.db` + `"newest usable version"` fallback with `load_approved_segments()` from `aegis.pipelines.registry`. Keeps `_flatten_model_payload()` for the showcase response shape.

### 2.2 `src/aegis/api/app.py`
Adds `/showcase` and `/scenario/{key}` routes; serves Jinja2 templates; injects banner; passes `Tier1SegmentTable` + validation summary to template.

### 2.3 `src/aegis/api/scenarios.py` (existing, unchanged structure)
Locks 4 `Scenario` instances with segment IDs drawn from v3.

---

## 3. Falsification Protocol

| # | Action | Expected Failure |
|---|---|---|
| 1 | Mock inject `proposed_rate_change` into response model | `Pydantic ValidationError` (extra="forbid") |
| 2 | Remove banner HTML from template | `test_banner_present` fails |
| 3 | Unregister `approved` alias temporarily | `ModelNotFoundError` returned cleanly (not crash) |
| 4 | Verify no `delta` / `adjust` / `new_premium` appears in any response payload | Assertion error |

---

## 4. Execution Sequence (Lean: T1–T5)

1. **T1** — Rewire `model_service.py`; add `load_approved_segments()` call.
2. **T2** — Lock scenario presets (S1–S4); wire `/showcase` routes in `app.py`.
3. **T3** — Add structural INV-5 assertion + falsification tests; add banner HTML + local Chart.js static asset.
4. **T4** — Remove `skip` markers from `test_showcase.py`; add 4 new tests.
5. **T5** — Execute against real `approved` v3 backend (`load_approved_segments()`); verify `test_invoke_approved_alias` returns `model_version="3"`; commit + report.

---

## 5. Gate 13 — Hard Close (lean checklist)

- `uvicorn aegis.api.app:app` launches against real `approved` v3.
- All 4 scenarios render; response schema has zero proposed-adjustment fields.
- Banner present; falsification tests pass.
- E2 satisfied; no skip markers remain in `test_showcase.py`.
