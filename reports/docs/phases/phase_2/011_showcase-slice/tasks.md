# Tasks: Stage 13 — Diagnostic Showcase Slice (P2-F11)

**Feature:** `P2-F11` (Full SDD, lean depth)  
**Spec:** APPROVED 2026-10-31 (OD-1 = Option A preset IDs; OD-2 = Option A FastAPI)  
**Plan:** APPROVED 2026-10-31  
**Status:** IN PROGRESS (plan approved; implementation not yet authorized)  
**Governing:** ADR-009 (showcase scope, INV-5 exemption), ADR-014 (`load_approved_segments`), P2-D12  
**Dependencies:** Stage 12 `approved` v3 artifact  

---

## Overview

Read-only glass-box showcase from the `approved` v3 bundle. Four curated presets (S1–S4); zero proposed adjustments; banner present; offline-safe charting; falsifiable via dummy delta injection.

---

## Task Decomposition

### T1 — Service layer rewired
- Replace `load_registered_model_payload()` in `model_service.py` with `load_approved_segments(config)` → returns `Tier1SegmentTable` + `lineage`.
- Keep `_flatten_model_payload()` for the showcase response shape.
- Assert `approved` alias resolves; else raise `ModelNotFoundError` (clean, not crash).

### T2 — Scenario presets + routes
- Lock S1 (`large-segment`), S2 (`small-segment`), S3 (`cap-sensitive`), S4 (`confounded` = `gt_0.05_gy_0.50`) in `scenarios.py`.
- Add `/showcase` (default S1) and `/scenario/{key}` routes in `app.py`.
- Pass segment table + validation summary to Jinja2 template.

### T3 — INV-5 structural guard + banner + chart asset
- Response model (`src/aegis/schemas/showcase.py` or inline): `ConfigDict(extra="forbid")` + `FORBIDDEN_FIELD_FRAGMENTS` guard.
- Banner HTML fragment in template: persistent on every view.
- Local Chart.js static file (no CDN fetch).

### T4 — Unit tests (replace skip markers)
- `test_invoke_approved_alias`: asserts `model_version == "3"` in response.
- `test_scenario_s4_from_sensitivity`: asserts `segment_id` references `gt_0.05_gy_0.50`.
- `test_no_delta_field`: inject `proposed_rate_change`; asserts `ValidationError`.
- `test_banner_present`: asserts banner substring in HTML response.
- `test_approved_missing_refuses`: temporarily unset `approved`; asserts clean `ModelNotFoundError`.

### T5 — Final verification + close
- [ ] `uv run uvicorn aegis.api.app:app` launches.
- [ ] All 4 scenarios render from real `approved` v3.
- [ ] No `delta`/`adjust`/`new_premium` appears in any payload.
- [ ] Banner present; Chart.js served locally.
- [ ] All `test_showcase.py` tests pass (zero skips).
- [ ] Update `execution_plan.md` (§ Stage 13), `stage_evidence.md` (E2), `implementation_plan.md` (P2-F11 status).

---

## Gate 13 — Hard Close (lean checklist)

- [ ] FastAPI serves from `approved` v3.
- [ ] 4 preset scenarios render; S4 references `gt_0.05_gy_0.50`.
- [ ] Response schema rejects any `Δ` field injection.
- [ ] Disclaimer banner present; falsification tests pass.
- [ ] Zero skip markers in `test_showcase.py`.
- [ ] E2 satisfied; no proposed rate adjustments emitted.
