# Specification: P2-F11 Diagnostic Showcase Slice
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F11` (`011_showcase-slice`)  
**Governing:** ADR-009 amendment (Phase 2 showcase scope, INV-5 exemption), ADR-014 (segment-level artifact contract), P2-D12 (showcase input / scenarios), INV-5 (no proposed adjustments in Tier 1 / showcase), INV-10 (no live data, curated input only)  
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`), lean  
**Status:** APPROVED — October 31, 2026 (OD-1: A, OD-2: A)  
**Date:** 2026-10-31  

---

## 1. Context and Problem

The Phase 2 showcase is a diagnostic, glass-box developer demo — not a pricing recommendation. It reads from the `approved` MLflow alias (`aegis-causal-elasticity` v3, registered Stage 12) and displays only baseline elasticity (`ε̂`), confidence intervals, and validation metrics. By construction it must carry zero proposed rate adjustments (`Δ`), enforced at the schema level and in tests (INV-5 exemption from ADR-009, Phase 2 amendment).

The existing showcase skeleton (`src/aegis/api/` + `tests/unit/test_showcase.py`) is disabled (`skip` markers) because it relied on a legacy `sqlite:///mlflow.db` payload reader (`model_service.py`, hardcoded). It must be rewired to use `load_approved_segments()` (Stage 12) — the only allowed registry entrypoint — and must render from the real `approved` version, not a hardcoded payload.

---

## 2. Objective and Expected Outcome

Deliver a working FastAPI showcase that:

1. Launches via `uv run uvicorn aegis.api.app:app` and serves from the `approved` alias (not from hardcoded payload).
2. Displays the four curated preset scenarios (S1–S4) using the actual `tier1_segments.csv` from v3.
3. Shows only `ε̂`, CI `[lower, upper]`, `baseline_retention_q0`, `tau_hat`, segment economics (`loss_cost_per_year`, `base_premium_per_year`), and gate metadata — **never a proposed rate delta**.
4. Bears the persistent disclaimer banner: *"Model validation demo on semi-synthetic data — not a pricing recommendation."*
5. Serves Chart.js locally (no external CDN in production/offline mode).
6. Passes all showcase tests without skip markers; a dummy `proposed_rate_change` injection fails loudly.
7. Satisfies Roadmap Exit Criterion **E2** (showcase renders correctly with INV-5 exemption honored).

---

## 3. Scope

### In Scope
- `src/aegis/api/app.py` / `model_service.py`: rewired from legacy payload reader to `load_approved_segments()`.
- `src/aegis/api/scenarios.py`: existing `Scenario` dataclass; four presets (S1–S4) locked.
- `tests/unit/test_showcase.py`: remove `skip`; add `test_invoke_approved_alias`, `test_no_delta_field`, `test_banner_present`, `test_scenario_s4_from_sensitivity`.
- Jinja2 templates and static assets (local Chart.js).
- Structural INV-5 assertion in the response model (Pydantic `extra="forbid"`; `FORBIDDEN_FIELD_FRAGMENTS` already defined in `schemas/segments.py`).

### Out of Scope (explicit exclusion — do not expand here)
- Any proposed rate adjustment (`Δ`) computation, recommendation, or display (IN-5 / INV-4).
- Free-form user input to the showcase (INV-10; only the four curated presets).
- Live carrier / production data (INV-10; showcases only from `approved` semi-synthetic artifact).
- Tier 2 agent nodes, LLM calls, or Gateway use (Phase 2 only).
- New model training or inference; the showcase is a pure reader of the already-registered v3.

---

## 4. Functional Requirements (EARS)

**FR-1 (showcase launches).** When `uv run uvicorn aegis.api.app:app` is executed, the server binds to the configured port, the root route responds, and the `approved` alias resolves to v3 (or fails clearly with `ModelNotFoundError` when `approved` is unset).

**FR-2 (scenarios from `approved`).** For each preset scenario (S1–S4), the API response must include the corresponding segment IDs from the `approved` bundle (S1: large / well-recovered; S2: small / wide intervals; S3: cap-sensitive from Stage 5; S4: confounded from Stage 11 `gt_0.05_gy_0.50`). All metrics must correspond to v3's `tier1_segments.csv`, not to hardcoded values.

**FR-3 (structural INV-5).** The response schema must contain strictly zero fields naming a proposed adjustment (`delta`, `adjust`, `proposed`, `new_premium`). Any injection of such a field must cause a schema test failure.

**FR-4 (disclaimer banner).** Every rendered HTML view must contain the banner text. Removing it must cause a test failure.

**FR-5 (local Chart.js).** The Chart.js library must be served from local static assets; no external CDN call appears in production/offline mode.

**FR-6 (falsification — dummy delta).** When a test injects `proposed_rate_change` into the response payload, the schema assertion must fail loudly (`ValidationError`).

---

## 5. Data Contract

### 5.1 Scenario Input (from `approved` bundle)

The showcase reads from `load_approved_segments()` (Stage 12) — not from a hardcoded payload or from the legacy `sqlite:///mlflow.db` reader (`model_service.py` before rewiring).

Segment fields displayed (from `Tier1SegmentOutput`):
- `segment_id`
- `tau_hat` (elasticity point estimate), `epsilon_hat`
- `epsilon_ci_lower` / `epsilon_ci_upper`
- `baseline_retention_q0`
- `policy_count`, `loss_cost_per_year`, `base_premium_per_year`
- `model_version` (v3), `data_version`
- Gate meta: `validation_status` (`GATE_PASSED_OPTION_F`), `segment_count` (35)

Fields explicitly excluded (INV-5 / ADR-009 exemption): `delta`, `adjusted_premium`, `proposed_rate`, `recommended_price`.

### 5.2 Scenario Reference (locked)

| Scenario | Key | Source Segment | Rationale |
|---|---|---|---|
| S1 | `large-segment` | Well-recovered segment with tight CI (high `n`, low `m2`) | Demonstrates healthy elasticity recovery |
| S2 | `small-segment` | Sparse segment with wide CI (`low policy_count`, wide epsilon interval) | Honest uncertainty, no overconfidence |
| S3 | `cap-sensitive` | Segment with large uncapped claim (>€100k influence from Stage 5 cap sweep) | Shows cap-policy effect on loss cost |
| S4 | `confounded` | `gt_0.05_gy_0.50` (Stage 11 sensitivity, Scenario S4) | Demonstrates estimator limits under hidden confounding |

---

## 6. Edge Cases & Error Handling

| Edge Case | Required System Behavior |
|---|---|
| `approved` alias unset (e.g., v3 unverified) | Raises `ModelNotFoundError`; render page shows governance-state banner, not a hard crash |
| Segment missing from `approved` bundle | Fallback to first available segment with a visible warning; never fabricate data |
| Chart.js asset missing locally | Render page with static table only; no external fetch (offline-safe) |
| Banner HTML stripped by injection | `test_banner_present` fails loudly |

---

## 7. Open Decisions (for approval before `plan.md`)

### OD-1 — Source of scenario segment selection
- **Option A (Recommend):** Hardcoded `segment_id` list in `scenarios.py` referencing the four curated cases (S1, S2, S3, S4). Simple, auditable, reproducible.
- **Option B:** Dynamic selection by metric thresholds (`m2`, `policy_count`, `epsilon` width) from the `approved` bundle.
- **Recommendation:** Option A. Phase 2 showcase is a fixed diagnostic set; dynamic selection introduces variability on each run.

### OD-2 — Template engine
- **Option A (Recommend):** Existing Jinja2 + `src/aegis/api/app.py` (FastAPI). Minimal change; already has template directory.
- **Option B:** Static HTML served from `artifacts/showcase/`.
- **Recommendation:** Option A. Keeps the interactive demo (scenario picker, metric display) without building a new server.

---

## 8. Falsification Protocol & Hard Gate

### Falsification Tests (all must pass for Gate 13 close)

1. `test_invoke_approved_alias`: server responds against v3 (not hardcoded payload) — assert `model_version == "3"` in response.
2. `test_no_delta_field`: inject `proposed_rate_change: 0.05` into response payload — assert `ValidationError`.
3. `test_banner_present`: assert banner substring appears in rendered HTML.
4. `test_scenario_s4_from_sensitivity`: S4 segment corresponds to `gt_0.05_gy_0.50` parameters from Stage 11.
5. `test_approved_missing_refuses`: temporarily rename alias — server returns clear error page, not a silent fallback.

### Gate 13 — Hard Close

- `uvicorn aegis.api.app:app` launches.
- All four preset scenarios render from the `approved` v3 artifact.
- Response schema has zero proposed-adjustment fields.
- Disclaimer banner present on all views.
- All showcase tests pass without skips.
- E2 satisfied (showcase renders correctly with INV-5 exemption).
