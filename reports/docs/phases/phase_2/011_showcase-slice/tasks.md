# Tasks: Stage 13 — Diagnostic Showcase Slice (P2-F11)

**Feature:** `P2-F11` (Full SDD, lean depth)  
**Spec:** APPROVED 2026-10-31 (OD-1 = preset IDs; OD-2 = FastAPI + Jinja2)  
**Plan:** APPROVED 2026-10-31  
**Status:** COMPLETE — Gate 13 closed 2026-10-31  
**Governing:** ADR-009 (showcase scope, INV-5 exemption), ADR-014 (`load_approved_segments`), P2-D12

---

## Decisions applied

- **OD-1 Option A:** four fixed presets in `scenarios.py`; no dynamic selection.
- **OD-2 Option A:** existing FastAPI + Jinja2 layer.
- **S4 amendment (operator-approved 2026-10-31):** S4 is the global Stage 11 cell
  `gt_0.05_gy_0.50`, which has no segment row. `ApprovedSegmentPackage` now exposes the
  bundled `sensitivity_summary`, and S4 renders that cell labelled "global stress result,
  not a policy segment". FR-2 therefore maps S1–S3 to segment IDs and S4 to the cell.

---

## Tasks

### T1 — Service layer rewired to the approved alias
- [x] `src/aegis/api/model_service.py` now calls `load_approved_segments()` only; the legacy
      hardcoded `sqlite:///mlflow.db` reader and "newest usable version" fallback are gone.
- [x] `clear_model_cache()` clears the approved-bundle cache.

### T2 — Presets and routes
- [x] `scenarios.py`: S1 `large-segment` (`D:45-64_C:0-4_Dens:<=500`), S2 `small-segment`
      (`D:18-24_C:0-4_Dens:>2000`), S3 `cap-sensitive` (`D:18-24_C:10+_Dens:<=500`,
      77.2% of uncapped loss above the 100k cap), S4 `confounded` (`gt_0.05_gy_0.50`).
- [x] Routes `/` (`?preset=`) and `/scenario/{key}` render from the approved bundle.
- [x] Missing preset segment falls back to the first bundle row with a visible notice; it never
      fabricates metrics.

### T3 — INV-5 guard, banner, local chart
- [x] `src/aegis/schemas/showcase.py`: `ShowcaseView` with `extra="forbid"` plus the shared
      `guard_forbidden_fields` fragment screen; exactly one of segment/stress must be set.
- [x] Persistent banner on every view, including governance-error pages.
- [x] Chart.js 4.4.4 vendored at `src/aegis/api/static/chart.umd.min.js` (205,754 B, SHA-256
      `fcb24ab5…7ab8da4`); no CDN.

### T4 — Unit tests (no skips)
- [x] `tests/unit/test_showcase.py` rewritten: 11 tests, zero `skip` markers.

### T5 — Verification and close
- [x] Real backend (`artifacts/mlflow.db`, approved v3) renders all five routes; S4 cell and
      35-segment count verified; banner and local asset verified.
- [x] `uv run uvicorn aegis.api.app:app --port 8123` served `/health` and `/scenario/confounded`
      (200) before shutdown.
- [x] Status documents updated.

---

## Verification (actual commands and outcomes)

| Command | Result |
|---|---|
| `uv run pytest -q` | **375 passed, 0 skipped** (5m 54s) |
| `uv run pytest tests/unit/test_showcase.py tests/unit/test_registry.py -q` | 31 passed |
| `uv run ruff check .` | All checks passed |
| `uv run pyright` | 0 errors, 0 warnings |
| `uv run python scripts/check_module_size.py` | 51/51 modules within 1,000 lines |

**INV-8 deltas:** new `schemas/showcase.py` 73 lines, `api/app.py` 74, `api/scenarios.py` 114,
`api/model_service.py` 30; `pipelines/registry.py` +8; `schemas/registry.py` +1.

---

## Gate 13 — Hard Close Checklist

- [x] `uvicorn aegis.api.app:app` launches against the real `approved` v3 artifact.
- [x] All four presets render; S1–S3 from `tier1_segments.csv`, S4 from the bundled
      `sensitivity_summary.json`.
- [x] Response schema rejects any price-change field (`ValidationError`); rendered pages name
      none of the forbidden fragments.
- [x] Banner present on segment, stress, invalid-preset, and governance-error views.
- [x] Zero skip markers in `test_showcase.py`.
- [x] Roadmap Exit Criterion E2 satisfied.

---

## Outcome

### Stage 13 (P2-F11) — COMPLETE (Gate 13 hard close, 2026-10-31)

The showcase reads only the `approved` alias through `load_approved_segments()`. Segment
presets display ε̂, its interval, retention, policy count, loss cost, and base premium from
the registered bundle; the confounding preset displays the frozen Stage 11 cell. The
response contract cannot represent a proposed price change, and Chart.js is served locally.

No commit or push performed; awaiting explicit authorization.
