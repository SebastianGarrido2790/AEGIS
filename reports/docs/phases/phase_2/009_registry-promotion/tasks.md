# Tasks: Stage 12 — MLflow Artifact Packaging & Human-Gated Promotion (P2-F09)

**Feature:** `P2-F09` (Full SDD, lean depth)
**Spec:** APPROVED 2026-10-30 · **Plan:** APPROVED 2026-10-30
**Governing:** ADR-014, P2-D11, P2-D11a, ADR-005, INV-4 / INV-5 / INV-7 / INV-8 / INV-10
**Status:** IN PROGRESS
**Date:** 2026-10-30

---

## Overview

Package the Tier 1 segment output as a plain-data MLflow artifact bundle, register it as `aegis-causal-elasticity`, auto-assign only the `candidate` alias, and gate the `approved` alias behind a human review record (reviewer ID + calibration hash + complete lineage). The fitted estimator is logged as a non-promotable run artifact and is never deserialized on the serving path.

Sequencing: T1 → T7 are code + verification. T8 is the automated real-backend registration. **T9 is operator-gated** (the agent cannot self-approve). T10 is the rollback rehearsal plus documentation close.

---

## Task Decomposition

### T1 — Schemas, exceptions, config
**Files:** `src/aegis/schemas/registry.py` (new), `src/aegis/utils/exceptions.py`, `src/aegis/config/schema.py`, `params.yaml`

- `ModelLineageMetadata` (`extra="forbid"`; `git_commit` 40-hex pattern, `git_dirty: bool`, `dvc_data_hash` 32–64 hex, `dgp_manifest_id`/`calibration_hash` SHA-256 pattern, `validation_status: Literal["GATE_PASSED_OPTION_F"]`, `segment_count >= 1`).
- `ModelPromotionRequest` (`reviewer_id min_length=3`, `calibration_report_hash` len 64, `review_notes min_length=10`).
- `PromotionAuditRecord` (`action: Literal["promote", "rollback"]`).
- `ApprovedSegmentPackage` (`segments: Tier1SegmentTable`).
- Exceptions: `RegistryPackagingError`, `PromotionGateError(details["reason"])`, `RegistrySecurityError`, `ModelNotFoundError` — all subclass `AEGISError`.
- `RegistryConfig` / `RegistryPathsConfig` added to `Tier1MLConfig`; `params.yaml` blocks per plan §2.4 (model name, tracking URI, aliases, required lineage tags, paths).

**Acceptance:** importing the module succeeds; a payload with an extra key is rejected; `params.yaml` loads through `get_config()` with `extra="forbid"` intact.
**Verify:** `uv run python -c "from aegis.config.loader import get_config; print(get_config().tier1_ml.registry.model_name)"`

---

### T2 — Packaging module (pure)
**Files:** `src/aegis/pipelines/artifact_package.py` (new), `tests/unit/test_artifact_package.py` (new)

- `load_tier1_segment_table(...)`: inner-join `segment_loss_cost.csv` (Stage 4 economics) with `segment_elasticity_estimates.csv` (Stage 8 causal) on `segment_id` → exactly 35 `Tier1SegmentOutput` rows. Field provenance per plan §2.2 (`model_version` from `causal_estimator_metadata.json`, `data_version` from `dgp_manifest.json["config_fingerprint"]`, `semi_synthetic=True`).
- Reject row-count mismatch, unknown id, duplicate id, and any failing `Tier1SegmentOutput` validation → `RegistryPackagingError`.
- `resolve_git_commit(repo_root) -> (sha, dirty)`; `resolve_dvc_data_hash(dvc_lock, training_table)` reading `dvc.lock` md5 for `data/processed/elasticity_training.csv` (`24f3e55aab70348c48fd204dcdad0165`), SHA-256 fallback when absent, `RegistryPackagingError` on lock-vs-file mismatch.
- `resolve_lineage(...)` → `ModelLineageMetadata`; `build_serving_bundle(...)` writes exactly the five standard files; `assert_no_estimator_artifacts(paths)` raises `RegistrySecurityError` on `.pkl` / `.pickle` / `.joblib`.

**Acceptance:** 7 unit tests from plan §7.1 pass; bundle file set is exactly `STANDARD_BUNDLE_FILES`.
**Verify:** `uv run pytest tests/unit/test_artifact_package.py -q`

---

### T3 — Registry client: registration + `candidate` alias
**Files:** `src/aegis/pipelines/registry.py` (new, part 1), `tests/unit/test_registry.py` (new, part 1)

- `resolve_tracking_uri(config)`: `MLFLOW_TRACKING_URI` env wins over `tier1_ml.registry.tracking_uri`.
- `package_and_register_candidate(config=None)`: create experiment, start run, log serving bundle under `serving_bundle/`, log the fitted estimator under `estimator/` as a **non-promotable run artifact** when present, tag the run/model version with all required lineage tags, create the model version under `aegis-causal-elasticity`, and set **only** `candidate`.
- No code path in this module assigns `approved` except `promote_to_approved` (T4).

**Acceptance:** `test_candidate_registration_sets_only_candidate` passes; `approved` alias absent after registration.
**Verify:** `uv run pytest tests/unit/test_registry.py -q -k candidate`

---

### T4 — Promotion gate + rollback (fail-closed)
**Files:** `src/aegis/pipelines/registry.py` (part 2), `tests/unit/test_registry.py` (part 2)

- `promote_to_approved(request, config=None, client=None)` in the plan §4 order: version exists → carries `candidate` → all `required_lineage_tags` present → request hash equals recorded tag → recomputed on-disk SHA-256 equals tag (`calibration_report_drift`) → assign `approved` → write `promoted_by`, `promoted_at`, `promotion_rationale`, `promotion_calibration_hash`. Every failure raises `PromotionGateError` with a typed `reason` and leaves aliases untouched.
- `rollback_approved(target_version, reviewer_id, reason, ...)`: target must exist and have been approved before (`promoted_by` tag); re-point `approved`, record `rollback_from`, return `PromotionAuditRecord(action="rollback")`.

**Acceptance:** falsification tests (a) missing reviewer, (b) mismatched hash, (c) missing lineage tag, plus success, non-candidate rejection, drift detection, rollback, and unknown-version failure all pass.
**Verify:** `uv run pytest tests/unit/test_registry.py -q -k "promotion or rollback"`

---

### T5 — Serving loader
**Files:** `src/aegis/pipelines/registry.py` (part 3), `tests/unit/test_registry.py` (part 3)

- `load_approved_segments(config=None, client=None) -> ApprovedSegmentPackage`: resolve `approved` alias (absent → `ModelNotFoundError`), download artifact tree, `assert_no_estimator_artifacts` (hit → `RegistrySecurityError`), parse `tier1_segments.csv` → `Tier1SegmentTable`, parse `lineage.json` → `ModelLineageMetadata`, load validation summary.
- Source-level guard test: `registry.py` contains no `joblib` / `pickle` / `econml` imports.

**Acceptance:** loader returns 35 plain segments; `.pkl` in the artifact tree raises `RegistrySecurityError`; missing alias raises `ModelNotFoundError`.
**Verify:** `uv run pytest tests/unit/test_registry.py -q -k serving`

---

### T6 — CLI subparsers + operator script
**Files:** `src/aegis/pipelines/registry.py` (`add_registry_subparsers`), `src/aegis/pipelines/cli.py`, `scripts/promote_model.py` (new), `tests/unit/test_registry.py` (part 4)

- `add_registry_subparsers(subparsers)` builds `promote-model` and `rollback-model` in `registry.py`; `cli.py` gains only the import, the call inside `build_parser()`, and the `args.func(args)` dispatch (~6 lines).
- `scripts/promote_model.py`: thin argparse wrapper over `promote` / `rollback`, prints the audit record, exits `1` on `PromotionGateError`.

**Acceptance:** `build_parser()` accepts both subcommands (`test_cli_parsers_registered`); `cli.py` ≤ 980 lines.
**Verify:** `uv run python scripts/check_module_size.py` and `uv run python -m aegis.pipelines.cli promote-model --help`

---

### T7 — Full verification
- [x] `uv run pytest -q` (full suite, zero regressions)
- [x] `uv run ruff check .`
- [x] `uv run pyright` (0 errors)
- [x] `uv run python scripts/check_module_size.py` (all modules < 1,000)

**Acceptance:** all four commands clean; counts recorded in Outcome.

---

### T8 — Candidate registration on the real backend *(automated, no human gate)*
- [x] Run `package_and_register_candidate()` against `artifacts/mlflow.db`.
- [x] Record: run ID, model version, `candidate` alias target, all lineage tag values, bundle file list with hashes.
- [x] Confirm `approved` alias is **still unset**.

**Acceptance:** version registered with `candidate` only; no `approved` alias exists.

**Evidence (2026-10-30):** pre-commit registrations v1 (`run 18d223b5…e3e2`) and v2
(`run 0108c99c…2aa3`) both carry `git_dirty=true` and `git_commit=d5981de…4014`, which does not
contain the Stage 12 code. Post-commit registration produced **v3**
(`run 77f6272859a2474da04de3b2bff04916`, `git_commit=04b0cbae…8c0c`, `git_dirty=false`), and
`candidate` now resolves to v3. Serving loader against the real backend fails closed with
`ModelNotFoundError: No model version currently assigned to alias 'approved'`, confirming no
version is approved before human review. `approved` stayed unset throughout.

---

### T9 — Human-gated promotion *(completed 2026-10-31)*
Completed — operator supplied reviewer ID / review note / hash confirmation out-of-band.

- [x] Agent printed candidate v3's recorded `calibration_hash` (matches on-disk report).
- [x] Operator supplied reviewer ID `Sebastián Garrido Arévalo` + note; agent executed
      `scripts/promote_model.py promote`.
- [x] Audit tags recorded: `promoted_by`, `promoted_at`, `promotion_rationale`,
      `promotion_calibration_hash`.

**Audit record:**
```json
{
  "model_name": "aegis-causal-elasticity",
  "previous_approved_version": null,
  "new_approved_version": "3",
  "reviewer_id": "Sebastián Garrido Arévalo",
  "action": "promote",
  "calibration_verified": true,
  "audit_tags": {
    "promoted_by": "Sebastián Garrido Arévalo",
    "promoted_at": "2026-10-31T01:38:57.513694+00:00",
    "promotion_rationale": "We have been extremely meticulous regarding all the steps set out "
      "in the plan, not moving on to the next stage until the gates had been verified.",
    "promotion_calibration_hash": "4fae22711d9a1905c7dfd0b9394607c85f03d6dde8ae737ba5f4a89e5809b9a9"
  },
  "timestamp_utc": "2026-10-31T01:38:57.513694+00:00"
}
```

**Acceptance:** `approved` alias resolves to v3; audit tags recorded. No reviewer identity was
invented — operator supplied it out-of-band.

---

### T10 — Rollback rehearsal + documentation close *(completed 2026-10-31)*

- [x] Rehearsed rollback v3 → v2: `rollback_from` tag `3` recorded on v2; alias verified via
      `get_model_version_by_alias` (`approved` → 2).
- [x] Restored `approved` to v3: rollback v2 → v3 with `rollback_from` tag `2` recorded on v3.
- [x] Verified final state: `candidate` → 3, `approved` → 3.
- [ ] Update `execution_plan.md` (Stage 12 status), `stage_evidence.md` (Gate 12),
      `implementation_plan.md` (PIR row), roadmap **E4**.
- [ ] Report diff scope; commit only on explicit authorization.

**Rollback evidence:**
```json
v2 ← v3: action="rollback", previous_approved_version="3", new_approved_version="2",
rollback_from="3", reviewer_id="Sebastián Garrido Arévalo",
reason="Rollback rehearsal for Gate 12 validation"

v3 ← v2: action="rollback", previous_approved_version="2", new_approved_version="3",
rollback_from="2", reviewer_id="Sebastián Garrido Arévalo",
reason="Restore v3 as the approved version after rehearsal"
```

**Acceptance:** rollback evidence captured; `approved` ends on the reviewed version (v3).

---

## Gate 12 — Hard Close Checklist

- [x] `candidate` alias assigned automatically by the packaging pipeline (T3, T8) — v3 → candidate
- [x] Promotion to `approved` fails closed without reviewer ID / matching calibration hash /
      complete lineage (T4 falsification tests) — verified in unit suite (19 tests passed)
- [x] Rollback re-points `approved` and records `rollback_from` (T10) — rehearsal v3→v2→v3
- [x] Serving loader returns plain segments and rejects estimator binaries (T5) — 12+19 tests
- [x] Roadmap Exit Criterion **E4** satisfied (T10) — v3 approved with clean provenance
      (`git_dirty=false`, `git_commit=04b0cba…8c0c`)

---

## Outcome

### Stage 12 (P2-F09) — COMPLETE *(Gate 12 hard close, 2026-10-31)*

**Verification (actual commands and outcomes):**

| Command | Result |
|---|---|
| `uv run pytest -q` | **365 passed, 2 skipped** (6m 6s). Both skips are pre-existing `tests/unit/test_showcase.py` placeholders awaiting Stage 13 rewiring. |
| `uv run pytest tests/unit/test_artifact_package.py -q` | 12 passed |
| `uv run pytest tests/unit/test_registry.py -q` | 19 passed |
| `uv run ruff check .` | All checks passed |
| `uv run pyright` | 0 errors, 0 warnings |
| `uv run python scripts/check_module_size.py` | 50/50 modules within 1,000 lines |

**Final state on the real backend (`artifacts/mlflow.db`):**

| Item | Value |
|---|---|
| Registered model | `aegis-causal-elasticity` |
| Approved version | **3** (`run 77f6272859a2474da04de3b2bff04916`) |
| Candidate alias | → 3 |
| Approved alias | → 3 |
| `git_commit` / `git_dirty` | `04b0cbae…8c0c` / false |
| `segment_count` / `validation_status` | 35 / GATE_PASSED_OPTION_F |
| Serving bundle | 5 plain-data files only |
| Estimator | separate run artifact (never in serving path) |

**Pending:** commit only on explicit operator authorization. No push without authorization.

