# Phase 1 — Execution Workflow

**Purpose:** translate the approved Phase 1 decisions (`phase_1_implementation_plan.md`, recorded as ADR-004 through ADR-008 in `system_design.md`) into a concrete, sequential build order. No code is written in this document — each stage below describes _what_ gets built and _how it's verified_, not the implementation itself.

**Rule for every stage:** the gate at the end of a stage must pass before the next stage begins. A stage is not "mostly done" — it is done, gate-verified, or not started. If a gate fails, work stays in that stage until it passes; nothing downstream gets touched in the meantime.

**Standing rule, from Stage 2 onward:** lint, type-check, and the module-size checker (INV-8) are re-run at the end of _every_ subsequent stage, not just once. They're cheap enough to run constantly, and re-running them catches a regression at the stage that introduced it rather than at the end of Phase 1.

Author: Sebastián Garrido Arévalo | Date: August 15, 2026

---

## Stage 1 — Repository & Environment Bootstrap

**Implements:** ADR-004 (namespaced `src/aegis/`, Hatchling, Python `>=3.12,<3.13`).

**Objective:** a clean, reproducible Python environment and directory skeleton exist before anything else is written into them.

**Steps:**

1. Confirm or initialize the git repository at the project root; add a `.gitignore` covering `__pycache__/`, `.venv/`, `uv.lock`-adjacent build artifacts (not `uv.lock` itself, which is committed), local DVC cache paths, and any `.env*`.
2. Create the full Phase 1 directory skeleton: `src/aegis/` (namespaced package root, with placeholder subpackages for `gateway/`, `agents/`, `governance/`, `pipelines/`, `bandit/`, `tools/`, `utils/`, `schemas/`, `config/` — empty except for `__init__.py` stubs, since none of these have real logic yet), `data_contracts/` and `data_contracts/fixtures/`, `tests/{unit,integration,evals}`, `scripts/`, `.github/workflows/`.
3. Initialize the `uv` environment: `pyproject.toml` declaring the package name, Hatchling as the build backend, and the Python requirement `>=3.12,<3.13`; `.python-version` pinned to `3.12`.
4. Declare Phase 1's actual dependency set — nothing speculative for later phases. Runtime: `pydantic` (config schema), `pyyaml`, `great-expectations`, `dvc`. Dev: `ruff`, `pyright` (or `basedpyright`), `pytest`.
5. Run `uv sync` to resolve and lock the environment (`uv.lock` committed).

**Gate 1 — must pass before Stage 2 begins:**

- `uv sync` completes cleanly on a fresh checkout (no manual intervention).
- `uv run python -c "import aegis"` resolves the namespace package without error, even though it currently contains no real logic.
- `uv run ruff check .` and `uv run pyright` both execute without configuration errors (passing trivially on an near-empty tree is fine — the point is that the tools are correctly wired, not that there's anything substantial to check yet).

---

## Stage 2 — Configuration Schema & Module-Size Guard

**Implements:** ADR-005 (domain-nested `params.yaml` with Pydantic validation); the INV-8 enforcement script approved under P1-D5c.

**Objective:** the single source of truth for tunable values exists and fails loudly on bad input, and the 1,000-line ceiling is enforced from this point forward, not retrofitted later.

**Steps:**

1. Create `params.yaml` with the domain-nested section skeleton from ADR-005: `gateway:`, `tier1_ml:`, `tier2_agents:`, `governance:`, `data_contracts:`, `dvc:`. Populate only what Phase 1 actually uses — `data_contracts:` (suite paths, fixture paths) and `dvc:` (local remote path, stage names) get real values; `gateway:`, `tier1_ml:`, `tier2_agents:`, and `governance:` exist as sections but stay minimal or empty until the phases that build those tiers.
2. Build the Pydantic configuration schema in `src/aegis/config/` — a `BaseModel`/`BaseSettings` hierarchy mirroring the `params.yaml` sections, with a single loader entry point the rest of the codebase will import from.
3. Build `scripts/check_module_size.py` — walks `src/aegis/`, flags any `.py` file at or over 1,000 lines, exits non-zero if any are found.

**Gate 2 — must pass before Stage 3 begins:**

- The valid `params.yaml` loads and validates cleanly through the Pydantic schema.
- A deliberately malformed copy (one required key removed, one value given the wrong type) is loaded against the same schema and produces a clear, typed validation error at load time — not a downstream failure somewhere it's used.
- Manual review confirms zero secret-shaped values anywhere in `params.yaml` (no API keys, tokens, or credentials — per P1-D2c/INV-1, this should be trivially true since nothing in Phase 1 needs one, but it's checked, not assumed).
- `uv run python scripts/check_module_size.py` runs clean (exit 0) against the current `src/aegis/` tree.

---

## Stage 3 — Data Contract: Regulatory Corpus Suite

**Implements:** ADR-006, regulatory-corpus half (JSON-based GX Core suite, hand-crafted fixtures).

**Objective:** prove the simpler of the two suites first — its checks are all standard, declarative GX expectation types with no custom logic, so it's the right place to validate the GX Core wiring pattern before layering a custom expectation class on top of it in Stage 4.

**Steps:**

1. Define the suite in `data_contracts/` as a native GX Core JSON expectation suite, covering: non-empty chunks, required metadata fields present (jurisdiction, section, effective date), and duplicate/near-duplicate chunk detection.
2. Build the fixtures in `data_contracts/fixtures/`: one clearly valid sample chunk set, and one malformed fixture per rule — a chunk missing the effective-date field, a chunk that's empty or whitespace-only, and a duplicate pair. Each malformed fixture should trip exactly one named rule, not several at once, so a failure clearly indicates which check caught it.

**Gate 3 — must pass before Stage 4 begins:**

- Running the suite against the valid fixture passes cleanly.
- Running the suite against each malformed fixture fails, and the failure specifically names the rule that was violated (not a generic suite-level failure) — confirming the suite catches the _right_ problem, not just _a_ problem.
- Re-run lint, type-check, and the module-size checker (standing rule).

---

## Stage 4 — Data Contract: Elasticity Training Suite

**Implements:** ADR-006, elasticity half (JSON-based suite plus a custom GX Expectation class for the post-treatment leakage check).

**Objective:** the more structurally interesting of the two suites — most of its checks are standard (schema conformance, value ranges), but the post-treatment leakage check has no natural built-in GX expectation type and needs a custom Expectation class, referenced from the same JSON suite rather than breaking format consistency with Stage 3.

**Steps:**

1. Define the suite in `data_contracts/`, again as a native JSON file, covering: schema conformance against the expected policy/claims columns, value-range checks (non-negative exposure, valid claim counts).
2. Implement the post-treatment leakage check as a custom GX Expectation class (small, isolated Python module — this is the one piece of "logic" in this phase's data-contract work, and it's exactly the case ADR-006 anticipated needing this pattern for). Register it so the JSON suite can reference it by name like any built-in expectation type.
3. Build the fixtures: one valid row set, one with a value-range violation (e.g., negative exposure), one that specifically trips the leakage check (a feature that's only knowable after the pricing decision was made).

**Gate 4 — must pass before Stage 5 begins:**

- Valid fixture passes; each malformed fixture fails against the specific rule it was built to violate.
- The leakage-check fixture specifically proves the custom Expectation class is actually executing as part of the suite run — not just present in the codebase unreferenced. This is the one check in this phase worth double-checking by hand, since it's the only non-declarative logic in either suite.
- Standing rule: lint, type-check, module-size checker re-run clean.

---

## Stage 5 — DVC Pipeline Skeleton

**Implements:** ADR-007 (local filesystem remote, fine-grained `ingest → validate_gx → version` stages).

**Objective:** wire both suites into an actual DVC-managed pipeline, not just prove they work standalone — this is what makes the data contracts CI-blocking and DVC-tracked, not just locally runnable scripts.

**Steps:**

1. Initialize DVC in the repository; configure a local filesystem cache/remote (no cloud credentials, per ADR-007 and INV-10).
2. Define `dvc.yaml` with two parallel three-stage pipelines — one for the elasticity path, one for the regulatory corpus path — each following `ingest → validate_gx → version`. The `validate_gx` stage for each path invokes the corresponding suite from Stage 3 or Stage 4 against whatever the `ingest` stage produced; the `version` stage only runs if `validate_gx` passed.
3. Track the Stage 3/4 fixtures through DVC as the Phase 1 stand-in for "raw data," since real dataset acquisition is explicitly deferred to Phase 2 (elasticity) and Phase 5 (regulatory corpus) per the approved fixture-strategy decision.

**Gate 5 — must pass before Stage 6 begins:**

- `dvc repro` runs the full DAG end-to-end for both paths against the valid fixtures, completing through the `version` stage.
- Swapping in a malformed fixture at the `ingest` stage causes `validate_gx` to fail visibly and blocks `version` from running — this is the proof that DVC and GX are actually wired together, not just that the GX suites work in isolation (already shown in Stages 3–4).
- `dvc dag` shows the expected fine-grained stage structure (three distinct nodes per path, not one collapsed stage) — a quick visual confirmation that ADR-007's granularity decision was actually implemented, not simplified away under time pressure.
- Standing rule: lint, type-check, module-size checker re-run clean.

---

## Stage 6 — CI Skeleton

**Implements:** ADR-008 (single unified GitHub Actions workflow, zero secrets, INV-8 enforced from day one, minimal `pytest` skeleton).

**Objective:** everything built in Stages 1–5 running automatically on every push/PR — this stage is where Phase 1's own exit criterion is actually met, not just locally reproducible.

**Steps:**

1. Build the minimal `pytest` skeleton: a single test in `tests/unit/` confirming the Stage 2 configuration loads and validates correctly (the one test scoped by P1-D5d — deliberately not more than this yet).
2. Write `.github/workflows/ci.yml` as one sequential workflow: lint (`ruff`) → type-check (`pyright`) → module-size check (`scripts/check_module_size.py`) → GX gates (`dvc repro` against the fixtures, or the suites invoked directly) → tests (`pytest`).
3. Confirm no step in the workflow references any secret or credential — nothing in Phase 1's scope needs one.

**Gate 6 — Phase 1's actual exit criterion, must pass before Phase 2 begins:**

- CI is green end-to-end on a clean push.
- A one-time, deliberate falsification pass: independently introduce one failure per gate — a lint violation, a type error, a dummy file padded past 1,000 lines, a malformed fixture swapped into the DVC-tracked data, a broken assertion in the one pytest test — and confirm each one independently turns CI red. Then revert each and confirm CI returns to green. This is the concrete proof that every gate in the pipeline is load-bearing, not decorative — a green CI that's never actually seen a failure hasn't demonstrated anything yet.
- Once this falsification pass is complete and reverted, Phase 1 is done. `system_design.md`'s per-phase update protocol (§6) applies: mark ADR-004 through ADR-008 as **Validated** if implementation matched the decision as written, or **Amended** with a dated entry if anything diverged during the build.

---

## Summary Table

| Stage | Builds                                                      | Gate proves                                                  | Blocks until gate passes |
| ----- | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------ |
| 1     | Repo skeleton, `uv` env, `pyproject.toml`                   | Environment reproducible, package importable                 | Stage 2                  |
| 2     | `params.yaml` + Pydantic schema, module-size checker        | Config fails loudly on bad input, size guard active          | Stage 3                  |
| 3     | Regulatory corpus GX suite + fixtures                       | Suite catches the right named failure, passes valid data     | Stage 4                  |
| 4     | Elasticity GX suite + custom leakage-check class + fixtures | Custom expectation logic actually executes, not just present | Stage 5                  |
| 5     | DVC pipeline (`ingest → validate_gx → version` × 2)         | GX gates are wired into DVC, not just standalone-runnable    | Stage 6                  |
| 6     | CI workflow, minimal pytest                                 | Every gate is load-bearing under deliberate falsification    | Phase 2                  |
