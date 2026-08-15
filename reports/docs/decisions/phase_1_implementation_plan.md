# Phase 1 — Implementation Plan & Decisions

**Status: DRAFT — awaiting Sebastián's review and approval.**
This is a living document. Decisions marked _No User Input Required_ are recorded here for completeness and ADR discipline, but do not block progress. All other decisions require explicit approval before any Phase 1 code is written. Once a decision is approved, its outcome is recorded as a new ADR in `system_design.md`, and this entry is marked `RESOLVED` in place rather than deleted, so the deliberation trail is preserved.

Author: Sebastián Garrido Arévalo | Date: August 15, 2026

---

## Part A — Current State Audit

Before any decision below, an honest inventory of what actually exists in Phase 1's scope. This audit is based strictly on what has been created to date in this project's Phase 0 planning work.

### A.1 — Phase 1 deliverables: what exists vs. what doesn't

| Artifact (per Roadmap Phase 1)                             | Status               | Detail                                                                                                                                                |
| ---------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repository directory structure                             | **Partially exists** | Only `reports/docs/{references,architecture,runbooks,decisions}` exist. `src/`, `tests/`, `scripts/`, `data_contracts/`, `.github/` do not exist yet. |
| `uv` environment (`pyproject.toml`, `uv.lock`, Python pin) | **Missing**          | Not created. No package name, build backend, or dependency list defined anywhere yet.                                                                 |
| `params.yaml` schema                                       | **Missing**          | Not created. No section structure, no validation approach defined.                                                                                    |
| Great Expectations suite — elasticity training data        | **Missing**          | No `data_contracts/` folder, no suite definition, no fixture data.                                                                                    |
| Great Expectations suite — regulatory corpus ingestion     | **Missing**          | Same as above.                                                                                                                                        |
| DVC pipeline skeleton (`dvc.yaml`, `.dvc/`, `.dvcignore`)  | **Missing**          | DVC has not been initialized. No remote configured.                                                                                                   |
| CI skeleton (`.github/workflows/`)                         | **Missing**          | No workflow file exists. Nothing currently runs on push or PR.                                                                                        |
| `tests/` (unit/integration/evals)                          | **Missing**          | No test scaffold exists, not even a placeholder.                                                                                                      |

**Bottom line:** every Phase 1 deliverable is _to be created_, not _exists but broken_. There is no legacy code to work around. This is a genuine from-scratch scaffold, which simplifies several of the decisions below — there's nothing to migrate.

---

## Part B — Decision Log Summary

| ID    | Decision                                   | Constraint(s) it resolves      | Status                                                              |
| ----- | ------------------------------------------ | ------------------------------ | ------------------------------------------------------------------- |
| P1-D1 | Python package layout & environment        | Modularity                     | **Needs approval**                                                  |
| P1-D2 | `params.yaml` schema & validation strategy | Modularity, cost               | **Needs approval** (D2c: no input needed)                           |
| P1-D3 | Great Expectations suite architecture      | Modularity, cost               | **Partially resolved** — D3b RESOLVED; D3a, D3c still need approval |
| P1-D4 | DVC pipeline skeleton                      | Cost, modularity               | **Needs approval** (D4a: no input needed)                           |
| P1-D5 | CI skeleton design                         | Cost, modularity, dev-velocity | **Needs approval** (D5b: no input needed)                           |

A note on **latency**: it isn't a live factor in any Phase 1 decision. Nothing in this phase sits on a user-facing request path — no agent, no LLM call, no retrieval. Latency becomes a real constraint starting Phase 4 (Gateway) and especially Phase 6 (agentic orchestration). Where I mention it below, it's only to note that a Phase 1 choice avoids constraining Phase 4+ latency later.

---

## Part C — Decisions

### P1-D1 — Python Package Layout & Environment

**Context:** The Phase 0 project structure lists `src/gateway/`, `src/agents/`, etc. directly — it doesn't specify whether `src/` itself is the importable package root or whether there's an intermediate package name. This has to be settled before `pyproject.toml` can be written, since it determines every import statement in the codebase going forward.

#### P1-D1a — Flat vs. namespaced `src` layout

- **Option A — Flat.** `src/gateway/`, `src/agents/`, etc. are top-level importable modules directly (`import gateway`, `from agents.compliance import ...`). Matches the originally drawn project tree exactly.
- **Option B — Namespaced.** Introduce `src/aegis/` as the actual package, with everything else nested under it (`src/aegis/gateway/`, `src/aegis/agents/`, ...). Imports become `from aegis.gateway import ...`.

**Trade-off:** Option A requires no changes to the tree as originally drawn, but top-level module names like `agents`, `tools`, or `schemas` are generic enough to risk shadowing a third-party package of the same name, and it's a nonstandard layout for a `uv`/`hatchling`-built package. Option B is the modern Python packaging convention specifically because it avoids that class of problem, at the cost of a small tree edit (prepend `aegis/` under `src/`).

**Recommendation:** Option B. This is close to a formality — the namespaced `src` layout is the standard `uv`-recommended pattern — but I'm surfacing it as a real decision because it changes a diagram we already agreed on, and touching an already-approved structure deserves your sign-off, not a silent edit.

#### P1-D1b — Build backend

- **Option A — Hatchling.** Widely used, well-documented, the most common pairing with `uv` in current tutorials and templates.
- **Option B — `uv_build`.** `uv`'s own native build backend — fewer moving parts (no separate backend dependency), tightly integrated with the `uv` toolchain you're already standardizing on.
- **Option C — setuptools.** The legacy default; no particular advantage here over A or B.

**Trade-off:** A vs. B is mostly a maturity-vs-integration trade-off — hatchling has a longer track record and more community documentation to lean on if something goes wrong; `uv_build` is newer but removes a dependency and keeps the whole toolchain in one vendor's hands. Setuptools (C) has no advantage for a project with no legacy constraint, so it's effectively eliminated by A and B.

**Recommendation:** Option A (hatchling) — the documentation depth matters more than the marginal integration benefit for a solo-practitioner project where you'll be debugging packaging issues yourself without a team to lean on.

#### P1-D1c — Python version pin

- **Option A — Exact patch pin** (e.g., `==3.12.x` in `.python-version`, matched in `pyproject.toml`).
- **Option B — Minor-version floor** (`>=3.12,<3.13`).

**Trade-off:** Negligible at this project's scale — this is close to a style preference, not an architectural fork.

**Recommendation:** Option B, minor-version floor, since `uv` manages exact interpreter resolution via its own lockfile regardless, and a floor avoids an unnecessary `.python-version` bump every time a patch release ships.

---

### P1-D2 — `params.yaml` Schema & Validation Strategy

**Context:** Referenced constantly across every Phase 0 document (INV-1, INV-2, INV-6; the project's coding conventions) as the single source of truth for every tunable value — but its actual shape has never been defined.

#### P1-D2a — Section structure

- **Option A — Flat namespace.** All keys at one level (e.g., `groundedness_threshold`, `bandit_corridor_width`, `gateway_fallback_chain`), disambiguated by naming convention alone.
- **Option B — Nested by domain.** Top-level sections mirroring the architecture tiers — `gateway:`, `tier1_ml:`, `tier2_agents:`, `governance:`, `data_contracts:`, `dvc:` — with related settings grouped under each.

**Trade-off:** Option A is marginally simpler to grep for a single key, but as the project grows through Phases 2–9 (this file will eventually hold GLM/causal-model hyperparameters, bandit corridor bounds, Gateway routing config, GX thresholds, and governance escalation rules all at once), a flat namespace becomes hard to scan and invites naming collisions between tiers.

**Recommendation:** Option B. This directly serves the modularity constraint — a section boundary in `params.yaml` should mirror the module boundary in `src/`, so a reviewer can find "everything that controls the Compliance Agent's behavior" in one place.

#### P1-D2b — Validation mechanism

- **Option A — Pydantic settings model.** Define a `BaseModel` (or `BaseSettings`) schema that `params.yaml` is loaded into and validated against at startup; malformed or missing values fail fast with a typed error.
- **Option B — Raw YAML load with manual assertions.** Load the file with `yaml.safe_load` and hand-write assertion checks where needed.

**Trade-off:** Option B is faster to stand up in Phase 1 but pushes validation errors to wherever a bad value happens to be _used_, not to load time — directly at odds with the project's fail-loudly principle for configuration and error handling. Option A costs a bit more upfront schema-writing but is consistent with the Pydantic-everywhere convention already established for tool I/O and agent output.

**Recommendation:** Option A. This isn't just consistency for its own sake — a schema-validated config is what makes the CI type-check gate meaningfully cover configuration, not just application code.

#### P1-D2c — Secrets exclusion _(No user input required)_

`params.yaml` will never contain API keys, credentials, or any secret value — full stop. This isn't a judgment call; it follows directly from INV-1 (Gateway exclusivity — provider keys live only in Gateway backend config or a secrets manager) and from the project's standing rule that secrets are sourced from a secrets manager, never committed. Recorded here for completeness of the decision record, not because there's a real alternative to weigh.

---

### P1-D3 — Great Expectations Suite Architecture

**Context:** Two suites are required per the Roadmap — elasticity training data and regulatory corpus ingestion — both CI-blocking per INV-3, both due to prove themselves this phase by failing a deliberately malformed fixture and passing a valid one.

#### P1-D3a — GX API/version to target

**Confidence note, per the project's 80% confidence rule:** Great Expectations' API surface (the "Fluent" datasource/expectation API vs. older patterns, and the split between open-source "GX Core" and "GX Cloud") has moved fast enough in recent releases that I'm not at ≥80% confidence on the exact current-recommended pattern as of today. Rather than guess and risk baking in a soon-to-be-deprecated pattern, I'm flagging this explicitly: **before writing any suite, verify the current stable API against GX's own documentation at implementation time**, rather than relying on my training-data recollection of the API shape.

- **Option A — Current stable GX Core (open-source), file-based expectation suites.** No GX Cloud account, no external service dependency — suites stored as version-controlled files in the repo.
- **Option B — GX Cloud.** Adds an external hosted dependency and, likely, an account/credential — which conflicts with the Local-First, zero-cost-during-development posture this project has held to since Phase 0 (Charter §6), and arguably brushes against the spirit of INV-10 (no live external system dependency in v1) even though INV-10 technically scopes only the carrier core system.

**Recommendation:** Option A, unambiguously — GX Cloud has no upside for a solo, cost-conscious, local-first project and a real downside (external dependency, likely credentials to manage). The only genuinely open question is the exact current API syntax within GX Core, which I'll verify against current docs before implementation rather than asserting here.

#### P1-D3b — Suite storage format

- **Option A — GX's native JSON expectation-suite files**, committed to `data_contracts/`.
- **Option B — Python-defined expectations** (a script that programmatically builds the suite), version-controlled as `.py` files instead.

**Trade-off:** Option A is GX's own default serialization and diffs reasonably cleanly in git; Option B gives more flexibility for programmatically generated expectations (useful if the leakage check in the elasticity suite ends up needing custom logic beyond GX's built-in expectation types) at the cost of being slightly less declarative/scannable.

**Recommendation:** Option A for the regulatory corpus suite (its checks — non-empty chunks, required metadata, duplicates — are all standard, declarative expectation types with no custom logic needed). Option B for the elasticity training suite specifically, since the post-treatment leakage check (Charter §9, PRD FR) is domain-specific logic unlikely to map cleanly onto a built-in GX expectation type and will likely need a custom expectation function. This is the one place in this document I'm recommending a genuine split rather than one answer for both suites — flag if you'd rather keep both suites in the same format for consistency, even at the cost of forcing the leakage check into a less natural shape.

**Resolution (approved by Sebastián, Aug 14, 2026):** going with fit over consistency, as recommended — JSON-based expectation suite for the regulatory corpus; Python-defined suite for the elasticity training data, to accommodate the custom leakage-check logic.

#### P1-D3c — Fixture/test-data strategy

- **Option A — Hand-crafted minimal fixtures.** A handful of small, deliberately constructed CSV/JSON rows per suite — one clearly valid, one or two clearly malformed in a specific, named way (missing metadata field, negative exposure value, duplicate chunk, a leaked post-treatment column) — committed to `data_contracts/fixtures/`.
- **Option B — Pull a real sample now.** Acquire a small slice of the actual French Motor Third-Party Liability dataset and a small slice of actual NAIC/regulatory text now, in Phase 1, rather than waiting for Phase 2/5.

**Trade-off:** Option B gets real data into the repo sooner and could save a step later, but it pulls Phase 2 and Phase 5 scope forward into Phase 1's 1–2 week box, and real data is a worse tool than a hand-crafted fixture for this specific exit criterion — proving a _specific, named_ failure mode (e.g., "this exact row is missing the effective-date metadata field") is easier to construct and easier to verify than hoping a real dataset happens to contain a naturally-occurring violation of each rule.

**Recommendation:** Option A. Keep Phase 1 scoped to exactly what its exit criteria ask for — the suites demonstrably catching known-bad and passing known-good — and defer real dataset acquisition to the phases that actually need it (Phase 2 for elasticity data, Phase 5 for the regulatory corpus). This also keeps CI fast, since hand-crafted fixtures are tiny.

---

### P1-D4 — DVC Pipeline Skeleton

**Context:** Roadmap Phase 1 calls for a DVC pipeline skeleton "wired to both GX gates," with raw datasets DVC-tracked by the end of the phase.

#### P1-D4a — Remote storage backend _(No user input required)_

A local filesystem remote (a local directory outside the repo, or the DVC default cache) is the only correct choice for Phase 1. This follows from three independent constraints all pointing the same direction: INV-10 (no live external system integration in v1), the zero-cost posture established in Charter §6, and the Local-First development principle already established as a standard across this portfolio. A cloud remote (S3, GCS, Azure Blob) introduces credentials, cost, and an external dependency for no benefit at this phase — there's no team to share data with yet and no CI runner that needs remote access, since Phase 1's fixtures are tiny and can live in the repo's own DVC cache. Recorded for completeness; revisit only if/when a genuine multi-environment or team-sharing need arises, which is not a Phase 1 concern.

#### P1-D4b — Pipeline stage granularity

- **Option A — Coarse.** One DVC stage per data path (elasticity, regulatory corpus), each stage doing ingest + GX validation + versioning together in a single script invocation.
- **Option B — Fine-grained.** Separate stages for raw ingest, GX validation, and validated/versioned output, so the DVC DAG shows validation as its own explicit node.

**Trade-off:** Option A is fewer moving parts to wire correctly within the phase's 1–2 week box. Option B costs a little more setup now but pays for itself immediately in DVC's own core value proposition — with fine-grained stages, DVC only reruns the stage whose inputs actually changed (e.g., re-running validation alone without re-ingesting), and the DAG becomes a legible diagram of the actual data-contract enforcement point, not something you have to read the script to find.

**Recommendation:** Option B. This is a direct instance of the modularity constraint you asked me to weigh explicitly — decomposed stages are the DVC-pipeline equivalent of INV-8's module-size philosophy, and the added setup cost is small relative to the benefit across the rest of the project's lifetime, not just Phase 1.

---

### P1-D5 — CI Skeleton Design

**Context:** Roadmap Phase 1 exit criteria: "CI is green on the scaffold," and each GX suite demonstrably fails/passes its fixtures — both need to run automatically, not just locally.

#### P1-D5a — Workflow structure

- **Option A — Single combined workflow** (`.github/workflows/ci.yml`) running lint → type-check → module-size check → GX gates (against fixtures) → tests as sequential jobs/steps, one pass/fail status.
- **Option B — Split workflows** (e.g., a `code-quality.yml` and a separate `data-contracts.yml`), each with its own status badge.

**Trade-off:** Option B gives more granular visibility (you can see at a glance whether a failure was a lint issue or a data-contract issue without opening the run) and could run in parallel, but adds workflow-file overhead disproportionate to this project's current size, and the Phase 1 exit criterion itself is phrased as a single binary — "CI is green" — not "both workflows are green."

**Recommendation:** Option A. Revisit splitting later if/when CI runtime or job count grows enough that a single workflow becomes hard to read — not a Phase 1 concern given how small this scaffold is.

#### P1-D5b — Secrets requirement this phase _(No user input required)_

Phase 1's CI needs zero API keys or secrets. No agent, LLM call, or Gateway path exists yet — everything CI touches this phase (lint, type-check, module-size check, GX suites against local fixtures, DVC repro against local remote) is fully self-contained. Recorded for completeness; the first phase that will actually need a CI secret is Phase 4 (Gateway integration, provider credentials) at the earliest, and even then likely only for a mocked/sandboxed test call rather than a real provider key in CI.

#### P1-D5c — Wire the INV-8 module-size checker into CI now?

- **Option A — Yes, now.** Add `scripts/check_module_size.py` and its CI step in Phase 1, even though `src/` is nearly empty.
- **Option B — Defer.** Add it once there's actually meaningful code to check, e.g., Phase 2.

**Trade-off:** Genuinely small either way, but Option A costs almost nothing today (the script and CI step are trivial against an empty tree) and means the 1,000-line ceiling is enforced from the very first file written in Phase 2 onward, rather than retrofitted after code already exists that might violate it.

**Recommendation:** Option A. Cheap now, and "enforced from day one" is the entire premise behind INV-8 being a hard invariant rather than a style guideline.

#### P1-D5d — Minimal `pytest` skeleton now, or defer?

- **Option A — Stand up a minimal skeleton now**, with a single trivial test (e.g., confirming the `params.yaml` schema loads and validates) so the CI "tests" stage isn't vacuous.
- **Option B — Defer `tests/` entirely to Phase 2**, since there's no application logic yet to meaningfully unit-test.

**Trade-off:** Option B keeps Phase 1 tightly scoped to exactly its stated deliverables. Option A means the CI pipeline described in the project's testing standard (lint → type-check → unit tests → integration tests → ...) is actually fully wired end-to-end by the end of Phase 1, rather than having an empty stage that silently passes because there's nothing to run — which is a minor but real gap between "CI is green" and "CI is meaningfully green."

**Recommendation:** Option A, but scoped minimally — one test, confirming config loads correctly. This isn't scope creep so much as making sure Phase 1's own exit criterion ("CI is green on the scaffold") is checking something real at every stage it claims to run, not passing trivially by having nothing to execute.

---

## Part D — What Happens After Approval

Once you've reviewed and approved (or amended) the decisions above:

1. `system_design.md` gets updated to reflect each finalized Phase 1 architectural decision as a new ADR (ADR-004 onward), per its own stated update discipline — decisions/ documents like this one are the deliberation trail; `system_design.md` is where the outcome becomes load-bearing record.
2. This document's entries stay marked `RESOLVED` in place, not deleted — preserving the trade-off reasoning for future reference rather than just the final answer.
3. Only then does implementation begin — no code has been written as part of this document, per your instruction.
