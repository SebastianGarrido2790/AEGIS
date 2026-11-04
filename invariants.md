# Standing Non-Negotiable Constraints — AEGIS

## Purpose

Durable constraints that MUST remain true across implementation changes. Violating one would materially compromise the architecture, security, data integrity, compliance, reliability, operational safety, or core business requirements of the system.

A rule becomes an invariant only when it is durable, broadly applicable, consequential if violated, independently understandable, and realistically enforceable or verifiable. Temporary implementation preferences MUST NOT be promoted to invariants.

These manuals instruct; they do not govern. This file states constraints in force — it does not replace the ADRs that record why each was decided (see `reports/docs/architecture/system_design.md`).

**Sync note:** this file is the canonical home of the invariant table; `AGENTS.md` §3 (Project Invariants) links here instead of duplicating it. Do not create a second copy of the table anywhere — cite INV IDs. If a discrepancy ever appears between this file and any summary, this file governs until resolved by ADR, surfaced via `.agents/workflows/decision-making.md`.

## Authority and Change Protocol

- Each invariant traces to its origin (ADR, PRD section, or scoping document). The origin preserves rationale and history; the invariant states the constraint currently in force.
- A change conflicting with an invariant requires a new ADR explicitly superseding it — never a silent workaround.
- If a request appears to violate an invariant, or two authorities conflict, stop the affected work and surface the conflict. Do not reinterpret or work around it.
- When a code artifact is governed by an invariant, reference the exact ID in a nearby docstring or module note when that reference materially helps maintainers. State the constraint precisely; do not paste governance text or invent IDs:

```python
"""Standalone promotion and calibration gate functions.

Governed by:
- INV-3 (ADR-003): serving promotion requires both discrimination and
  calibration checks; strong AUC alone is insufficient.
- PRD FR12: block promotion unless both checks pass.
- D-1.7: pure, independently unit-testable functions with no training-pipeline
  or MLflow dependency.
"""
```

## The Invariants

| ID | Invariant | Origin | Enforcement |
|---|---|---|---|
| **INV-1** | **Gateway exclusivity.** No module may import or instantiate a raw LLM provider SDK client. Every LLM call routes through the LiteLLM Gateway. | ADR-001 | CI import-boundary check: build fails if a provider SDK is imported outside `src/aegis/gateway/`. |
| **INV-2** | **Single vector store.** Redis Stack (RedisVL) is the only vector store. No second vector database without a superseding ADR. | ADR-002 | CI dependency audit flags any new vector-store package in `pyproject.toml`. |
| **INV-3** | **Data contracts are blocking.** No dataset or corpus document is DVC-versioned or indexed without first passing its Great Expectations suite. A failed suite halts the pipeline. | ADR-003 | GX validation is a hard upstream dependency of every downstream DVC stage. |
| **INV-4** | **No autonomous publication.** The system recommends a premium adjustment; it never publishes one. Every recommendation terminates at auto-approved-for-review or HITL escalation. | PRD §4; `scoping_doc.md` §§3,7 | Governance output schema has no "published" state; publication is manual, out-of-system. |
| **INV-5** | **Causal output is advisory, not authoritative.** Elasticity/bandit output is never surfaced without accompanying Compliance and Revenue/Loss-Ratio Impact context. | PRD §6; `scoping_doc.md` §11 | Governance output schema and recommendation record require non-null Compliance and Impact fields. (Phase 2 model diagnostic showcase carries an explicit developer-only exemption — $\epsilon$/recovery metrics only, never a proposed adjustment; ADR-011). |
| **INV-6** | **Grounding failure triggers fallback, not guessing.** Grounding is a joint read across Faithfulness, Answer Relevancy, Context Recall, Context Precision — never a single score. Below threshold: fall back to the last compliance-approved rate table and escalate. | ADR-003 rationale; `scoping_doc.md` §11; refined by ADR-010 | Threshold check raises a typed exception in the Compliance node, caught by the governance fallback handler. |
| **INV-7** | **Complete audit record, no exceptions.** Every recommendation produces one complete, structured, independently reconstructable audit record. | PRD FR-7, PRD §11 | Audit schema validation is the last gate before any API response; failure raises. |
| **INV-8** | **Module size ceiling.** No file under `src/` exceeds 1,000 lines. Decompose, don't extend. | Project router (Phase 1) | `uv run python scripts/check_module_size.py`; over-limit fails the build. |
| **INV-9** | **Single-jurisdiction scope is explicit, never implied.** Every compliance verdict states the corpus models one illustrative jurisdiction. | PRD §12, User Story §7 | Fixed disclaimer field in every Compliance output schema. |
| **INV-10** | **No live core-system integration in v1.** Carrier core-system contracts are modeled/mocked, never connected live. | PRD §12 | No production carrier credential slot exists in configuration. |

**Amendment record:** INV-6 was refined from an ad hoc single grounding score to the four-metric joint read above, recorded as ADR-010 — not a silent edit. The current `tier2_agents.grounding_threshold` / `evidence_coverage_threshold` fields in `params.yaml` are Phase 1-era placeholders (ADR-005) and MUST be replaced with the full four-metric threshold set before the Compliance Agent ships (Phase 5/6). Standalone Phase 2 Tier 1 developer showcase / model diagnostic outputs (ADR-009) carry an explicit diagnostic-only exemption from INV-5 ($\epsilon$/recovery metrics only, never a proposed adjustment), which strictly binds pricing recommendations and decision outputs (ADR-011).

**Interim scoping (October 2026):** INV-7 and INV-6 enforcement starts at Phase 7, when audit persistence and the fallback engine land. Phase 6 emits an in-memory recommendation record flagged non-persisted, and a grounding failure in Phase 6 produces `ESCALATED` with no proposal presented as a recommendation (no fallback reversion yet).

## Verification Mapping

| Invariant | Verified by |
|---|---|
| INV-1, INV-2 | CI import-boundary check + dependency audit (see `deployment.md`) |
| INV-3 | GX suites + `dvc repro` gating (see `mlops.md`) |
| INV-4, INV-5, INV-7, INV-9 | Output-schema validation + code review (see `.agents/engineering/agentic-architecture.md`) |
| INV-6 | Four-metric diagnostic matrix (see `evaluation.md`) |
| INV-8 | `scripts/check_module_size.py` (see `python.md`) |
| INV-10 | Configuration review: no carrier credential slot (see `deployment.md`) |

## Related Guidance

- Change protocol: `.agents/workflows/decision-making.md`, `.agents/workflows/change-management.md`
- Decision history: `reports/docs/architecture/system_design.md` (ADR ledger)
- Enforcement in CI: `.agents/engineering/deployment.md`
- Open gaps these invariants do not cover (no kill-switch, no adversarial suite, no cost ceiling): `reports/docs/evaluations/capability_profile_gates.md`
