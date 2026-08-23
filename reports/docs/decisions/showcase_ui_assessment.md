# Showcase UI — Scope Assessment & Decision

**Status: DRAFT — awaiting Sebastián's review and approval.** This is a cross-cutting scope decision, not part of any single roadmap phase — it belongs in `decisions/` per its actual purpose: an assessment of something new we need to decide.

Author: Sebastián Garrido Arévalo (decision), Claude (drafting) | Date: August 16, 2026

---

## Context

The PRD already names a secondary audience distinct from the in-universe personas (Elena, Marco, Renata): hiring managers, technical evaluators, and academic reviewers assessing this as portfolio evidence. Everything built so far speaks to that audience through documentation and code quality — there's no artifact that lets someone *see* the system do something without reading an ADR or running a CLI command.

This is a different question from one already settled. PRD §12 excludes a "full production UI" from scope — that exclusion is about not building a real underwriting console for Marco to use daily. A showcase interface, if built, serves a different purpose entirely: letting a recruiter or evaluator watch the governed multi-agent decision loop actually run, in under two minutes, without cloning the repo. These aren't in tension; they're different artifacts for different audiences, and worth being explicit that approving one doesn't reopen the other.

---

## UI-D1 — Should a showcase interface exist at all?

- **Option A — No UI. Documentation only.** Rely on the architecture walkthrough document (matching the format already used for prior portfolio pieces), the README, diagrams, and the ADR trail to communicate what the system does. Evaluators read; they don't click.
- **Option B — A minimal, deliberately "glass-box" demo interface.** Not a production dashboard — an interface whose entire purpose is to make the multi-agent trace, the governance decision, and the audit record visible as they happen. Explicitly labeled as illustrative, not the production system.
- **Option C — A full, polished web application** styled to resemble a real underwriting console — closer to what PRD §12 already excluded, just rebranded as "for demo purposes."

**Trade-offs:**

Option A costs nothing further but has a real weakness for exactly the audience being asked about: a non-technical recruiter doing an initial screen is not going to read `system_design.md` before deciding whether to move a candidate forward. The system's actual differentiator — that it's *governed*, not just agentic — is invisible in a code listing. You can't see an escalation trigger fire, or a fallback engage, or an audit record assemble, by reading a schema definition.

Option C is the wrong instinct for this specific project, and worth being direct about why: the positioning established since this project's first planning session was explicitly *against* being read as "wrapper for an LLM," and a polished console is the single fastest way to make a governed agentic platform look like a CRUD app with a model bolted on — undercutting the exact narrative this whole project exists to demonstrate. It also reopens the solo-practitioner scope risk flagged repeatedly since Charter §9, for a component that doesn't strengthen the technical story more than a much cheaper option would.

Option B is the one that actually serves the stated goal: making the governance mechanics — the parts that are genuinely differentiated — visible, without pretending to be a production interface or competing for engineering time against Phases 2–9's real work.

**Recommendation:** Option B.

---

## UI-D1a — Tooling, if Option B is approved

- **Option A — Streamlit or Gradio.** Fastest to build, widely recognized in data-science portfolios, minimal code.
- **Option B — Minimal FastAPI + server-rendered HTML** (Jinja2, HTML-over-the-wire), reusing the serving layer already planned for Tier 2/3 rather than introducing a separate demo-app framework.

**Trade-off:** Streamlit/Gradio is meaningfully faster to stand up, but both are strongly associated with quick model demos specifically — a technical evaluator who's seen a hundred Streamlit apps this year may read it as "prototype," not "production engineering," even with the right framing. The FastAPI + Jinja2 route costs more build time but is consistent with the precedent already set on a prior project in this same portfolio, and demonstrates the same full-stack discipline (async FastAPI, server-rendered dashboard, no separate frontend framework dependency) rather than reaching for a purpose-built demo library.

**Recommendation:** FastAPI + Jinja2, for consistency with your own established pattern across this portfolio — but flag this one if build time is the binding constraint once Phase 2–9 scope is clearer. Streamlit is a legitimate fallback if the calendar gets tight; it is not a wrong choice, just a slightly weaker signal for this specific audience.

---

## UI-D1b — When does it get built?

- **Option A — A dedicated final phase** (e.g., a new Phase 10), built once after Phase 9 closes.
- **Option B — Incremental slices**, one added at the close of each phase that produces something worth showing (a simple output view after Phase 2's elasticity/bandit results; the multi-agent trace panel after Phase 6; the audit record and HITL view after Phase 7).

**Trade-off:** Option A is simpler to plan for but concentrates all the UI risk at the very end, after nine phases of real engineering work, when time pressure and fatigue are both highest — exactly when a "minimal" scope tends to quietly expand or quietly get cut. Option B produces portfolio-visible progress throughout the build rather than only at the finish line, and mirrors the project's own phase-gated philosophy: each slice is gated on the same "does this actually work" standard as everything else, not bolted on afterward.

**Recommendation:** Option B, incremental. It's also a stronger interview answer than a dedicated UI phase would be: "the interface grew alongside the system, validated at each phase gate" is a better story than "I built a dashboard at the end."

---

## UI-D1c — What does it actually show?

- **Option A — Fully free-form input.** Let the evaluator type in arbitrary policy data.
- **Option B — Read-only, fixed walkthrough.** A single canned scenario, static.
- **Option C — A curated set of preset scenarios** (3–5), selected from a dropdown, each engineered to demonstrate a specific governance behavior — one within-bounds-and-compliant approval, one bounds violation triggering escalation, one low-groundedness case triggering the fallback.

**Trade-off:** Option A sounds most impressive but is mostly theater here — since INV-10 means there's no live data source, "free-form" really just means typing numbers into a form that still resolves against the same synthetic/fixture-backed logic underneath, with the added cost of needing real input validation and graceful handling of nonsense input for no real benefit. Option B is cheap but only shows the happy path, missing the actual differentiator. Option C costs a bit more scenario design work than B, but it's the only option that puts governance itself on screen — an evaluator watching the fallback deliberately engage on a bad-groundedness scenario sees INV-6 working, which is a stronger demonstration of "governed system" than any number of successful approvals would be.

**Recommendation:** Option C. Specifically include at least one scenario engineered to fail gracefully — the escalation and fallback paths are the actual point of this system, and a demo that only shows success stories undersells exactly what makes it different from a plain pricing model.

---

## If Approved: Downstream Document Updates

Not performed yet, pending your sign-off on the four decisions above — flagged here so nothing gets updated silently:

- **`canvas.md` §4 (Solution):** add the showcase interface as a component, explicitly distinguished from the excluded production UI.
- **`prd.md` §12 (Out of Scope):** clarify that "full production UI" refers to a real underwriting console, not this demo interface — avoid the two being read as contradictory later.
- **`technical_roadmap.md`:** add the incremental UI slice as a deliverable at the close of Phases 2, 6, and 7, rather than a new standalone phase.
- **`system_design.md`:** a new ADR (ADR-009) recording this decision once finalized.
