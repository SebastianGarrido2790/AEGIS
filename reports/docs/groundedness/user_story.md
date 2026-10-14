# User Story & Problem Framing — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026 | Phase 0 — Planning

---

## 1. Personas & Stories

### Persona 1 — Elena, Senior Pricing Actuary

Builds and defends elasticity and rate models. Measured on how quickly and how defensibly her team can propose rate changes that hold up under both financial and regulatory scrutiny.

- **Story 1.1:** As Elena, I want a causal (not correlational) elasticity estimate per segment, so that I can defend a proposed rate change on genuine treatment-effect grounds rather than a demand proxy that could be confounded by factors I haven't controlled for.
- **Story 1.2:** As Elena, I want to see the compliance evidence attached to my proposal _before_ I send it to Marco, so that I can catch a regulatory conflict early and revise my proposal myself instead of having it bounced back after a multi-week legal review.

### Persona 2 — Marco, Underwriting & Compliance Manager

Reviews and approves or rejects proposed rate changes. Measured on regulatory exposure avoided and on review throughput without cutting corners.

- **Story 2.1:** As Marco, I want every proposal that reaches my desk to already carry a compliance evidence citation, so that my review time is spent on genuine judgment calls, not on re-deriving whether a proposal is even in scope for approval.
- **Story 2.2:** As Marco, I want a complete audit record for every decision — approved, rejected, or escalated — so that when a regulator asks why a rate changed, I can produce the full justification chain without reconstructing it from memory or scattered emails.

### Persona 3 — Renata, Chief Pricing Officer

Owns the P&L and competitive pricing strategy. Measured on portfolio-level financial performance and on the organization's audit-readiness as a whole, not any single rate decision.

- **Story 3.1:** As Renata, I want visibility into the projected revenue and loss-ratio impact of every proposed rate change, not just the ones flagged as risky, so that I can see the aggregate financial picture the pricing team is producing, not only the exceptions.
- **Story 3.2:** As Renata, I want confidence that no rate change reaches the market without passing both a financial and a regulatory check, so that I can represent our pricing process as governed and defensible to the board and to regulators, without having to personally verify each case.

## 2. The 5 Whys (Root Cause Analysis)

1. **Why** does it take weeks to adjust a rate that the data already justifies? Because the elasticity analysis, the compliance review, and the financial sign-off happen sequentially, in separate tools, by separate teams.
2. **Why** do they happen sequentially instead of in parallel? Because none of the three functions has a shared, structured artifact they can all act on simultaneously — each waits for the prior team's output before starting their own review.
3. **Why** is there no shared artifact? Because the elasticity model, the compliance check, and the financial projection were each built as standalone tools solving their own team's problem, with no integration layer connecting them.
4. **Why** was no integration layer built? Because building one requires expertise spanning causal ML, regulatory/RAG systems, and agentic orchestration — a combination that doesn't naturally sit inside any single existing team's mandate.
5. **Why** does that combination matter now specifically? Because the class of system that closes this gap — deterministic ML models governed by agentic orchestration with built-in compliance grounding and human escalation — is only now mature enough (LangGraph, production RAG, LLM gateways) to build reliably. **Root cause:** the organizational gap is real, but it has persisted because the technical substrate to close it safely has not been standard practice until recently — this is a capability gap, not a willingness gap.

## 3. Jobs-to-Be-Done (JTBD)

- When I have evidence that a rate segment's retention is drifting, I want to produce a defensible adjustment proposal quickly, so I can act while the market conditions that motivated it are still current.
- When I receive a rate change proposal, I want to know immediately whether it's regulatorily sound, so I can focus my judgment on genuine trade-offs rather than compliance triage.
- When I'm asked to justify a pricing decision after the fact — to a regulator, an auditor, or the board — I want a complete, reconstructable record, so I don't have to manually rebuild the justification from disconnected sources.

## 4. Problem Statement (User Perspective)

From the actuary's chair: "I can see the number is stale. I don't have a fast, defensible way to prove the new number is right _and_ compliant _and_ worth the revenue trade-off, all at once — so by the time all three checks clear, I'm proposing a fix for a problem that's already changed shape."

From the compliance reviewer's chair: "I'm the last checkpoint before a rate goes live, but I'm often the first person to actually see the regulatory implications spelled out — which means I either become the bottleneck, or I approve faster than I'm fully comfortable with."

## 5. Failure Mode Analysis (What Breaks Without This System)

| Failure mode                                                     | Consequence without AEGIS                                                                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Elasticity model built without causal rigor                      | Rate changes based on correlation get challenged in review, wasting the cycle time they were meant to save                                                    |
| Compliance review happens only at the end                        | Regulatory conflicts surface late, forcing full proposal rework instead of early correction                                                                   |
| No shared audit trail across the three functions                 | Post-hoc regulatory inquiries require manual reconstruction from emails and spreadsheets, increasing both response time and error risk                        |
| No financial projection attached to compliance-cleared proposals | Leadership sees only the exceptions that get escalated, not the aggregate financial pattern across all proposals — a distorted picture of pricing performance |
| No bounded exploration corridor                                  | Any automation introduced without guardrails either moves too conservatively to matter or exposes the business to unreviewed pricing risk                     |

## 6. User Journey Map

**Current State:**

1. Elena builds a rate proposal using a correlational demand model.
2. Proposal is emailed to compliance/legal for manual review — no shared state, no structured evidence.
3. Compliance review takes 1–3 weeks; findings come back as unstructured commentary, sometimes requiring a full rebuild of the proposal.
4. Once compliance clears it, finance reviews the revenue/loss-ratio impact separately, potentially surfacing objections that reopen the cycle.
5. Marco approves or rejects based on whichever version of the proposal reaches him last, without a consolidated record of how it evolved.
6. If a regulator later asks why the rate changed, the justification is manually reconstructed from emails, spreadsheets, and institutional memory.

**Future State (with AEGIS):**

1. Elena submits a segment for review; the causal elasticity model and bounded bandit proposal are generated immediately.
2. The Compliance Agent retrieves and cites relevant regulatory material in the same pass, returning a pass/flag verdict with evidence — no separate email cycle.
3. The Revenue/Loss-Ratio Impact Agent attaches a financial projection to the same proposal, before it reaches any human.
4. If the proposal is within bounds and compliant, it reaches Marco already substantiated, ready for routine sign-off; if not, it's escalated directly with full context, not routed through an ad hoc review chain.
5. Every step — inputs, agent outputs, evidence citations, financial projection, and final human decision — is captured in one structured audit record from the start.
6. A later regulatory inquiry is answered by retrieving the existing audit record, not reconstructing one.

## 7. Constraints Acknowledged by Users

- Users understand AEGIS does not replace their judgment or their regulatory accountability — it accelerates and documents the analysis they are still responsible for approving.
- Users accept that AEGIS operates on a single illustrative jurisdiction's regulatory corpus in this version, not their full real-world multi-state compliance landscape.
- Users accept that the bounded exploration corridor is intentionally conservative — the system is designed to escalate rather than guess when a proposal is ambiguous, which means some proposals that a confident human might approve outright will still be routed to review.
- Users accept that this version operates on a synthetic production-analog stream, not a live policy feed, so online (bandit) performance claims are illustrative of the mechanism, not a validated real-portfolio result.
