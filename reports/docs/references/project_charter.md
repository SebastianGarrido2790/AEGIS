# Project Charter — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 12, 2026 | Phase 0 — Planning

---

## 1. End State

AEGIS is a production-grade agentic AI platform that recommends insurance premium adjustments by combining causal elasticity estimation with a RAG-grounded regulatory compliance check, escalating to human underwriters whenever a recommendation falls outside pre-approved bounds. Every recommendation ships with its causal justification, its compliance-evidence citation, and its projected financial impact, captured in a structured, auditable decision trail.

## 2. Audience

**Primary:** insurance pricing and actuarial teams and underwriters who need faster, defensible rate-adjustment recommendations without bypassing compliance review.

**Secondary:** a portfolio audience, hiring managers and technical evaluators assessing production-grade agentic systems in regulated domains, who should read this project as evidence of designing _governed, production-grade agentic platforms_, not standalone ML models.

## 3. Problem Framing

**Surface problem:** "Our pricing team can't adjust rates fast enough to keep up with elasticity and competitive shifts."

**Real engineering problem:** there is no shared, auditable decision substrate connecting causal pricing science, regulatory compliance validation, and financial impact assessment — each function operates in an isolated tool or spreadsheet, so speed and auditability trade off against each other by design. The engineering problem is building a governed multi-agent decision loop where causal ML, regulatory grounding, and human oversight are integrated by construction, not reconciled after the fact.

## 4. The ROI Situation

Brutally honest: as a solo portfolio project, this system generates no direct revenue. Its ROI is demonstrative — it is a reusable architectural template proving that causal inference, RAG-grounded compliance, and HITL governance can be integrated into one auditable pipeline.

If deployed inside a real insurer, the ROI case would rest on cycle-time reduction (weeks of actuarial-legal back-and-forth compressed into a same-day auditable recommendation) and reduced compliance exposure (every change pre-validated with evidence, versus after-the-fact review). That claim would require real loss-ratio data to substantiate, which this project does not have access to. Any ROI claim from this artifact should be read as "this pattern is technically sound and reduces a specific, nameable class of operational risk" — not as a validated financial outcome.

## 5. Definition of Done

The system is done when:

- The causal elasticity model outperforms the GLM baseline on held-out data with defensible confidence intervals.
- The Compliance Agent achieves a defined groundedness/evidence-coverage threshold on the adversarial evaluation suite.
- The bandit demonstrates lower cumulative regret than the static-price baseline on the synthetic production-analog stream.
- Every recommendation produces a complete, structured audit record end-to-end.
- The full pipeline runs under CI with reproducibility and coverage gates, containerized and documented to the same standard as the rest of this portfolio.

## 6. Large-Scale Costs

At real-carrier scale, the recurring costs would be:

1. **LLM API calls per recommendation** across three agents — the dominant variable cost, mitigated by semantic caching and by invoking the full agent chain only when Tier 1 flags a segment for review, not on every policy.
2. **Vector store hosting/refresh** for the regulatory corpus — small and infrequently updated, since regulations don't change often, so this is a minor fixed cost.
3. **MLOps overhead** — retraining cadence, monitoring, and drift-detection compute, comparable to any standard ML system in production.
4. **Human review capacity at the HITL gate** — this is the cost that actually dominates at enterprise scale, not compute. If the escalation rate is miscalibrated, the system either creates a compliance-approval bottleneck or lets too much through unchecked. Sizing that gate correctly is a governance cost, not an infrastructure cost.

## 7. Technology Stack

Python 3.12, `uv` for environment management; EconML/DoWhy for causal estimation; scikit-learn for the GLM baseline; LangGraph for multi-agent orchestration; LiteLLM as the in-process LLM Gateway (unified provider abstraction, fallback/resiliency, smart routing, exact/semantic caching, pre-flight input guardrails, cost and trace export); Redis Stack (RedisVL/HNSW) as the shared store for the regulatory RAG index and the Gateway's two-tier prompt cache; Pydantic for state and structured-output validation; MLflow for experiment tracking and model registry; DVC for data/pipeline versioning paired with Great Expectations for CI-blocking data-contract validation on both the elasticity training data and the regulatory corpus ingestion path; FastAPI for the serving layer; Docker/Docker Compose for containerization; GitHub Actions for CI with coverage and reproducibility gates; OTel for observability and tracing.

## 8. Core Concepts

- **Causal elasticity vs. correlational demand** — the distinction the entire ML layer is built on.
- **FTI (Feature-Training-Inference) discipline** governing the deterministic layer.
- **Agent orchestration (LangGraph)** coordinating three specialized agents around a shared, typed state.
- **RAG grounding** — the Compliance Agent's recommendations are only as trustworthy as its retrieval quality; evidence coverage and groundedness are first-class metrics, not afterthoughts.
- **HITL governance** — a bounded corridor of autonomous action with mandatory human escalation outside it.
- **LLM Gateway as mandatory enforcement boundary** — no agent holds a raw provider client or key; provider-level resiliency, pre-flight guardrails, and cost/trace observability are centralized at a single interception point rather than duplicated per agent.
- **Observability/evaluation as a system component** — tracing, LLM-as-judge regression, and drift detection are built in from Phase 0, not retrofitted.

## 9. What Could Go Wrong Along the Way?

- **Confounding in causal estimation:** observational data is fragile — unmeasured confounding could produce a plausible-looking but wrong elasticity estimate. Mitigated by sensitivity analysis and by treating the causal model's output as a recommendation input, never ground truth.
- **Regulatory corpus oversimplification:** real-world insurance regulation is fragmented and multi-jurisdictional; the project must be explicit that this is a single-jurisdiction illustrative model, not a compliance product.
- **RAG grounding failures:** a hallucinated compliance citation is the highest-severity failure mode in this system — a wrong compliance ruling is worse than a wrong price. This drives the evaluation suite's priority and the deterministic fallback design, and is mitigated upstream by a CI-blocking Great Expectations suite on regulatory corpus ingestion (malformed or duplicated chunks are a direct contributor to this failure mode and are rejected before indexing, not caught after the fact).
- **Solo-practitioner scope risk:** three agents, two ML sub-layers, and a full evaluation/observability layer is a large surface for one contributor. Strict phase gating and the `src/` line-count ceiling are the primary controls against scope creep or an unfinished system.
- **Synthetic production-analog data:** may not reflect real portfolio dynamics, which limits how strongly online (bandit) results can be claimed to generalize.
