"""AEGIS system configuration schema.

Strict Pydantic models for domain-nested parameters declared in params.yaml (ADR-005).
Fails loudly on missing keys, invalid types, or unexpected configuration fields.
"""

from pydantic import BaseModel, ConfigDict, Field


class GatewayConfig(BaseModel):
    """Configuration for Tier 1 LiteLLM Gateway (Phase 3)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="gpt-4o-mini", description="Primary LLM provider and model")
    fallback_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Secondary fallback LLM provider and model",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)


class Tier1MLConfig(BaseModel):
    """Configuration for Tier 1 Deterministic ML and Causal Elasticity models (Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.20, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    causal_estimator: str = Field(default="CausalForestDML")
    cv_folds: int = Field(default=5, gt=1)
    exploration_corridor_width: float = Field(default=0.15, gt=0.0, lt=1.0)


class Tier2AgentsConfig(BaseModel):
    """Configuration for Tier 2 LangGraph agentic orchestration (Phase 4-5)."""

    model_config = ConfigDict(extra="forbid")

    groundedness_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    evidence_coverage_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    max_tool_calls: int = Field(default=5, gt=0)
    execution_timeout_seconds: int = Field(default=60, gt=0)


class GovernanceConfig(BaseModel):
    """Configuration for Tier 3 Governance, HITL escalation, and fallback (Phase 7)."""

    model_config = ConfigDict(extra="forbid")

    audit_storage_backend: str = Field(default="sqlite")
    audit_db_path: str = Field(default="artifacts/audit_log.db")
    auto_approval_corridor_pct: float = Field(default=0.05, ge=0.0, le=0.50)
    enforce_hitl_on_flag: bool = Field(default=True)


class DataContractsConfig(BaseModel):
    """Configuration for Great Expectations data contracts and fixtures (Phase 1)."""

    model_config = ConfigDict(extra="forbid")

    elasticity_suite_path: str = Field(..., description="Path to elasticity GX JSON suite")
    regulatory_corpus_suite_path: str = Field(..., description="Path to regulatory GX JSON suite")
    fixtures_dir: str = Field(default="data_contracts/fixtures")
    elasticity_valid_fixture: str = Field(...)
    elasticity_invalid_leakage_fixture: str = Field(...)
    elasticity_invalid_range_fixture: str = Field(...)
    regulatory_valid_fixture: str = Field(...)
    regulatory_invalid_missing_meta_fixture: str = Field(...)
    regulatory_invalid_empty_fixture: str = Field(...)
    regulatory_invalid_duplicate_fixture: str = Field(...)


class DVCConfig(BaseModel):
    """Configuration for DVC local remote storage (ADR-007, R-3)."""

    model_config = ConfigDict(extra="forbid")

    remote_name: str = Field(default="local_storage")
    remote_url: str = Field(default=".dvc/local_remote")


class AEGISConfig(BaseModel):
    """Root configuration model representing params.yaml (ADR-005)."""

    model_config = ConfigDict(extra="forbid")

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tier1_ml: Tier1MLConfig = Field(default_factory=Tier1MLConfig)
    tier2_agents: Tier2AgentsConfig = Field(default_factory=Tier2AgentsConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    data_contracts: DataContractsConfig
    dvc: DVCConfig = Field(default_factory=DVCConfig)
