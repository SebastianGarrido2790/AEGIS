"""AEGIS system configuration schema.

Strict Pydantic models for domain-nested parameters declared in params.yaml (ADR-005).
Fails loudly on missing keys, invalid types, or unexpected configuration fields.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GatewayConfig(BaseModel):
    """Configuration for Tier 1 LiteLLM Gateway (Phase 3)."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="gpt-4o-mini", description="Primary LLM provider and model")
    fallback_model: str = Field(
        default="claude-sonnet-4-6",
        description="Secondary fallback LLM provider and model",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)


class FeaturePipelineConfig(BaseModel):
    """Configuration for the deterministic dual-view feature pipeline (Phase 2, P2-F02)."""

    model_config = ConfigDict(extra="forbid")

    exposure_cap: float = Field(default=1.0, gt=0.0)
    driver_age_bounds: tuple[int, int] = Field(default=(18, 90))
    car_age_bounds: tuple[int, int] = Field(default=(0, 34))
    driver_age_band_edges: list[int] = Field(default=[18, 25, 45, 65, 100])
    driver_age_band_labels: list[str] = Field(
        default=["18-24", "25-44", "45-64", "65+"]
    )
    car_age_band_edges: list[int] = Field(default=[0, 5, 10, 101])
    car_age_band_labels: list[str] = Field(default=["0-4", "5-9", "10+"])
    density_band_edges: list[float] = Field(default=[0, 501, 2001, 1000000000])
    density_band_labels: list[str] = Field(default=["<=500", "501-2000", ">2000"])
    power_levels: list[str] = Field(
        default=["d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
    )
    brand_levels: list[str] = Field(
        default=[
            "Renault, Nissan or Citroen",
            "Japanese (except Nissan) or Korean",
            "Opel, General Motors or Ford",
            "Volkswagen, Audi, Skoda or Seat",
            "Mercedes, Chrysler or BMW",
            "Fiat",
            "other",
        ]
    )
    gas_levels: list[str] = Field(default=["Regular", "Diesel"])
    region_levels: list[str] = Field(
        default=[
            "Centre",
            "Ile-de-France",
            "Bretagne",
            "Pays-de-la-Loire",
            "Aquitaine",
            "Nord-Pas-de-Calais",
            "Poitou-Charentes",
            "Basse-Normandie",
            "Haute-Normandie",
            "Limousin",
        ]
    )

    @model_validator(mode="after")
    def _validate_bands_and_levels(self) -> "FeaturePipelineConfig":
        """Validates edge monotonicity, label alignment, and factor level uniqueness."""
        band_specs = [
            (
                "driver_age",
                self.driver_age_band_edges,
                self.driver_age_band_labels,
            ),
            ("car_age", self.car_age_band_edges, self.car_age_band_labels),
            ("density", self.density_band_edges, self.density_band_labels),
        ]
        for name, edges, labels in band_specs:
            if len(edges) != len(labels) + 1:
                raise ValueError(
                    f"{name} band edges ({len(edges)}) must equal labels ({len(labels)}) + 1"
                )
            if any(left >= right for left, right in zip(edges, edges[1:], strict=False)):
                raise ValueError(f"{name} band edges must be strictly increasing")
        for name, bounds in [
            ("driver_age", self.driver_age_bounds),
            ("car_age", self.car_age_bounds),
        ]:
            if bounds[0] >= bounds[1]:
                raise ValueError(f"{name} bounds must be strictly increasing")
        level_groups = [
            ("power", self.power_levels),
            ("brand", self.brand_levels),
            ("gas", self.gas_levels),
            ("region", self.region_levels),
        ]
        for name, levels in level_groups:
            if not levels:
                raise ValueError(f"{name} levels must not be empty")
            if len(set(levels)) != len(levels):
                raise ValueError(f"{name} levels must be unique")
        return self


class FeaturePathsConfig(BaseModel):
    """Configuration for deterministic feature-pipeline output paths (Phase 2, P2-F02)."""

    model_config = ConfigDict(extra="forbid")

    glm_view: str = Field(default="data/processed/features_glm.csv")
    forest_view: str = Field(default="data/processed/features_forest.csv")
    metadata: str = Field(default="data/processed/feature_metadata.json")


class GLMBaselineConfig(BaseModel):
    """Configuration for the cap-parameterized Tweedie baseline (Phase 2, P2-F03)."""

    model_config = ConfigDict(extra="forbid")

    default_cap: float = Field(default=100000.0, gt=0.0)
    tweedie_p_min: float = Field(default=1.05, gt=1.0, lt=2.0)
    tweedie_p_max: float = Field(default=1.95, gt=1.0, lt=2.0)
    tweedie_p_step: float = Field(default=0.05, gt=0.0)
    tweedie_p_refine_step: float = Field(default=0.01, gt=0.0)
    tweedie_p_fallback: float = Field(default=1.5, gt=1.0, lt=2.0)
    decile_count: int = Field(default=10, gt=1)
    link: str = Field(default="log")

    @model_validator(mode="after")
    def _validate_grid(self) -> "GLMBaselineConfig":
        """Validates the variance-power grid and link."""
        if self.tweedie_p_min >= self.tweedie_p_max:
            raise ValueError("tweedie_p_min must be less than tweedie_p_max")
        if self.link != "log":
            raise ValueError("only the log link is supported for the Tweedie baseline")
        return self


class GLMPathsConfig(BaseModel):
    """Configuration for GLM baseline output paths (Phase 2, P2-F03)."""

    model_config = ConfigDict(extra="forbid")

    fitted_loss_cost: str = Field(default="data/processed/glm_fitted_loss_cost.csv")
    model_metadata: str = Field(default="data/processed/glm_baseline_model.json")
    calibration_report: str = Field(default="data/processed/glm_calibration_report.json")


class CellMergeConfig(BaseModel):
    """Approved merge of a sub-floor actuarial cell into an adjacent density band (P2-F05)."""

    model_config = ConfigDict(extra="forbid")

    driver_age_band: str
    car_age_band: str
    from_density_band: str
    to_density_band: str
    merged_density_band: str


class SegmentsConfig(BaseModel):
    """Configuration for the Tier 1 actuarial segment grid (Phase 2, P2-F05)."""

    model_config = ConfigDict(extra="forbid")

    min_policy_count: int = Field(default=1000, ge=1)
    premium_loading: float = Field(default=0.20, ge=0.0)
    cell_merges: list[CellMergeConfig] = Field(default_factory=list)


class SegmentPathsConfig(BaseModel):
    """Configuration for segment pipeline output paths (Phase 2, P2-F05)."""

    model_config = ConfigDict(extra="forbid")

    segment_loss_cost: str = Field(default="data/processed/segment_loss_cost.csv")
    segment_definitions: str = Field(default="data/processed/segment_definitions.json")
    policy_segments: str = Field(default="data/processed/policy_segments.csv")


class Tier1MLConfig(BaseModel):
    """Configuration for Tier 1 Deterministic ML and Causal Elasticity models (Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(default=0.20, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    causal_estimator: str = Field(default="CausalForestDML")
    cv_folds: int = Field(default=5, gt=1)
    exploration_corridor_width: float = Field(default=0.15, gt=0.0, lt=1.0)
    feature_pipeline: FeaturePipelineConfig = Field(default_factory=FeaturePipelineConfig)
    feature_paths: FeaturePathsConfig = Field(default_factory=FeaturePathsConfig)
    glm_baseline: GLMBaselineConfig = Field(default_factory=GLMBaselineConfig)
    glm_paths: GLMPathsConfig = Field(default_factory=GLMPathsConfig)
    segments: SegmentsConfig = Field(default_factory=SegmentsConfig)
    segment_paths: SegmentPathsConfig = Field(default_factory=SegmentPathsConfig)

    @model_validator(mode="after")
    def _validate_segment_merges(self) -> "Tier1MLConfig":
        """Validates that every configured cell merge references frozen band labels."""
        labels = self.feature_pipeline
        merged_labels: set[str] = set()
        for merge in self.segments.cell_merges:
            checks = [
                ("driver_age_band", merge.driver_age_band, labels.driver_age_band_labels),
                ("car_age_band", merge.car_age_band, labels.car_age_band_labels),
                ("from_density_band", merge.from_density_band, labels.density_band_labels),
                ("to_density_band", merge.to_density_band, labels.density_band_labels),
            ]
            for name, value, allowed in checks:
                if value not in allowed:
                    raise ValueError(
                        f"cell merge {name} '{value}' is not a frozen band label"
                    )
            if merge.from_density_band == merge.to_density_band:
                raise ValueError("cell merge from/to density bands must differ")
            if merge.merged_density_band in labels.density_band_labels:
                raise ValueError(
                    f"merged density label '{merge.merged_density_band}' collides with a band label"
                )
            if merge.merged_density_band in merged_labels:
                raise ValueError(
                    f"duplicate merged density label '{merge.merged_density_band}'"
                )
            merged_labels.add(merge.merged_density_band)
        return self


class Tier2AgentsConfig(BaseModel):
    """Configuration for Tier 2 LangGraph agentic orchestration (Phase 4-5)."""

    model_config = ConfigDict(extra="forbid")

    groundedness_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    evidence_coverage_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    max_tool_calls: int = Field(default=5, gt=0)
    execution_timeout_seconds: int = Field(default=60, gt=0)
    briefing_timeout_seconds: int = Field(default=2, gt=0)


class GovernanceConfig(BaseModel):
    """Configuration for Tier 3 Governance, HITL escalation, and fallback (Phase 7)."""

    model_config = ConfigDict(extra="forbid")

    audit_storage_backend: str = Field(default="sqlite")
    audit_db_path: str = Field(default="artifacts/audit_log.db")
    auto_approval_corridor_pct: float = Field(default=0.05, ge=0.0, le=0.50)
    enforce_hitl_on_flag: bool = Field(default=True)


class DataContractsConfig(BaseModel):
    """Configuration for Great Expectations data contracts and fixtures (Phase 1 & Phase 2)."""

    model_config = ConfigDict(extra="forbid")

    raw_policy_suite_path: str = Field(
        default="data_contracts/raw_policy_suite.json",
        description="Path to raw policy GX JSON suite",
    )
    raw_intake_manifest_path: str = Field(
        default="data_contracts/raw_intake_manifest.json",
        description="Path to raw intake checksum manifest",
    )
    raw_policy_valid_fixture: str = Field(
        default="data_contracts/fixtures/raw_policy_valid.csv",
        description="Path to valid raw policy fixture",
    )
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
