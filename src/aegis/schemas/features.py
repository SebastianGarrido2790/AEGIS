"""Typed contracts for the deterministic feature pipeline (P2-F02, ADR-012).

These models define the persisted metadata interface consumed by later stages and
enforce schema stability when the metadata JSON is reloaded.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class TailClipCounts(BaseModel):
    """Counts of values clipped at the approved DriverAge and CarAge bounds."""

    model_config = ConfigDict(extra="forbid")

    driver_age_below: int = Field(ge=0)
    driver_age_above: int = Field(ge=0)
    car_age_below: int = Field(ge=0)
    car_age_above: int = Field(ge=0)


class BandSpec(BaseModel):
    """Frozen band edges and labels for one derived factor."""

    model_config = ConfigDict(extra="forbid")

    edges: list[float] = Field(min_length=2)
    labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_edges(self) -> "BandSpec":
        """Validates strict edge monotonicity and label alignment."""
        if len(self.edges) != len(self.labels) + 1:
            raise ValueError("band edges must equal labels + 1")
        if any(left >= right for left, right in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("band edges must be strictly increasing")
        return self


class NumericSummary(BaseModel):
    """Descriptive summary for one numeric column."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p25: float
    p50: float
    p75: float
    p99: float


class CategoricalSummary(BaseModel):
    """Descriptive summary for one categorical column."""

    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int]
    missing: int = Field(ge=0)


class DistributionSummary(BaseModel):
    """Baseline feature-distribution summary persisted with the views."""

    model_config = ConfigDict(extra="forbid")

    numeric: dict[str, NumericSummary]
    categorical: dict[str, CategoricalSummary]


class ViewContract(BaseModel):
    """Column contract and byte hash for one persisted view."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    row_count: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class FeatureMetadata(BaseModel):
    """Versioned metadata artifact for the dual-view feature pipeline."""

    model_config = ConfigDict(extra="forbid")

    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    config_fingerprint: str = Field(pattern=SHA256_PATTERN)
    exposure_clip_count: int = Field(ge=0)
    tail_clip_counts: TailClipCounts
    band_edges: dict[str, BandSpec]
    level_mappings: dict[str, list[str]]
    one_hot_columns: list[str]
    forest_covariates: list[str]
    glm_view: ViewContract
    forest_view: ViewContract
    distribution_summary: DistributionSummary
