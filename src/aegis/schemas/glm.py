"""Typed contracts for the Tweedie GLM baseline and calibration report (P2-F03, ADR-012).

These models define the persisted metadata and report interfaces consumed by Stage 4
segment aggregation and Stage 5 cap sweeping, and enforce schema stability on reload.
"""

from pydantic import BaseModel, ConfigDict, Field

from aegis.schemas.features import SHA256_PATTERN


class ProfilePoint(BaseModel):
    """One variance-power grid point and its fitted deviance and convergence status."""

    model_config = ConfigDict(extra="forbid")

    p: float = Field(gt=1.0, lt=2.0)
    deviance: float | None = None
    converged: bool


class GLMBaselineMetadata(BaseModel):
    """Versioned metadata artifact for a fitted Tweedie baseline."""

    model_config = ConfigDict(extra="forbid")

    cap: float = Field(gt=0.0)
    tweedie_p: float = Field(gt=1.0, lt=2.0)
    fallback: bool
    profile: list[ProfilePoint]
    coefficients: dict[str, float]
    dispersion: float = Field(gt=0.0)
    converged: bool
    train_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    random_state: int
    config_fingerprint: str = Field(pattern=SHA256_PATTERN)


class AERow(BaseModel):
    """One actual-versus-expected ratio-of-sums row for a decile or segment band."""

    model_config = ConfigDict(extra="forbid")

    group: str
    actual: float = Field(ge=0.0)
    expected: float = Field(ge=0.0)
    ae_ratio: float
    count: int = Field(ge=0)
    exposure: float = Field(ge=0.0)


class SecondaryDiagnostics(BaseModel):
    """Reference-only regression diagnostics, never used as primary gates (P2-D1)."""

    model_config = ConfigDict(extra="forbid")

    mae: float
    rmse: float


class GLMCalibrationReport(BaseModel):
    """Versioned calibration report for a fitted Tweedie baseline."""

    model_config = ConfigDict(extra="forbid")

    cap: float = Field(gt=0.0)
    tweedie_p: float = Field(gt=1.0, lt=2.0)
    holdout_mean_deviance: float = Field(ge=0.0)
    decile_ae: list[AERow]
    segment_ae: list[AERow]
    secondary: SecondaryDiagnostics
    train_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    random_state: int
