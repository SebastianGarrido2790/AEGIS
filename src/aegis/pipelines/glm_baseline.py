"""Cap-parameterized Tweedie GLM baseline (P2-F03, ADR-012).

Fits a single compound Poisson-Gamma Tweedie GLM (log link, exposure weights) on the
capped pure premium of the seeded training split, selects the variance power by a
deterministic profile-deviance grid, and writes versioned model metadata, per-policy
fitted loss cost, and the calibration report. No separate frequency or severity model
exists; the Tweedie is the sanctioned combined model (P2-D1).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie
from statsmodels.genmod.families.links import Log

from aegis.config.loader import get_config
from aegis.config.schema import GLMBaselineConfig, Tier1MLConfig
from aegis.pipelines.glm_calibration import (
    CalibrationResult,
    decile_labels,
    evaluate_holdout,
)
from aegis.schemas.glm import GLMBaselineMetadata, GLMCalibrationReport, ProfilePoint
from aegis.utils.exceptions import GLMBaselineError

GLM_VIEW_COLUMNS: list[str] = [
    "PolicyID",
    "exposure_capped",
    "ClaimAmount",
    "DriverAge",
    "CarAge",
    "Density",
    "log_density",
    "driver_age_band",
    "car_age_band",
    "density_band",
    "Power",
    "Brand",
    "Gas",
    "Region",
]

GLM_FACTOR_COLUMNS: list[str] = [
    "Power",
    "Brand",
    "Gas",
    "Region",
    "driver_age_band",
    "car_age_band",
    "density_band",
]


class FittedGLM(Protocol):
    """Minimal interface for a fitted statsmodels GLM result (incomplete stubs)."""

    converged: bool
    deviance: float
    scale: float
    params: pd.Series

    def predict(self, exog: pd.DataFrame) -> pd.Series: ...


def _fail(
    message: str,
    rule: str,
    column: str | None = None,
    count: int = 0,
    sample_keys: list[Any] | None = None,
) -> None:
    """Raises a typed GLM-baseline error with structured details."""
    details: dict[str, Any] = {"rule": rule, "count": count}
    if column is not None:
        details["column"] = column
    if sample_keys:
        details["sample_keys"] = sample_keys[:10]
    raise GLMBaselineError(message, details=details)


def validate_glm_input(view: pd.DataFrame) -> None:
    """Validates the Stage 2 GLM view before fitting.

    Args:
        view: GLM view produced by the Stage 2 feature pipeline.

    Raises:
        GLMBaselineError: If columns, exposure, or loss values are invalid.
    """
    if view.empty:
        _fail("GLM view is empty; nothing to fit.", "empty_input")

    actual = list(view.columns)
    missing = [col for col in GLM_VIEW_COLUMNS if col not in actual]
    if missing:
        _fail(
            f"GLM view is missing required columns: {missing}",
            "missing_columns",
            count=len(missing),
            sample_keys=missing,
        )
    unexpected = [col for col in actual if col not in GLM_VIEW_COLUMNS]
    if unexpected:
        _fail(
            f"GLM view contains unexpected columns: {unexpected}",
            "unexpected_columns",
            count=len(unexpected),
            sample_keys=unexpected,
        )

    invalid_exposure = view["exposure_capped"] <= 0
    if bool(invalid_exposure.any()):
        keys = view.loc[invalid_exposure, "PolicyID"].tolist()
        _fail(
            "exposure_capped must be strictly positive.",
            "invalid_exposure",
            column="exposure_capped",
            count=int(invalid_exposure.sum()),
            sample_keys=keys,
        )

    invalid_loss = view["ClaimAmount"] < 0
    if bool(invalid_loss.any()):
        keys = view.loc[invalid_loss, "PolicyID"].tolist()
        _fail(
            "ClaimAmount must be non-negative.",
            "invalid_claim_amount",
            column="ClaimAmount",
            count=int(invalid_loss.sum()),
            sample_keys=keys,
        )


def build_design_matrix(view: pd.DataFrame, config: Tier1MLConfig) -> pd.DataFrame:
    """Builds the deterministic, reference-coded GLM design matrix.

    Args:
        view: Validated GLM view.
        config: Validated Tier 1 configuration.

    Returns:
        A design matrix with an intercept column followed by drop-first dummy columns
        for each of the seven rating factors, in a frozen column order.
    """
    fp = config.feature_pipeline
    factor_specs: list[tuple[str, list[str]]] = [
        ("Power", fp.power_levels),
        ("Brand", fp.brand_levels),
        ("Gas", fp.gas_levels),
        ("Region", fp.region_levels),
        ("driver_age_band", fp.driver_age_band_labels),
        ("car_age_band", fp.car_age_band_labels),
        ("density_band", fp.density_band_labels),
    ]
    parts: list[pd.DataFrame] = []
    for column, levels in factor_specs:
        categorical = pd.Categorical(view[column], categories=list(levels))
        dummies = pd.get_dummies(categorical, prefix=column, prefix_sep="__", drop_first=True)
        parts.append(dummies.astype(int))
    design = pd.concat(parts, axis=1)
    design.insert(0, "const", 1)
    return design


def build_target(view: pd.DataFrame, cap: float) -> pd.Series:
    """Builds the capped pure-premium target: min(ClaimAmount, cap) / exposure_capped."""
    capped_loss = view["ClaimAmount"].clip(upper=cap).astype(float)
    return capped_loss / view["exposure_capped"].astype(float)


def split_indices(
    n: int, random_state: int, test_size: float
) -> tuple[pd.Index, pd.Index]:
    """Splits positional indices into train/test with a fixed seed."""
    from sklearn.model_selection import train_test_split

    train_pos, test_pos = train_test_split(
        range(n), test_size=test_size, random_state=random_state, shuffle=True
    )
    return pd.Index(train_pos), pd.Index(test_pos)


def _p_grid(cfg: GLMBaselineConfig) -> list[float]:
    """Builds the deterministic coarse variance-power grid."""
    steps = int(round((cfg.tweedie_p_max - cfg.tweedie_p_min) / cfg.tweedie_p_step))
    return [round(cfg.tweedie_p_min + i * cfg.tweedie_p_step, 6) for i in range(steps + 1)]


def _refine(p: float, cfg: GLMBaselineConfig) -> list[float]:
    """Builds the deterministic refinement grid around a coarse minimizer."""
    grid = set(_p_grid(cfg))
    candidates: list[float] = []
    for offset in (-1, 1):
        candidate = round(p + offset * cfg.tweedie_p_refine_step, 6)
        if cfg.tweedie_p_min - 1e-9 <= candidate <= cfg.tweedie_p_max + 1e-9:
            candidates.append(candidate)
    return [candidate for candidate in candidates if candidate not in grid]


def fit_glm(
    X: pd.DataFrame, y: pd.Series, w: pd.Series, p: float
) -> FittedGLM:
    """Fits a Tweedie GLM with log link and exposure frequency weights."""
    family = Tweedie(var_power=p, link=Log())
    model = sm.GLM(endog=y, exog=X, family=family, freq_weights=w)
    return cast(FittedGLM, model.fit())


def _fit_point(X: pd.DataFrame, y: pd.Series, w: pd.Series, p: float) -> ProfilePoint:
    """Fits one grid point, recording convergence status and deviance."""
    try:
        result = fit_glm(X, y, w, p)
        return ProfilePoint(
            p=round(p, 6),
            deviance=round(float(result.deviance), 6),
            converged=bool(result.converged),
        )
    except Exception:
        return ProfilePoint(p=round(p, 6), deviance=None, converged=False)


def select_tweedie_p(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    cfg: GLMBaselineConfig,
) -> tuple[float, list[ProfilePoint], bool]:
    """Selects the Tweedie variance power by profile deviance.

    Args:
        X: Training design matrix.
        y: Training target.
        w: Training exposure weights.
        cfg: Validated GLM baseline configuration.

    Returns:
        A tuple of the selected power, the full profile table, and whether the
        documented fallback value was used (no converged grid point).
    """
    profile = [_fit_point(X, y, w, p) for p in _p_grid(cfg)]
    converged = [pt for pt in profile if pt.converged and pt.deviance is not None]
    if not converged:
        return cfg.tweedie_p_fallback, profile, True

    best = min(converged, key=lambda pt: cast(float, pt.deviance))
    refined = [_fit_point(X, y, w, p) for p in _refine(best.p, cfg)]
    profile.extend(refined)
    refined_converged = [pt for pt in refined if pt.converged and pt.deviance is not None]
    if refined_converged:
        best = min(refined_converged, key=lambda pt: cast(float, pt.deviance))
    return round(best.p, 6), profile, False


def _config_fingerprint(cfg: GLMBaselineConfig) -> str:
    """Hashes the canonical GLM-baseline configuration."""
    payload = cfg.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(model: Any, path: Path) -> None:
    """Writes a validated Pydantic model as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def run_glm_baseline(
    input_path: Path | str | None = None,
    cap: float | None = None,
    fitted_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> GLMCalibrationReport:
    """Runs the full Tweedie baseline fit, calibration, and artifact writing.

    Args:
        input_path: Stage 2 GLM view CSV. Defaults to params.yaml feature_paths.glm_view.
        cap: Loss cap level. Defaults to params.yaml glm_baseline.default_cap.
        fitted_path: Fitted loss-cost CSV. Defaults to params.yaml.
        metadata_path: Model metadata JSON. Defaults to params.yaml.
        report_path: Calibration report JSON. Defaults to params.yaml.

    Returns:
        The validated calibration report that was written.

    Raises:
        GLMBaselineError: If input is invalid, the fit fails to converge, or writing fails.
    """
    config = get_config()
    glm_cfg = config.tier1_ml.glm_baseline
    paths = config.tier1_ml.glm_paths

    cap_value = float(cap) if cap is not None else glm_cfg.default_cap
    if cap_value <= 0:
        _fail(f"Loss cap must be positive, got {cap_value}.", "invalid_cap", count=1)

    source = (
        Path(input_path)
        if input_path is not None
        else Path(config.tier1_ml.feature_paths.glm_view)
    )
    if not source.is_file():
        _fail(f"GLM view input not found: {source}", "input_not_found", column="input")

    view = pd.read_csv(source)
    validate_glm_input(view)

    design = build_design_matrix(view, config.tier1_ml)
    target = build_target(view, cap_value)
    weights = view["exposure_capped"].astype(float)

    train_pos, test_pos = split_indices(
        len(view), config.tier1_ml.random_state, config.tier1_ml.test_size
    )

    X_train = design.iloc[train_pos].reset_index(drop=True)
    y_train = target.iloc[train_pos].reset_index(drop=True)
    y_test = target.iloc[test_pos].reset_index(drop=True)
    w_train = weights.iloc[train_pos].reset_index(drop=True)
    w_test = weights.iloc[test_pos].reset_index(drop=True)
    view_test = view.iloc[test_pos].reset_index(drop=True)

    selected_p, profile, fallback = select_tweedie_p(X_train, y_train, w_train, glm_cfg)

    result = fit_glm(X_train, y_train, w_train, selected_p)
    if not bool(result.converged):
        _fail(
            "Tweedie GLM did not converge at the selected variance power.",
            "non_convergence",
            count=1,
        )

    mu_all = pd.Series(result.predict(design), index=design.index, dtype=float)
    mu_test = mu_all.iloc[test_pos].reset_index(drop=True)

    capped_actual = view["ClaimAmount"].clip(upper=cap_value).astype(float)
    actual_test = capped_actual.iloc[test_pos].reset_index(drop=True)

    family = Tweedie(var_power=selected_p, link=Log())
    calibration: CalibrationResult = evaluate_holdout(
        family,
        view_test,
        y_test,
        mu_test,
        w_test,
        actual_test,
        glm_cfg.decile_count,
    )

    decile = pd.Series("", index=view.index, dtype=object)
    test_labels = decile_labels(mu_test, glm_cfg.decile_count).astype(str)
    decile.iloc[test_pos] = test_labels.values

    split = pd.Series("train", index=view.index, dtype=object)
    split.iloc[test_pos] = "test"

    fitted = pd.DataFrame(
        {
            "PolicyID": view["PolicyID"],
            "exposure_capped": weights,
            "capped_actual_loss": capped_actual,
            "fitted_loss_cost_per_year": mu_all.round(6),
            "split": split,
            "decile": decile,
        }
    )
    fitted_target = Path(fitted_path or paths.fitted_loss_cost)
    fitted_target.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(fitted_target, index=False, lineterminator="\n")

    coefficients = {str(key): float(value) for key, value in result.params.items()}
    metadata = GLMBaselineMetadata(
        cap=cap_value,
        tweedie_p=selected_p,
        fallback=fallback,
        profile=profile,
        coefficients=coefficients,
        dispersion=round(float(result.scale), 6),
        converged=bool(result.converged),
        train_rows=int(len(train_pos)),
        test_rows=int(len(test_pos)),
        random_state=config.tier1_ml.random_state,
        config_fingerprint=_config_fingerprint(glm_cfg),
    )
    _write_json(metadata, Path(metadata_path or paths.model_metadata))

    report = GLMCalibrationReport(
        cap=cap_value,
        tweedie_p=selected_p,
        holdout_mean_deviance=round(calibration.holdout_mean_deviance, 6),
        decile_ae=calibration.decile_ae,
        segment_ae=calibration.segment_ae,
        secondary=calibration.secondary,
        train_rows=int(len(train_pos)),
        test_rows=int(len(test_pos)),
        random_state=config.tier1_ml.random_state,
    )
    _write_json(report, Path(report_path or paths.calibration_report))

    print(
        f"[GLM] cap={cap_value:,.0f} p={selected_p} fallback={fallback} "
        f"converged={result.converged} train={len(train_pos):,} test={len(test_pos):,} "
        f"holdout_deviance={report.holdout_mean_deviance:.4f}"
    )
    return report
