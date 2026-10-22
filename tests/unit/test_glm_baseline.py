"""Unit and integration tests for the Tweedie GLM baseline (P2-F03)."""

import math
import random
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.pipelines import glm_baseline as gb
from aegis.schemas.glm import GLMBaselineMetadata, GLMCalibrationReport
from aegis.utils.exceptions import GLMBaselineError


def _feature_levels() -> tuple[list[str], ...]:
    fp = get_config().tier1_ml.feature_pipeline
    return (
        fp.power_levels,
        fp.brand_levels,
        fp.gas_levels,
        fp.region_levels,
        fp.driver_age_band_labels,
        fp.car_age_band_labels,
        fp.density_band_labels,
    )


_DENSITY_BY_BAND = {"<=500": 300.0, "501-2000": 1000.0, ">2000": 3000.0}
_DRIVER_AGE_BY_BAND = {"18-24": 21, "25-44": 35, "45-64": 55, "65+": 70}
_CAR_AGE_BY_BAND = {"0-4": 2, "5-9": 7, "10+": 15}


def make_view(seed: int = 0) -> pd.DataFrame:
    """Builds a full Cartesian-product GLM view so every factor level is observed."""
    power, brand, gas, region, da, ca, de = _feature_levels()
    rng = random.Random(seed)
    rows: list[dict] = []
    for idx, (p, b, g, r, a, c, d) in enumerate(
        product(power, brand, gas, region, da, ca, de), start=1
    ):
        exposure = round(rng.uniform(0.2, 1.0), 6)
        claim = round(rng.uniform(200.0, 12000.0) * exposure, 2) if rng.random() < 0.08 else 0.0
        density = _DENSITY_BY_BAND[d]
        rows.append(
            {
                "PolicyID": idx,
                "exposure_capped": exposure,
                "ClaimAmount": claim,
                "DriverAge": _DRIVER_AGE_BY_BAND[a],
                "CarAge": _CAR_AGE_BY_BAND[c],
                "Density": density,
                "log_density": math.log(density),
                "driver_age_band": a,
                "car_age_band": c,
                "density_band": d,
                "Power": p,
                "Brand": b,
                "Gas": g,
                "Region": r,
            }
        )
    return pd.DataFrame(rows)


def _minimal_view() -> pd.DataFrame:
    """Builds a small, structurally valid GLM view for rule-isolation tests."""
    power, brand, gas, region, da, ca, de = _feature_levels()
    return pd.DataFrame(
        {
            "PolicyID": [1, 2, 3],
            "exposure_capped": [0.5, 1.0, 0.8],
            "ClaimAmount": [0.0, 1000.0, 250.0],
            "DriverAge": [21, 35, 55],
            "CarAge": [2, 7, 15],
            "Density": [300.0, 1000.0, 3000.0],
            "log_density": [math.log(300.0), math.log(1000.0), math.log(3000.0)],
            "driver_age_band": [da[0], da[1], da[2]],
            "car_age_band": [ca[0], ca[1], ca[2]],
            "density_band": [de[0], de[1], de[2]],
            "Power": [power[0], power[1], power[2]],
            "Brand": [brand[0], brand[1], brand[2]],
            "Gas": [gas[0], gas[1], gas[0]],
            "Region": [region[0], region[1], region[2]],
        }
    )


def _small_dgp(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Builds a small Tweedie-like response on a two-column numeric design."""
    rng = random.Random(seed)
    x1 = [rng.gauss(0.0, 1.0) for _ in range(n)]
    x2 = [rng.gauss(0.0, 1.0) for _ in range(n)]
    w = pd.Series([rng.uniform(0.2, 1.0) for _ in range(n)])
    y_values: list[float] = []
    for i in range(n):
        mu = math.exp(0.4 + 0.3 * x1[i] - 0.2 * x2[i])
        if rng.random() < 0.05:
            y_values.append(round(random.gammavariate(mu, 1.0) * w.iloc[i], 6))
        else:
            y_values.append(0.0)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    X.insert(0, "const", 1.0)
    return X, pd.Series(y_values), w


def test_validate_glm_input_missing_column_raises() -> None:
    """Verifies a missing required column raises a typed error."""
    view = _minimal_view().drop(columns=["log_density"])
    with pytest.raises(GLMBaselineError) as exc_info:
        gb.validate_glm_input(view)
    assert "missing required columns" in str(exc_info.value)


def test_validate_glm_input_non_positive_exposure_raises() -> None:
    """Verifies a non-positive exposure raises a typed error."""
    view = _minimal_view()
    view.loc[0, "exposure_capped"] = 0.0
    with pytest.raises(GLMBaselineError) as exc_info:
        gb.validate_glm_input(view)
    assert "exposure_capped" in str(exc_info.value)


def test_validate_glm_input_negative_claim_raises() -> None:
    """Verifies a negative claim amount raises a typed error."""
    view = _minimal_view()
    view.loc[0, "ClaimAmount"] = -1.0
    with pytest.raises(GLMBaselineError) as exc_info:
        gb.validate_glm_input(view)
    assert "ClaimAmount" in str(exc_info.value)


def test_build_target_caps_and_divides() -> None:
    """Verifies target = min(ClaimAmount, cap) / exposure_capped."""
    view = _minimal_view()
    target = gb.build_target(view, 500.0)
    assert target.iloc[1] == pytest.approx(500.0 / 1.0)
    assert target.iloc[2] == pytest.approx(250.0 / 0.8)


def test_build_design_matrix_columns() -> None:
    """Verifies the design matrix is reference-coded with no leakage columns."""
    config = get_config()
    design = gb.build_design_matrix(_minimal_view(), config.tier1_ml)
    columns = list(design.columns)
    assert columns[0] == "const"
    assert len(columns) == 35
    forbidden = {
        "PolicyID",
        "exposure_capped",
        "ClaimAmount",
        "log_density",
        "DriverAge",
        "CarAge",
        "Density",
    }
    assert not forbidden.intersection(columns)
    assert "Power__e" in columns
    assert "driver_age_band__25-44" in columns


def test_split_indices_deterministic_and_ratio() -> None:
    """Verifies the seeded split is 80/20 and deterministic."""
    first = gb.split_indices(100, 42, 0.2)
    second = gb.split_indices(100, 42, 0.2)
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() == second[1].tolist()
    assert len(first[0]) == 80
    assert len(first[1]) == 20


def test_select_tweedie_p_returns_interior_p() -> None:
    """Verifies profile deviance returns an interior power without fallback."""
    X, y, w = _small_dgp()
    cfg = get_config().tier1_ml.glm_baseline
    selected, profile, fallback = gb.select_tweedie_p(X, y, w, cfg)
    assert 1.0 < selected < 2.0
    assert fallback is False
    assert any(pt.converged for pt in profile)


def test_select_tweedie_p_falls_back_when_no_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the documented fallback is used when no grid point converges."""
    X, y, w = _small_dgp(n=500)

    def _boom(*args, **kwargs):
        raise ValueError("fit failure")

    monkeypatch.setattr(gb, "fit_glm", _boom)
    cfg = get_config().tier1_ml.glm_baseline
    selected, profile, fallback = gb.select_tweedie_p(X, y, w, cfg)
    assert fallback is True
    assert selected == cfg.tweedie_p_fallback
    assert all(not pt.converged for pt in profile)


def test_fit_glm_converges() -> None:
    """Verifies a Tweedie fit converges on synthetic data."""
    X, y, w = _small_dgp()
    result = gb.fit_glm(X, y, w, 1.5)
    assert result.converged is True
    assert all(math.isfinite(float(value)) for value in result.params)


def test_schemas_reject_extra_fields() -> None:
    """Verifies metadata and report forbid unexpected fields."""
    base = {
        "cap": 100000.0,
        "tweedie_p": 1.5,
        "fallback": False,
        "profile": [],
        "coefficients": {"const": -3.0},
        "dispersion": 1.0,
        "converged": True,
        "train_rows": 10,
        "test_rows": 2,
        "random_state": 42,
        "config_fingerprint": "a" * 64,
    }
    with pytest.raises(ValidationError):
        GLMBaselineMetadata.model_validate({**base, "extra": 1})
    report = {
        "cap": 100000.0,
        "tweedie_p": 1.5,
        "holdout_mean_deviance": 1.0,
        "decile_ae": [],
        "segment_ae": [],
        "secondary": {"mae": 1.0, "rmse": 1.0},
        "train_rows": 10,
        "test_rows": 2,
        "random_state": 42,
    }
    with pytest.raises(ValidationError):
        GLMCalibrationReport.model_validate({**report, "extra": 1})


def test_no_frequency_or_severity_model() -> None:
    """Verifies no statsmodels Poisson or Gamma family is used in the fit path."""
    source = Path(gb.__file__).read_text(encoding="utf-8")
    assert "sm.Poisson" not in source
    assert "sm.Gamma" not in source
    assert "Tweedie" in source


def test_end_to_end_writes_artifacts_determinism_and_cap(tmp_path: Path) -> None:
    """Runs the pipeline twice at one cap and checks artifacts and determinism."""
    view = make_view()
    source = tmp_path / "glm.csv"
    view.to_csv(source, index=False)

    paths1 = {
        "fitted_path": tmp_path / "run1" / "fitted.csv",
        "metadata_path": tmp_path / "run1" / "metadata.json",
        "report_path": tmp_path / "run1" / "report.json",
    }
    paths2 = {
        "fitted_path": tmp_path / "run2" / "fitted.csv",
        "metadata_path": tmp_path / "run2" / "metadata.json",
        "report_path": tmp_path / "run2" / "report.json",
    }

    report1 = gb.run_glm_baseline(input_path=source, cap=5000.0, **paths1)
    report2 = gb.run_glm_baseline(input_path=source, cap=5000.0, **paths2)

    assert report1.cap == 5000.0
    assert report1.tweedie_p == report2.tweedie_p
    assert report1.decile_ae and report2.decile_ae
    assert report1.segment_ae and report2.segment_ae

    for key in ("fitted_path", "metadata_path", "report_path"):
        assert paths1[key].read_bytes() == paths2[key].read_bytes()

    metadata = GLMBaselineMetadata.model_validate_json(
        paths1["metadata_path"].read_text(encoding="utf-8")
    )
    assert metadata.converged is True
    assert metadata.cap == 5000.0
    assert len(metadata.coefficients) == 35

    fitted = pd.read_csv(paths1["fitted_path"])
    assert list(fitted.columns) == [
        "PolicyID",
        "exposure_capped",
        "capped_actual_loss",
        "fitted_loss_cost_per_year",
        "split",
        "decile",
    ]
    assert fitted["capped_actual_loss"].max() <= 5000.0 + 1e-9
    assert set(fitted["split"]) == {"train", "test"}


def test_non_convergence_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a non-converged final fit raises instead of writing a report."""

    class FakeResult:
        converged = False
        deviance = 100.0
        scale = 1.0
        params = pd.Series([0.0], index=["const"])

        def predict(self, exog):
            return pd.Series([1.0] * len(exog), index=exog.index)

    monkeypatch.setattr(gb, "fit_glm", lambda *a, **k: FakeResult())

    view = _minimal_view()
    source = tmp_path / "glm.csv"
    view.to_csv(source, index=False)

    with pytest.raises(GLMBaselineError) as exc_info:
        gb.run_glm_baseline(input_path=source, fitted_path=tmp_path / "f.csv")
    assert "non_convergence" in str(exc_info.value)


def test_empirical_glm_baseline(tmp_path: Path) -> None:
    """Runs the full baseline on the empirical GLM view when present."""
    source = Path("data/processed/features_glm.csv")
    if not source.is_file():
        pytest.skip("Processed GLM view not present; run dvc repro first.")

    report = gb.run_glm_baseline(
        input_path=source,
        fitted_path=tmp_path / "fitted.csv",
        metadata_path=tmp_path / "metadata.json",
        report_path=tmp_path / "report.json",
    )
    assert report.train_rows + report.test_rows == 413_169
    assert report.test_rows > 0
    assert report.holdout_mean_deviance >= 0
    assert len(report.decile_ae) == get_config().tier1_ml.glm_baseline.decile_count
    assert report.segment_ae
