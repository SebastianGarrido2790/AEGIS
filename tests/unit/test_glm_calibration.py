"""Unit tests for the Tweedie GLM calibration evaluation (P2-F03)."""

from typing import Any, cast

import pandas as pd
import pytest
from statsmodels.genmod.families import Tweedie
from statsmodels.genmod.families.links import Log

from aegis.pipelines.glm_calibration import (
    decile_ae,
    decile_labels,
    mean_tweedie_deviance,
    secondary_diagnostics,
    segment_ae,
)


def _family() -> Tweedie:
    return Tweedie(var_power=1.5, link=Log())


def test_mean_tweedie_deviance_matches_family() -> None:
    """Verifies mean deviance equals the exposure-weighted family deviance divided by n."""
    fam = _family()
    y = pd.Series([0.0, 10.0, 0.0, 5.0])
    mu = pd.Series([1.0, 8.0, 1.0, 6.0])
    w = pd.Series([0.5, 1.0, 0.3, 1.0])
    got = mean_tweedie_deviance(fam, y, mu, w)
    expected = float(fam.deviance(y, mu, freq_weights=cast(Any, w), scale=1.0)) / len(y)
    assert got == pytest.approx(expected)


def test_decile_labels_partition() -> None:
    """Verifies decile labels cover 1..k and partition the rows."""
    mu = pd.Series([float(i) for i in range(1, 13)])
    labels = decile_labels(mu, 4)
    assert set(labels) == {1, 2, 3, 4}
    assert labels.nunique() == 4


def test_decile_ae_is_ratio_of_sums() -> None:
    """Verifies decile A/E uses ratio-of-sums semantics."""
    mu = pd.Series([1.0, 2.0, 3.0, 4.0])
    w = pd.Series([1.0, 1.0, 1.0, 1.0])
    actual = pd.Series([1.0, 4.0, 9.0, 16.0])
    rows = decile_ae(mu, w, actual, 2)
    assert len(rows) == 2
    assert rows[0].group == "decile_1"
    assert rows[0].actual == pytest.approx(5.0)
    assert rows[0].expected == pytest.approx(3.0)
    assert rows[0].ae_ratio == pytest.approx(5.0 / 3.0)


def test_segment_ae_covers_bands_and_cells() -> None:
    """Verifies segment A/E emits one-way bands and the cell cross-product."""
    view = pd.DataFrame(
        {
            "driver_age_band": ["18-24"] * 4 + ["25-44"] * 4,
            "car_age_band": ["0-4"] * 8,
            "density_band": ["<=500"] * 4 + [">2000"] * 4,
        }
    )
    mu = pd.Series([1.0] * 8)
    w = pd.Series([1.0] * 8)
    actual = pd.Series([1.0] * 8)
    rows = segment_ae(view, mu, w, actual)
    groups = {row.group for row in rows}
    assert "driver_age_band:18-24" in groups
    assert "driver_age_band:25-44" in groups
    assert "car_age_band:0-4" in groups
    assert "density_band:<=500" in groups
    assert "density_band:>2000" in groups
    assert any(group.startswith("cell:") for group in groups)
    one_way = [group for group in groups if not group.startswith("cell:")]
    assert len(one_way) == 5


def test_secondary_diagnostics_values() -> None:
    """Verifies MAE and RMSE are computed correctly."""
    actual = pd.Series([2.0, 4.0, 6.0])
    expected = pd.Series([1.0, 4.0, 7.0])
    diag = secondary_diagnostics(actual, expected)
    assert diag.mae == pytest.approx(2.0 / 3.0)
    assert diag.rmse == pytest.approx((2.0 / 3.0) ** 0.5)
