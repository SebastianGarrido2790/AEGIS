"""Calibration evaluation for the Tweedie GLM baseline (P2-F03, ADR-012).

Computes holdout mean Tweedie deviance, actual-versus-expected ratios by predicted-loss
decile and by segment band, and secondary MAE/RMSE diagnostics. All metrics are
deterministic and computed from fitted values; MAE/RMSE are never used as primary gates.
"""

from dataclasses import dataclass

import pandas as pd

from aegis.schemas.glm import AERow, SecondaryDiagnostics

SEGMENT_BAND_COLUMNS: list[str] = ["driver_age_band", "car_age_band", "density_band"]


@dataclass(frozen=True)
class CalibrationResult:
    """Holdout calibration outputs for one fitted baseline."""

    holdout_mean_deviance: float
    decile_ae: list[AERow]
    segment_ae: list[AERow]
    secondary: SecondaryDiagnostics


def mean_tweedie_deviance(family, y: pd.Series, mu: pd.Series, w: pd.Series) -> float:
    """Returns the out-of-sample mean Tweedie deviance.

    The exposure-weighted summed unit deviance (scale=1) divided by the number of
    policies. Deterministic for a fixed fit and fixed test split.
    """
    total = float(family.deviance(y, mu, freq_weights=w, scale=1.0))
    return total / len(y)


def decile_labels(mu: pd.Series, decile_count: int) -> pd.Series:
    """Assigns deterministic decile labels 1..decile_count by predicted pure premium."""
    order = mu.rank(method="first", ascending=True)
    decile = ((order - 1) * decile_count // len(mu)) + 1
    return decile.clip(upper=decile_count).astype(int)


def _ae_row(group: str, actual: float, expected: float, count: int, exposure: float) -> AERow:
    """Builds one A/E row with ratio-of-sums semantics, guarding against zero expected."""
    ratio = actual / expected if expected > 0 else 0.0
    return AERow(
        group=group,
        actual=round(float(actual), 6),
        expected=round(float(expected), 6),
        ae_ratio=round(ratio, 6),
        count=int(count),
        exposure=round(float(exposure), 6),
    )


def decile_ae(mu: pd.Series, w: pd.Series, actual: pd.Series, decile_count: int) -> list[AERow]:
    """Computes ratio-of-sums A/E per predicted-loss decile."""
    labels = decile_labels(mu, decile_count)
    expected_loss = mu * w
    rows: list[AERow] = []
    for decile in range(1, decile_count + 1):
        mask = labels == decile
        rows.append(
            _ae_row(
                f"decile_{decile}",
                float(actual[mask].sum()),
                float(expected_loss[mask].sum()),
                int(mask.sum()),
                float(w[mask].sum()),
            )
        )
    return rows


def segment_ae(
    view: pd.DataFrame, mu: pd.Series, w: pd.Series, actual: pd.Series
) -> list[AERow]:
    """Computes ratio-of-sums A/E per band dimension and the 36-cell cross-product."""
    expected_loss = mu * w
    rows: list[AERow] = []

    for column in SEGMENT_BAND_COLUMNS:
        for value, group_mask in view.groupby(column, sort=True).groups.items():
            mask = pd.Series(False, index=view.index)
            mask.loc[group_mask] = True
            rows.append(
                _ae_row(
                    f"{column}:{value}",
                    float(actual[mask].sum()),
                    float(expected_loss[mask].sum()),
                    int(mask.sum()),
                    float(w[mask].sum()),
                )
            )

    cell_key = (
        view["driver_age_band"].astype(str)
        + "|"
        + view["car_age_band"].astype(str)
        + "|"
        + view["density_band"].astype(str)
    )
    for key, group_mask in view.groupby(cell_key, sort=True).groups.items():
        mask = pd.Series(False, index=view.index)
        mask.loc[group_mask] = True
        rows.append(
            _ae_row(
                f"cell:{key}",
                float(actual[mask].sum()),
                float(expected_loss[mask].sum()),
                int(mask.sum()),
                float(w[mask].sum()),
            )
        )

    return rows


def secondary_diagnostics(actual: pd.Series, expected_loss: pd.Series) -> SecondaryDiagnostics:
    """Computes reference-only MAE and RMSE on per-policy total loss."""
    error = actual - expected_loss
    mae = float(error.abs().mean())
    rmse = float((error**2).mean()) ** 0.5
    return SecondaryDiagnostics(mae=round(mae, 6), rmse=round(rmse, 6))


def evaluate_holdout(
    family,
    view: pd.DataFrame,
    y: pd.Series,
    mu: pd.Series,
    w: pd.Series,
    actual: pd.Series,
    decile_count: int,
) -> CalibrationResult:
    """Computes all holdout calibration metrics for a fitted baseline."""
    return CalibrationResult(
        holdout_mean_deviance=mean_tweedie_deviance(family, y, mu, w),
        decile_ae=decile_ae(mu, w, actual, decile_count),
        segment_ae=segment_ae(view, mu, w, actual),
        secondary=secondary_diagnostics(actual, mu * w),
    )
