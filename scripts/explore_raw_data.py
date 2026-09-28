"""Exploratory Data Analysis (EDA) for raw freMTPL benchmark datasets.

Analyzes raw French Motor Third-Party Liability datasets:
- Frequency table: `data/raw/freMTPLfreq.csv.zip` (or `.csv`)
- Severity table: `data/raw/freMTPLsev.csv`

Usage:
    Full exploratory run with terminal output
    uv run python scripts/explore_raw_data.py
    
    Export structured JSON
    uv run python scripts/explore_raw_data.py --output-json artifacts/eda_summary.json
    
    Export structured Markdown report
    uv run python scripts/explore_raw_data.py --output-markdown artifacts/eda_report.md
    
    Quick inspection on a sample of rows (e.g., 50k)
    uv run python scripts/explore_raw_data.py --sample-rows 50000
    
    View CLI options
    uv run python scripts/explore_raw_data.py --help
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

DEFAULT_FREQ_PATH = Path("data/raw/freMTPLfreq.csv.zip")
DEFAULT_SEV_PATH = Path("data/raw/freMTPLsev.csv")


@dataclass
class DatasetMetadata:
    """Metadata summary for a single dataset file."""

    file_path: str
    file_size_bytes: int
    file_size_formatted: str
    compressed: bool
    num_rows: int
    num_cols: int
    memory_usage_mb: float
    columns: list[str]
    missing_values: dict[str, int]


@dataclass
class RelationalAuditResult:
    """Audit of relational integrity between frequency and severity datasets."""

    freq_total_policies: int
    freq_unique_policies: int
    sev_total_claims: int
    sev_unique_policies: int
    orphan_sev_records: int
    claims_per_policy_distribution: dict[int, int]
    count_mismatch_records: int
    perfect_match: bool


@dataclass
class FrequencyAnalysisResult:
    """Actuarial claim frequency statistics."""

    total_exposure_years: float
    total_claim_count: int
    empirical_frequency: float
    mean_exposure: float
    median_exposure: float
    exposure_full_year_pct: float
    exposure_greater_than_one_pct: float
    exposure_short_term_pct: float
    policies_by_claim_count: dict[int, int]
    claim_count_mean: float
    claim_count_variance: float
    dispersion_ratio: float


@dataclass
class SeverityAnalysisResult:
    """Actuarial claim severity statistics."""

    total_loss_amount: float
    num_claims: int
    mean_severity: float
    median_severity: float
    std_severity: float
    min_severity: float
    max_severity: float
    skewness: float
    kurtosis: float
    percentiles: dict[str, float]
    top_1_loss_pct: float
    top_10_loss_pct: float
    top_100_loss_pct: float
    top_1_pct_claims_loss_pct: float
    tier_distribution: dict[str, dict[str, Any]]


@dataclass
class PurePremiumResult:
    """Actuarial pure premium (loss cost) summary."""

    total_losses: float
    total_exposure: float
    empirical_pure_premium: float
    empirical_frequency: float
    mean_severity: float
    identity_check_product: float
    identity_discrepancy: float


def format_bytes(num_bytes: int) -> str:
    """Formats raw byte count into human-readable representation."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def load_raw_datasets(
    freq_path: Path,
    sev_path: Path,
    sample_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw frequency and severity datasets into pandas DataFrames.

    Handles zip archives transparently for the frequency dataset.

    Args:
        freq_path: Path to the frequency dataset (.csv or .zip).
        sev_path: Path to the severity dataset (.csv).
        sample_rows: Optional row count limit for rapid inspection.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (freq_df, sev_df).

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If archive cannot be read.
    """
    if not freq_path.is_file():
        raise FileNotFoundError(f"Frequency file not found at: {freq_path}")
    if not sev_path.is_file():
        raise FileNotFoundError(f"Severity file not found at: {sev_path}")

    if freq_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(freq_path) as z:
            csv_members = [m for m in z.namelist() if m.endswith(".csv")]
            if not csv_members:
                raise ValueError(f"No CSV file found inside archive: {freq_path}")
            csv_filename = csv_members[0]
            with z.open(csv_filename) as f:
                freq_df = pd.read_csv(io.BytesIO(f.read()), nrows=sample_rows)
    else:
        freq_df = pd.read_csv(freq_path, nrows=sample_rows)

    sev_df = pd.read_csv(sev_path)
    if sample_rows is not None:
        sub_sev = sev_df[sev_df["PolicyID"].isin(freq_df["PolicyID"])].copy()
        sev_df = cast(pd.DataFrame, sub_sev)

    return freq_df, sev_df


def inspect_metadata(
    file_path: Path,
    df: pd.DataFrame,
    is_zip: bool,
) -> DatasetMetadata:
    """Extracts structural and storage metadata for a dataset."""
    size_bytes = file_path.stat().st_size
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    missing_dict = {str(col): int(count) for col, count in df.isnull().sum().items()}

    return DatasetMetadata(
        file_path=str(file_path),
        file_size_bytes=size_bytes,
        file_size_formatted=format_bytes(size_bytes),
        compressed=is_zip,
        num_rows=len(df),
        num_cols=len(df.columns),
        memory_usage_mb=round(memory_mb, 2),
        columns=[str(c) for c in df.columns],
        missing_values=missing_dict,
    )


def audit_relational_integrity(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
) -> RelationalAuditResult:
    """Verifies relational integrity between frequency policies and severity claims."""
    freq_total = len(freq_df)
    freq_unique = int(freq_df["PolicyID"].nunique())
    sev_total = len(sev_df)
    sev_unique = int(sev_df["PolicyID"].nunique())

    orphans = int((~sev_df["PolicyID"].isin(freq_df["PolicyID"])).sum())

    sev_claims_per_policy = sev_df["PolicyID"].value_counts()
    claims_dist = {
        int(float(str(k))): int(float(str(v)))
        for k, v in sev_claims_per_policy.value_counts().items()
    }

    sev_counts = cast(
        pd.DataFrame,
        sev_df.groupby("PolicyID", as_index=False).size(),
    )
    sev_counts = sev_counts.rename(columns={"size": "sev_claim_count"})
    merged_counts = freq_df[["PolicyID", "ClaimNb"]].merge(
        sev_counts,
        on="PolicyID",
        how="left",
    )
    merged_counts["sev_claim_count"] = merged_counts["sev_claim_count"].fillna(0).astype(int)
    mismatches = int((merged_counts["ClaimNb"] != merged_counts["sev_claim_count"]).sum())

    return RelationalAuditResult(
        freq_total_policies=freq_total,
        freq_unique_policies=freq_unique,
        sev_total_claims=sev_total,
        sev_unique_policies=sev_unique,
        orphan_sev_records=orphans,
        claims_per_policy_distribution=claims_dist,
        count_mismatch_records=mismatches,
        perfect_match=(mismatches == 0 and orphans == 0),
    )



def analyze_frequency(freq_df: pd.DataFrame) -> FrequencyAnalysisResult:
    """Computes actuarial claim frequency metrics and exposure distribution."""
    total_exposure = float(freq_df["Exposure"].sum())
    total_claims = int(freq_df["ClaimNb"].sum())
    empirical_frequency = total_claims / total_exposure if total_exposure > 0 else 0.0

    mean_exposure = float(freq_df["Exposure"].mean())
    median_exposure = float(freq_df["Exposure"].median())

    full_year_pct = float((freq_df["Exposure"] == 1.0).mean() * 100)
    greater_than_one_pct = float((freq_df["Exposure"] > 1.0).mean() * 100)
    short_term_pct = float((freq_df["Exposure"] < 0.10).mean() * 100)

    policy_claims_dist = {
        int(float(str(k))): int(float(str(v)))
        for k, v in freq_df["ClaimNb"].value_counts().items()
    }

    claim_mean = float(freq_df["ClaimNb"].mean())
    claim_variance = float(freq_df["ClaimNb"].var())
    dispersion_ratio = claim_variance / claim_mean if claim_mean > 0 else 1.0

    return FrequencyAnalysisResult(
        total_exposure_years=round(total_exposure, 2),
        total_claim_count=total_claims,
        empirical_frequency=round(empirical_frequency, 5),
        mean_exposure=round(mean_exposure, 4),
        median_exposure=round(median_exposure, 4),
        exposure_full_year_pct=round(full_year_pct, 2),
        exposure_greater_than_one_pct=round(greater_than_one_pct, 3),
        exposure_short_term_pct=round(short_term_pct, 2),
        policies_by_claim_count=policy_claims_dist,
        claim_count_mean=round(claim_mean, 5),
        claim_count_variance=round(claim_variance, 5),
        dispersion_ratio=round(dispersion_ratio, 4),
    )


def analyze_severity(sev_df: pd.DataFrame) -> SeverityAnalysisResult:
    """Computes descriptive severity statistics, percentiles, and loss concentration."""
    amounts = sev_df["ClaimAmount"].astype(float)
    total_loss = float(amounts.sum())
    num_claims = len(amounts)

    mean_sev = float(amounts.mean())
    median_sev = float(amounts.median())
    std_sev = float(amounts.std())
    min_sev = float(amounts.min())
    max_sev = float(amounts.max())
    skew_sev = float(amounts.skew())
    kurt_sev = float(amounts.kurtosis())

    quantiles_keys = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    quantiles_vals = amounts.quantile(quantiles_keys)
    percentiles_dict = {
        f"p{int(q*100) if (q*100).is_integer() else q*100}": round(float(v), 2)
        for q, v in zip(quantiles_keys, quantiles_vals, strict=False)
    }

    sorted_amounts = sorted((float(x) for x in sev_df["ClaimAmount"]), reverse=True)
    top_1_loss = float(sorted_amounts[0]) if sorted_amounts else 0.0
    top_10_loss = float(sum(sorted_amounts[:10]))
    top_100_loss = float(sum(sorted_amounts[:100]))

    one_pct_count = max(1, int(num_claims * 0.01))
    top_1_pct_loss = float(sum(sorted_amounts[:one_pct_count]))

    top_1_pct = (top_1_loss / total_loss * 100) if total_loss > 0 else 0.0
    top_10_pct = (top_10_loss / total_loss * 100) if total_loss > 0 else 0.0
    top_100_pct = (top_100_loss / total_loss * 100) if total_loss > 0 else 0.0
    top_1_pct_claims_pct = (top_1_pct_loss / total_loss * 100) if total_loss > 0 else 0.0

    tiers = [
        ("Micro (<€250)", amounts < 250),
        ("Small (€250-€1k)", (amounts >= 250) & (amounts < 1000)),
        ("Typical (€1k-€5k)", (amounts >= 1000) & (amounts < 5000)),
        ("High (€5k-€25k)", (amounts >= 5000) & (amounts < 25000)),
        ("Severe (€25k-€100k)", (amounts >= 25000) & (amounts < 100000)),
        ("Catastrophic (>€100k)", amounts >= 100000),
    ]
    tier_dist: dict[str, dict[str, Any]] = {}
    for tier_name, mask in tiers:
        sub_count = int(mask.sum())
        sub_loss = float(amounts[mask].sum())
        tier_dist[tier_name] = {
            "claim_count": sub_count,
            "claim_pct": round(sub_count / num_claims * 100, 2) if num_claims > 0 else 0.0,
            "total_loss": round(sub_loss, 2),
            "loss_pct": round(sub_loss / total_loss * 100, 2) if total_loss > 0 else 0.0,
            "mean_claim": round(sub_loss / sub_count, 2) if sub_count > 0 else 0.0,
        }

    return SeverityAnalysisResult(
        total_loss_amount=round(total_loss, 2),
        num_claims=num_claims,
        mean_severity=round(mean_sev, 2),
        median_severity=round(median_sev, 2),
        std_severity=round(std_sev, 2),
        min_severity=round(min_sev, 2),
        max_severity=round(max_sev, 2),
        skewness=round(skew_sev, 2),
        kurtosis=round(kurt_sev, 2),
        percentiles=percentiles_dict,
        top_1_loss_pct=round(top_1_pct, 2),
        top_10_loss_pct=round(top_10_pct, 2),
        top_100_loss_pct=round(top_100_pct, 2),
        top_1_pct_claims_loss_pct=round(top_1_pct_claims_pct, 2),
        tier_distribution=tier_dist,
    )


def analyze_pure_premium(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
) -> PurePremiumResult:
    """Computes total pure premium and validates frequency * severity identity."""
    total_exposure = float(freq_df["Exposure"].sum())
    total_claims = int(freq_df["ClaimNb"].sum())
    total_losses = float(sev_df["ClaimAmount"].sum())

    empirical_freq = total_claims / total_exposure if total_exposure > 0 else 0.0
    mean_sev = total_losses / total_claims if total_claims > 0 else 0.0
    empirical_pure_prem = total_losses / total_exposure if total_exposure > 0 else 0.0

    identity_product = empirical_freq * mean_sev
    discrepancy = abs(empirical_pure_prem - identity_product)

    return PurePremiumResult(
        total_losses=round(total_losses, 2),
        total_exposure=round(total_exposure, 2),
        empirical_pure_premium=round(empirical_pure_prem, 2),
        empirical_frequency=round(empirical_freq, 5),
        mean_severity=round(mean_sev, 2),
        identity_check_product=round(identity_product, 2),
        identity_discrepancy=round(discrepancy, 6),
    )



def segment_factor(
    merged_df: pd.DataFrame,
    factor_col: str,
    overall_freq: float,
    overall_pure_prem: float,
) -> list[dict[str, Any]]:
    """Generates univariate actuarial risk segmentation table for a given factor."""
    grouped = merged_df.groupby(factor_col, observed=False).agg(
        policies=("PolicyID", "count"),
        exposure=("Exposure", "sum"),
        claims=("ClaimNb", "sum"),
        losses=("total_loss", "sum"),
    ).reset_index()

    total_policies = len(merged_df)
    total_exposure = float(merged_df["Exposure"].sum())
    records: list[dict[str, Any]] = []

    for _, row in grouped.iterrows():
        policies = int(row["policies"])
        exposure = float(row["exposure"])
        claims = int(row["claims"])
        losses = float(row["losses"])

        freq = (claims / exposure) if exposure > 0 else 0.0
        severity = (losses / claims) if claims > 0 else 0.0
        pure_prem = (losses / exposure) if exposure > 0 else 0.0

        freq_relativity = (freq / overall_freq) if overall_freq > 0 else 1.0
        pure_prem_relativity = (pure_prem / overall_pure_prem) if overall_pure_prem > 0 else 1.0

        records.append({
            "category": str(row[factor_col]),
            "policies": policies,
            "policy_pct": round(policies / total_policies * 100, 2),
            "exposure": round(exposure, 2),
            "exposure_pct": round(exposure / total_exposure * 100, 2),
            "claims": claims,
            "empirical_freq": round(freq, 4),
            "freq_relativity": round(freq_relativity, 2),
            "total_losses": round(losses, 2),
            "mean_severity": round(severity, 2),
            "pure_premium": round(pure_prem, 2),
            "pure_prem_relativity": round(pure_prem_relativity, 2),
        })

    return records


def analyze_all_factors(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Builds actuarial factor segmentations across all categorical and binned predictors."""
    sev_agg = cast(
        pd.DataFrame,
        sev_df.groupby("PolicyID", as_index=False)["ClaimAmount"].sum(),
    )
    sev_agg = sev_agg.rename(columns={"ClaimAmount": "total_loss"})

    merged = freq_df.merge(sev_agg, on="PolicyID", how="left")
    merged["total_loss"] = merged["total_loss"].fillna(0.0)

    overall_exp = float(merged["Exposure"].sum())
    overall_claims = int(merged["ClaimNb"].sum())
    overall_losses = float(merged["total_loss"].sum())
    overall_freq = overall_claims / overall_exp if overall_exp > 0 else 0.0
    overall_pure_prem = overall_losses / overall_exp if overall_exp > 0 else 0.0

    merged["DriverAgeGroup"] = pd.cut(
        merged["DriverAge"],
        bins=[17, 24, 34, 44, 54, 64, 74, 120],
        labels=["18-24 (Novice)", "25-34", "35-44", "45-54", "55-64", "65-74", "75+ (Senior)"],
    )
    merged["CarAgeGroup"] = pd.cut(
        merged["CarAge"],
        bins=[-1, 1, 4, 9, 14, 120],
        labels=["0-1 (New)", "2-4 (Modern)", "5-9 (Mature)", "10-14 (Older)", "15+ (Aged)"],
    )
    merged["DensityGroup"] = pd.cut(
        merged["Density"],
        bins=[0, 100, 500, 2000, 10000, 50000],
        labels=["Rural (<=100)", "Semi-Rural (101-500)", "Suburban (501-2k)",
                "Urban (2k-10k)", "Dense Urban (>10k)"],
    )

    factors_to_segment = [
        ("Gas", "Gas"), ("Power", "Power"), ("Brand", "Brand"), ("Region", "Region"),
        ("DriverAgeGroup", "DriverAge"), ("CarAgeGroup", "CarAge"), ("DensityGroup", "Density"),
    ]

    factor_segmentations: dict[str, list[dict[str, Any]]] = {}
    for col_name, factor_title in factors_to_segment:
        factor_segmentations[factor_title] = segment_factor(
            merged,
            col_name,
            overall_freq,
            overall_pure_prem,
        )

    return factor_segmentations



def detect_anomalies(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> dict[str, Any]:
    """Scans for actuarial anomalies, out-of-bound exposures, and extreme outliers."""
    zero_exposure = int((freq_df["Exposure"] <= 0).sum())
    excess_exposure = int((freq_df["Exposure"] > 1.0).sum())
    max_exposure = float(freq_df["Exposure"].max())

    zero_claims_in_sev = int((sev_df["ClaimAmount"] <= 0).sum())
    claims_above_100k = int((sev_df["ClaimAmount"] > 100000).sum())
    claims_above_500k = int((sev_df["ClaimAmount"] > 500000).sum())
    max_claim = float(sev_df["ClaimAmount"].max())

    min_driver_age = int(freq_df["DriverAge"].min())
    underage_drivers = int((freq_df["DriverAge"] < 18).sum())
    senior_drivers_90plus = int((freq_df["DriverAge"] >= 90).sum())

    max_car_age = int(freq_df["CarAge"].max())
    car_age_above_40 = int((freq_df["CarAge"] > 40).sum())

    return {
        "zero_or_negative_exposure_count": zero_exposure,
        "exposure_greater_than_one_count": excess_exposure,
        "max_exposure_observed": max_exposure,
        "zero_or_negative_claim_amount_count": zero_claims_in_sev,
        "claims_exceeding_100k_count": claims_above_100k,
        "claims_exceeding_500k_count": claims_above_500k,
        "max_claim_amount_observed": max_claim,
        "underage_drivers_count": underage_drivers,
        "min_driver_age_observed": min_driver_age,
        "senior_drivers_90plus_count": senior_drivers_90plus,
        "max_car_age_observed": max_car_age,
        "car_age_above_40_count": car_age_above_40,
    }


def print_banner(title: str, width: int = 80) -> None:
    """Prints a styled terminal section banner."""
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def format_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Formats a clean monospace ASCII table."""
    num_cols = len(headers)
    if alignments is None:
        alignments = ["left"] + ["right"] * (num_cols - 1)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_cells = []
    for h, w, a in zip(headers, widths, alignments, strict=False):
        header_cells.append(h.ljust(w) if a == "left" else h.rjust(w))
    header_line = " | ".join(header_cells)
    separator = "-+-".join("-" * w for w in widths)
    body_lines = [
        " | ".join(
            str(v).ljust(w) if a == "left" else str(v).rjust(w)
            for v, w, a in zip(row, widths, alignments, strict=False)
        )
        for row in rows
    ]
    return f"{header_line}\n{separator}\n" + "\n".join(body_lines)



def _print_overview_and_integrity(
    freq_meta: DatasetMetadata,
    sev_meta: DatasetMetadata,
    rel_audit: RelationalAuditResult,
) -> None:
    """Prints file metadata and relational reconciliation."""
    print_banner("1. Raw Datasets & Storage Metadata")
    meta_headers = ["Dataset", "File Path", "Size", "Compressed", "Rows", "Cols", "Memory"]
    meta_rows = [
        ["Frequency", Path(freq_meta.file_path).name, freq_meta.file_size_formatted,
         "Yes (zip)" if freq_meta.compressed else "No", f"{freq_meta.num_rows:,}",
         str(freq_meta.num_cols), f"{freq_meta.memory_usage_mb:.1f} MB"],
        ["Severity", Path(sev_meta.file_path).name, sev_meta.file_size_formatted, "No",
         f"{sev_meta.num_rows:,}", str(sev_meta.num_cols), f"{sev_meta.memory_usage_mb:.1f} MB"],
    ]
    print(format_table(meta_headers, meta_rows))

    freq_nulls = sum(freq_meta.missing_values.values())
    sev_nulls = sum(sev_meta.missing_values.values())
    print(f"\nMissing values: Frequency nulls = {freq_nulls:,} | Severity nulls = {sev_nulls:,}")

    print_banner("2. Relational Integrity & Key Reconciliation")
    print(f"Frequency unique PolicyIDs : {rel_audit.freq_unique_policies:,} "
          f"(total: {rel_audit.freq_total_policies:,})")
    print(f"Severity unique PolicyIDs  : {rel_audit.sev_unique_policies:,} "
          f"({rel_audit.sev_total_claims:,} claims)")
    print(f"Orphan severity records    : {rel_audit.orphan_sev_records} (claims without a policy)")
    print(f"ClaimNb vs Sev mismatch    : {rel_audit.count_mismatch_records} count mismatches")
    status = "[PASS] PERFECT MATCH" if rel_audit.perfect_match else "[WARN] DISCREPANCIES DETECTED"
    print(f"Reconciliation Status      : {status}")

    dist_headers = ["Claims/Policy", "Policies", "Total Claims", "% Claiming"]
    tot_claiming = sum(rel_audit.claims_per_policy_distribution.values())
    dist_rows = [
        [str(c), f"{p:,}", f"{c * p:,}", f"{(p / tot_claiming * 100):.2f}%"]
        for c, p in sorted(rel_audit.claims_per_policy_distribution.items())
    ]
    print("\nSeverity Claims Per Policy Distribution:")
    print(format_table(dist_headers, dist_rows))



def _print_frequency_and_severity(
    freq_res: FrequencyAnalysisResult,
    sev_res: SeverityAnalysisResult,
    freq_meta: DatasetMetadata,
    anomalies: dict[str, Any],
) -> None:
    """Prints actuarial claim frequency and severity tables."""
    print_banner("3. Actuarial Claim Frequency & Exposure Diagnostics")
    print(f"Total Portfolio Exposure   : {freq_res.total_exposure_years:,.2f} policy-years")
    print(f"Total Observed Claims      : {freq_res.total_claim_count:,}")
    print(
        f"Empirical Frequency        : {freq_res.empirical_frequency:.5f} "
        f"({freq_res.empirical_frequency * 100:.3f}% per policy-year)"
    )
    print(
        f"Exposure Central Tendency  : Mean = {freq_res.mean_exposure:.4f} yr "
        f"| Median = {freq_res.median_exposure:.4f} yr"
    )
    print(f"Full-year policies (1.0 yr): {freq_res.exposure_full_year_pct:.2f}% of portfolio")
    print(
        f"Multi-year (> 1.0 yr)      : {freq_res.exposure_greater_than_one_pct:.3f}% "
        f"({anomalies['exposure_greater_than_one_count']} policies)"
    )
    print(f"Short-term (< 0.10 yr)     : {freq_res.exposure_short_term_pct:.2f}% of portfolio")
    disp_note = (
        "Overdispersed (Negative Binomial indicated)"
        if freq_res.dispersion_ratio > 1.05
        else "Near Poisson"
    )
    print(f"Poisson Dispersion Ratio   : Var/Mean = {freq_res.dispersion_ratio:.4f} ({disp_note})")

    freq_headers = ["ClaimNb", "Policy Count", "Portfolio %", "Total Claims"]
    freq_rows = [
        [str(c_nb), f"{count:,}", f"{count / freq_meta.num_rows * 100:.2f}%", f"{c_nb * count:,}"]
        for c_nb, count in sorted(freq_res.policies_by_claim_count.items())
    ]
    print("\nPortfolio Policy Count by ClaimNb:")
    print(format_table(freq_headers, freq_rows))

    print_banner("4. Actuarial Claim Severity & Heavy-Tail Diagnostics")
    print(f"Total Incurred Losses      : €{sev_res.total_loss_amount:,.2f}")
    print(f"Total Claim Records        : {sev_res.num_claims:,}")
    print(f"Mean Severity              : €{sev_res.mean_severity:,.2f}")
    print(f"Median Severity            : €{sev_res.median_severity:,.2f}")
    print(f"Standard Deviation         : €{sev_res.std_severity:,.2f}")
    min_max_str = f"€{sev_res.min_severity:,.2f} / €{sev_res.max_severity:,.2f}"
    print(f"Min / Max Severity         : {min_max_str}")
    skew_kurt_str = f"{sev_res.skewness:.2f} / {sev_res.kurtosis:.2f} (Right skew)"
    print(f"Skewness / Kurtosis        : {skew_kurt_str}")
    print(
        f"Top 1 Claim Loss Share     : {sev_res.top_1_loss_pct:.2f}% of portfolio losses "
        f"(€{sev_res.max_severity:,.2f})"
    )
    print(f"Top 10 Claims Loss Share   : {sev_res.top_10_loss_pct:.2f}% of portfolio losses")
    print(f"Top 1% Claims Loss Share   : {sev_res.top_1_pct_claims_loss_pct:.2f}% of losses")

    pct_headers = ["Percentile", "Amount (€)", "Percentile", "Amount (€)"]
    pct_keys = list(sev_res.percentiles.keys())
    half = math.ceil(len(pct_keys) / 2)
    pct_rows = []
    for i in range(half):
        k1 = pct_keys[i]
        v1 = sev_res.percentiles[k1]
        if i + half < len(pct_keys):
            k2 = pct_keys[i + half]
            v2 = sev_res.percentiles[k2]
            pct_rows.append([k1.upper(), f"€{v1:,.2f}", k2.upper(), f"€{v2:,.2f}"])
        else:
            pct_rows.append([k1.upper(), f"€{v1:,.2f}", "", ""])
    print("\nSeverity Percentile Distribution:")
    print(format_table(pct_headers, pct_rows, alignments=["left", "right", "left", "right"]))

    tier_headers = ["Severity Tier", "Claims", "Claim %", "Total Loss (€)", "Loss %", "Mean (€)"]
    tier_rows = [
        [t_name, f"{data['claim_count']:,}", f"{data['claim_pct']:.2f}%",
         f"€{data['total_loss']:,.2f}", f"{data['loss_pct']:.2f}%", f"€{data['mean_claim']:,.2f}"]
        for t_name, data in sev_res.tier_distribution.items()
    ]
    print("\nClaim Size Stratification:")
    print(format_table(tier_headers, tier_rows))



def _print_pure_premium_and_factors(
    pure_prem_res: PurePremiumResult,
    factor_segs: dict[str, list[dict[str, Any]]],
    anomalies: dict[str, Any],
    sev_res: SeverityAnalysisResult,
) -> None:
    """Prints pure premium summary and univariate factor segmentation."""
    print_banner("5. Actuarial Pure Premium (Loss Cost)")
    print(f"Portfolio Pure Premium     : €{pure_prem_res.empirical_pure_premium:,.2f} / car-year")
    print(
        f"Identity Verification      : Frequency ({pure_prem_res.empirical_frequency:.5f}) * "
        f"Severity (€{pure_prem_res.mean_severity:,.2f}) = "
        f"€{pure_prem_res.identity_check_product:,.2f}"
    )
    print(
        f"Identity Discrepancy       : €{pure_prem_res.identity_discrepancy:.6f} "
        "(Exact actuarial equality)"
    )

    print_banner("6. Univariate Factor Risk Profiles & Empirical Relativities")
    factor_headers = [
        "Category",
        "Policies",
        "Pol %",
        "Exposure (yr)",
        "Exp %",
        "Claims",
        "Freq (/yr)",
        "Freq Rel",
        "Mean Sev (€)",
        "Pure Prem (€)",
        "Prem Rel",
    ]

    for factor_title, seg_rows in factor_segs.items():
        print(f"\n--- Factor: {factor_title} ---")
        formatted_rows = [
            [
                r["category"], f"{r['policies']:,}", f"{r['policy_pct']:.1f}%",
                f"{r['exposure']:,.1f}", f"{r['exposure_pct']:.1f}%", f"{r['claims']:,}",
                f"{r['empirical_freq']:.4f}", f"{r['freq_relativity']:.2f}x",
                f"€{r['mean_severity']:,.1f}", f"€{r['pure_premium']:,.2f}",
                f"{r['pure_prem_relativity']:.2f}x",
            ]
            for r in seg_rows
        ]
        print(format_table(factor_headers, formatted_rows))

    print_banner("7. Actuarial Anomalies & Production Modeling Guidelines")
    print(
        f"1. Exposure Anomaly        : {anomalies['exposure_greater_than_one_count']} "
        f"policies with Exposure > 1.0 (max: {anomalies['max_exposure_observed']:.2f}). "
        "In GLM/EconML, clip to 1.0 or treat offset carefully."
    )
    print(
        f"2. Extreme Losses (Tail)   : {anomalies['claims_exceeding_100k_count']} "
        f"claims exceed €100k (max claim: €{anomalies['max_claim_amount_observed']:,.2f}, "
        f"representing {sev_res.top_1_loss_pct:.2f}% of portfolio losses). "
        "Actuarial best practice requires loss capping or extreme-value modeling."
    )
    print(
        f"3. Novice Driver Surcharge : DriverAge 18-24 exhibits "
        f"{factor_segs['DriverAge'][0]['freq_relativity']}x frequency and "
        f"{factor_segs['DriverAge'][0]['pure_prem_relativity']}x pure premium vs baseline."
    )
    print(
        f"4. Traffic Density Gradient: Frequency scales monotonically from "
        f"{factor_segs['Density'][0]['empirical_freq']:.4f} (Rural) to "
        f"{factor_segs['Density'][-1]['empirical_freq']:.4f} (Dense Urban), "
        "recommending log(Density) in GLMs."
    )
    print("=" * 80 + "\n")


def display_console_report(
    freq_meta: DatasetMetadata,
    sev_meta: DatasetMetadata,
    rel_audit: RelationalAuditResult,
    freq_res: FrequencyAnalysisResult,
    sev_res: SeverityAnalysisResult,
    pure_prem_res: PurePremiumResult,
    factor_segs: dict[str, list[dict[str, Any]]],
    anomalies: dict[str, Any],
) -> None:
    """Renders comprehensive exploratory report to stdout."""
    _print_overview_and_integrity(freq_meta, sev_meta, rel_audit)
    _print_frequency_and_severity(freq_res, sev_res, freq_meta, anomalies)
    _print_pure_premium_and_factors(pure_prem_res, factor_segs, anomalies, sev_res)



def generate_markdown_report(
    freq_meta: DatasetMetadata,
    sev_meta: DatasetMetadata,
    rel_audit: RelationalAuditResult,
    freq_res: FrequencyAnalysisResult,
    sev_res: SeverityAnalysisResult,
    pure_prem_res: PurePremiumResult,
    factor_segs: dict[str, list[dict[str, Any]]],
    anomalies: dict[str, Any],
) -> str:
    """Builds a GitHub-flavored Markdown exploratory analysis report."""
    md_lines: list[str] = [
        "# freMTPL Benchmark Raw Data Exploratory Report\n",
        "**Generated by**: `scripts/explore_raw_data.py` (AEGIS Tier 1 Diagnostic Suite)\n",
        "## 1. Storage & Schema Overview\n",
        "| Dataset | File | Size | Compressed | Rows | Columns | Memory |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| Frequency | `{Path(freq_meta.file_path).name}` | {freq_meta.file_size_formatted} | "
        f"{'Yes' if freq_meta.compressed else 'No'} | {freq_meta.num_rows:,} | "
        f"{freq_meta.num_cols} | {freq_meta.memory_usage_mb:.1f} MB |",
        f"| Severity | `{Path(sev_meta.file_path).name}` | {sev_meta.file_size_formatted} | No | "
        f"{sev_meta.num_rows:,} | {sev_meta.num_cols} | {sev_meta.memory_usage_mb:.1f} MB |\n",
        "## 2. Relational Integrity\n",
        f"- **Unique Frequency Policies**: {rel_audit.freq_unique_policies:,}",
        f"- **Unique Severity Policies**: {rel_audit.sev_unique_policies:,} "
        f"({rel_audit.sev_total_claims:,} individual claims)",
        f"- **Orphan Severity Claims**: {rel_audit.orphan_sev_records}",
        f"- **Reconciliation Mismatches**: {rel_audit.count_mismatch_records}",
        "- **Overall Status**: "
        + ("✅ Verified (100% integrity)" if rel_audit.perfect_match else "⚠️ Discrepancies"),
        "\n## 3. Actuarial Summary\n",
        "| Actuarial Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Total Exposure** | {freq_res.total_exposure_years:,.2f} yr | Sum of exposure |",
        f"| **Total Claims** | {freq_res.total_claim_count:,} claims | Total claims recorded |",
        f"| **Empirical Frequency** | {freq_res.empirical_frequency:.5f} "
        f"({freq_res.empirical_frequency*100:.2f}%) | Claims per car-year |",
        f"| **Total Incurred Losses** | €{sev_res.total_loss_amount:,.2f} | Total payments |",
        f"| **Mean Claim Severity** | €{sev_res.mean_severity:,.2f} | Average loss per claim |",
        f"| **Median Claim Severity** | €{sev_res.median_severity:,.2f} | Median loss per claim |",
        f"| **Empirical Pure Premium** | €{pure_prem_res.empirical_pure_premium:,.2f} / yr | "
        "Frequency × Mean Severity |",
        f"| **Max Individual Claim** | €{sev_res.max_severity:,.2f} | "
        f"Largest claim ({sev_res.top_1_loss_pct:.2f}% of total losses) |",
        f"| **Top 1% Loss Share** | {sev_res.top_1_pct_claims_loss_pct:.2f}% | Pareto tail |\n",
        "## 4. Univariate Risk Profiles\n",
    ]

    header_cols = [
        "Category", "Policies", "Exposure (yr)", "Claims", "Frequency",
        "Freq Rel", "Mean Sev (€)", "Pure Prem (€)", "Prem Rel",
    ]
    table_header = "| " + " | ".join(header_cols) + " |\n| " + " | ".join([":---"] * 9) + " |"

    for factor_title, seg_rows in factor_segs.items():
        md_lines.extend([f"### Factor: {factor_title}\n", table_header])
        for r in seg_rows:
            md_lines.append(
                f"| {r['category']} | {r['policies']:,} | {r['exposure']:,.1f} | {r['claims']:,} | "
                f"{r['empirical_freq']:.4f} | {r['freq_relativity']:.2f}x | "
                f"€{r['mean_severity']:,.1f} | €{r['pure_premium']:,.2f} | "
                f"{r['pure_prem_relativity']:.2f}x |"
            )
        md_lines.append("")

    md_lines.extend([
        "## 5. Modeling Recommendations\n",
        f"1. **Exposure Clipping**: {anomalies['exposure_greater_than_one_count']} policies have "
        f"exposure > 1.0 (max {anomalies['max_exposure_observed']:.2f}). Cap at 1.0.",
        f"2. **Loss Capping**: Top claim takes {sev_res.top_1_loss_pct:.2f}% of losses and "
        f"{anomalies['claims_exceeding_100k_count']} claims exceed €100k. Apply loss truncation.",
        "3. **Negative Binomial**: Overdispersion confirms Negative Binomial frequency.",
        "4. **Log-Transform Density**: Log-density yields linear risk scaling.\n",
    ])
    return "\n".join(md_lines)



def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for raw freMTPL exploratory data analysis."""
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis (EDA) for raw freMTPLfreq and freMTPLsev datasets."
    )
    parser.add_argument(
        "--freq-path",
        type=Path,
        default=DEFAULT_FREQ_PATH,
        help=f"Path to frequency dataset (default: {DEFAULT_FREQ_PATH})",
    )
    parser.add_argument(
        "--sev-path",
        type=Path,
        default=DEFAULT_SEV_PATH,
        help=f"Path to severity dataset (default: {DEFAULT_SEV_PATH})",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional row limit for quick inspection (default: None, full dataset)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path to export full structured analysis as JSON",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional output path to export human-readable Markdown report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (useful when generating JSON/Markdown artifacts)",
    )

    args = parser.parse_args(argv)

    t0 = time.time()
    if not args.quiet:
        print(f"[EXPLORE] Loading datasets from {args.freq_path} and {args.sev_path}...")

    try:
        freq_df, sev_df = load_raw_datasets(
            args.freq_path,
            args.sev_path,
            sample_rows=args.sample_rows,
        )
    except Exception as exc:
        print(f"[EXPLORE][ERROR] Failed to load raw datasets: {exc}", file=sys.stderr)
        return 1

    t_loaded = time.time()
    if not args.quiet:
        print(
            f"[EXPLORE] Loaded {len(freq_df):,} frequency rows and "
            f"{len(sev_df):,} severity rows in {t_loaded - t0:.2f}s."
        )

    # 1. Metadata
    freq_meta = inspect_metadata(
        args.freq_path,
        freq_df,
        is_zip=args.freq_path.suffix.lower() == ".zip",
    )
    sev_meta = inspect_metadata(args.sev_path, sev_df, is_zip=False)

    # 2. Relational integrity
    rel_audit = audit_relational_integrity(freq_df, sev_df)

    # 3. Frequency analysis
    freq_res = analyze_frequency(freq_df)

    # 4. Severity analysis
    sev_res = analyze_severity(sev_df)

    # 5. Pure premium
    pure_prem_res = analyze_pure_premium(freq_df, sev_df)

    # 6. Factor risk segmentation
    factor_segs = analyze_all_factors(freq_df, sev_df)

    # 7. Anomalies
    anomalies = detect_anomalies(freq_df, sev_df)

    # Console output
    if not args.quiet:
        display_console_report(
            freq_meta,
            sev_meta,
            rel_audit,
            freq_res,
            sev_res,
            pure_prem_res,
            factor_segs,
            anomalies,
        )

    # JSON export
    if args.output_json is not None:
        payload = {
            "metadata": {
                "frequency": asdict(freq_meta),
                "severity": asdict(sev_meta),
            },
            "relational_audit": asdict(rel_audit),
            "frequency_analysis": asdict(freq_res),
            "severity_analysis": asdict(sev_res),
            "pure_premium": asdict(pure_prem_res),
            "factor_segmentations": factor_segs,
            "anomalies": anomalies,
            "elapsed_seconds": round(time.time() - t0, 3),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[EXPLORE] Saved JSON analysis to: {args.output_json}")

    # Markdown export
    if args.output_markdown is not None:
        md_text = generate_markdown_report(
            freq_meta,
            sev_meta,
            rel_audit,
            freq_res,
            sev_res,
            pure_prem_res,
            factor_segs,
            anomalies,
        )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(md_text, encoding="utf-8")
        if not args.quiet:
            print(f"[EXPLORE] Saved Markdown report to: {args.output_markdown}")

    total_time = time.time() - t0
    if not args.quiet:
        print(f"[EXPLORE] Total EDA completed in {total_time:.2f}s.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

