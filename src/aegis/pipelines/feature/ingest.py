"""Dataset ingestion pipeline for the freMTPL2 benchmark (ADR-011, ADR-019).

Fetches the French Motor Third-Party Liability datasets (freMTPL2freq and freMTPL2sev)
via OpenML, applies canonical schema mapping, aggregates claim amounts, derives baseline
pure premium ratings, and outputs DVC-trackable raw datasets.
"""

from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.datasets import fetch_openml

from aegis.utils.exceptions import DataContractError


def fetch_fremtpl2_data(
    data_id_freq: int = 41214,
    data_id_sev: int = 41215,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches freMTPL2 frequency and severity datasets from OpenML.

    Args:
        data_id_freq: OpenML dataset ID for freMTPL2freq (default 41214).
        data_id_sev: OpenML dataset ID for freMTPL2sev (default 41215).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (frequency_df, severity_df).

    Raises:
        DataContractError: If download fails or data cannot be retrieved.
    """
    try:
        print(f"[INGEST] Fetching freMTPL2freq (data_id={data_id_freq}) from OpenML...")
        freq_bunch: Any = fetch_openml(data_id=data_id_freq, as_frame=True, parser="auto")
        freq_df = cast(pd.DataFrame, getattr(freq_bunch, "frame", freq_bunch))

        print(f"[INGEST] Fetching freMTPL2sev (data_id={data_id_sev}) from OpenML...")
        sev_bunch: Any = fetch_openml(data_id=data_id_sev, as_frame=True, parser="auto")
        sev_df = cast(pd.DataFrame, getattr(sev_bunch, "frame", sev_bunch))

        return freq_df, sev_df
    except Exception as exc:
        raise DataContractError(
            f"Failed to fetch freMTPL2 datasets from OpenML: {exc}",
            details={"data_id_freq": data_id_freq, "data_id_sev": data_id_sev, "error": str(exc)},
        ) from exc


def clean_and_merge_fremtpl2(
    freq_df: pd.DataFrame,
    sev_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cleans, standardizes, and joins freMTPL2 frequency and severity tables.

    Performs:
    1. Aggregation of severity claims by policy ID (`IDpol`).
    2. Left join onto frequency table (policies with no claims receive 0.0 claim amount).
    3. Column renaming to AEGIS canonical schema (ADR-019).
    4. Baseline rating pure premium derivation.
    5. Neutral baseline treatment initialization (0.0).

    Args:
        freq_df: Raw frequency DataFrame.
        sev_df: Raw severity DataFrame.

    Returns:
        pd.DataFrame: Cleaned and standardized policy-level insurance dataset.
    """
    # 1. Aggregate claim amounts by policy ID
    sev_grouped = cast(pd.DataFrame, sev_df.groupby("IDpol", as_index=False)["ClaimAmount"].sum())
    sev_agg = sev_grouped.rename(columns={"ClaimAmount": "claim_amount"})

    # 2. Merge frequency policies with aggregated claims
    merged = pd.merge(freq_df, sev_agg, on="IDpol", how="left")
    merged["claim_amount"] = merged["claim_amount"].fillna(0.0).astype(float)

    # 3. Canonical column renaming (ADR-019)
    rename_mapping = {
        "IDpol": "policy_id",
        "ClaimNb": "claim_count",
        "Exposure": "exposure",
        "Area": "area",
        "VehPower": "veh_power",
        "VehAge": "veh_age",
        "DrivAge": "driver_age",
        "BonusMalus": "bonus_malus",
        "VehBrand": "veh_brand",
        "VehGas": "veh_gas",
        "Density": "density",
        "Region": "region",
    }
    df = merged.rename(columns=rename_mapping)

    # Format policy_id as clean identifier string
    df["policy_id"] = df["policy_id"].apply(lambda x: f"POL-{int(x)}")

    # Ensure integer and float column types
    df["driver_age"] = df["driver_age"].astype(float)
    df["veh_age"] = df["veh_age"].astype(float)
    df["exposure"] = df["exposure"].astype(float)
    df["claim_count"] = df["claim_count"].astype(float)
    df["bonus_malus"] = df["bonus_malus"].astype(float)
    df["veh_power"] = df["veh_power"].astype(float)
    df["density"] = df["density"].astype(float)

    # 4. Derive baseline rating pure premium (ADR-019)
    # Standard actuarial tariff proxy based on rating factors (BonusMalus, VehPower, Exposure)
    base_tariff = 250.0
    bm_factor = df["bonus_malus"] / 100.0
    power_factor = 1.0 + (df["veh_power"] - 6.0).clip(lower=0.0) * 0.05
    df["premium"] = (base_tariff * bm_factor * power_factor * df["exposure"]).round(2)
    # Ensure strict positivity floor
    df["premium"] = df["premium"].clip(lower=10.0)

    # 5. Initialize neutral treatment rate change (0.0) for baseline ingestion
    df["treatment_rate_change"] = 0.0

    return df


def ingest_fremtpl2_pipeline(
    output_path: Path | str = "data/raw/elasticity_fremtpl2.csv",
    force_download: bool = False,
) -> Path:
    """Executes the full freMTPL2 ingestion pipeline and writes the output CSV.

    If output_path already exists and force_download is False, skips network download
    to ensure offline reproducibility for DVC and CI runs (ADR-011, INV-10).

    Args:
        output_path: Target CSV path for the cleaned dataset.
        force_download: If True, forces re-download from OpenML even if output exists.

    Returns:
        Path: Absolute or relative path to the saved raw dataset.
    """
    dest_path = Path(output_path)
    if dest_path.is_file() and not force_download:
        print(f"[INGEST] Existing dataset found at {dest_path}. Skipping network fetch.")
        return dest_path

    print(f"[INGEST] Ingesting freMTPL2 from OpenML into {dest_path}...")
    freq_df, sev_df = fetch_fremtpl2_data()
    clean_df = clean_and_merge_fremtpl2(freq_df, sev_df)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(dest_path, index=False)
    print(f"[INGEST] Successfully written {len(clean_df):,} records to {dest_path}")
    return dest_path
