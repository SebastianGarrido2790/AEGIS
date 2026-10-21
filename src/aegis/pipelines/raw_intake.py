"""Raw observational data intake and joining module (INV-3, ADR-015, ADR-003).

Provides cryptographic checksum verification against a committed manifest and
deterministic policy-level joining of frequency and severity records with
multi-claim aggregation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from aegis.config.loader import get_config
from aegis.utils.exceptions import DataContractError

RAW_POLICY_COLUMNS: list[str] = [
    "PolicyID",
    "ClaimNb",
    "Exposure",
    "Power",
    "CarAge",
    "DriverAge",
    "Brand",
    "Gas",
    "Region",
    "Density",
    "ClaimAmount",
    "sev_claim_count",
    "is_orphan_severity",
]


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 hash of a file on disk.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte chunk size for streaming hash computation.

    Returns:
        Hexadecimal SHA-256 string.

    Raises:
        DataContractError: If the file does not exist or cannot be read.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataContractError(
            f"File not found for checksum computation: {path}",
            details={"file_path": str(path)},
        )

    hasher = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    except Exception as exc:
        raise DataContractError(
            f"Failed to read file for checksum computation: {exc}",
            details={"file_path": str(path), "error": str(exc)},
        ) from exc

    return hasher.hexdigest().lower()


def verify_checksums(
    manifest_path: Path | str | None = None,
    base_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Verifies that raw files match the SHA-256 hashes recorded in the manifest.

    Args:
        manifest_path: Path to the JSON manifest. Defaults to params.yaml.
        base_dir: Base directory for relative file paths in manifest. Defaults to cwd.

    Returns:
        List of verified file metadata records.

    Raises:
        DataContractError: If manifest is missing/invalid or any file fails verification.
    """
    config = get_config()
    target_manifest = Path(manifest_path or config.data_contracts.raw_intake_manifest_path)
    root = Path(base_dir or ".")

    if not target_manifest.is_file():
        raise DataContractError(
            f"Raw intake manifest file not found: {target_manifest}",
            details={"manifest_path": str(target_manifest)},
        )

    try:
        with target_manifest.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as exc:
        raise DataContractError(
            f"Failed to parse raw intake manifest JSON: {exc}",
            details={"manifest_path": str(target_manifest), "error": str(exc)},
        ) from exc

    file_entries = manifest_data.get("files", [])
    if not file_entries:
        raise DataContractError(
            "Raw intake manifest contains no file entries to verify.",
            details={"manifest_path": str(target_manifest)},
        )

    verified_records: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for entry in file_entries:
        rel_path = entry.get("path")
        expected_sha = (entry.get("sha256") or "").lower()
        expected_size = entry.get("size_bytes")

        if not rel_path or not expected_sha:
            raise DataContractError(
                f"Malformed manifest entry: {entry}",
                details={"entry": entry},
            )

        target_file = root / rel_path
        if not target_file.is_file():
            mismatches.append(
                {
                    "path": rel_path,
                    "error": "File does not exist on disk",
                    "expected_sha256": expected_sha,
                }
            )
            continue

        actual_sha = compute_sha256(target_file)
        actual_size = target_file.stat().st_size

        if actual_sha != expected_sha:
            mismatches.append(
                {
                    "path": rel_path,
                    "error": "SHA-256 hash mismatch",
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
        elif expected_size is not None and actual_size != expected_size:
            mismatches.append(
                {
                    "path": rel_path,
                    "error": "File size mismatch",
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                }
            )
        else:
            verified_records.append(
                {
                    "path": rel_path,
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                    "status": "VERIFIED",
                }
            )

    if mismatches:
        raise DataContractError(
            f"Raw intake checksum verification failed for {len(mismatches)} file(s).",
            details={
                "manifest_path": str(target_manifest),
                "mismatches": mismatches,
                "verified_count": len(verified_records),
            },
        )

    return verified_records


def load_and_join_raw_policy_data(
    freq_path: Path | str,
    sev_path: Path | str,
) -> pd.DataFrame:
    """Loads frequency and severity datasets and performs policy-level joining.

    Aggregates multi-claim severity records per policy via sum of ClaimAmount,
    reconciles claim counts, and flags any orphan severity records.

    Args:
        freq_path: Path to frequency CSV or CSV.ZIP file.
        sev_path: Path to severity CSV file.

    Returns:
        pd.DataFrame containing the joined policy records.

    Raises:
        DataContractError: If files cannot be read or joined.
    """
    f_path = Path(freq_path)
    s_path = Path(sev_path)

    if not f_path.is_file():
        raise DataContractError(
            f"Frequency file not found: {f_path}",
            details={"freq_path": str(f_path)},
        )
    if not s_path.is_file():
        raise DataContractError(
            f"Severity file not found: {s_path}",
            details={"sev_path": str(s_path)},
        )

    try:
        freq_df = pd.read_csv(f_path)
    except Exception as exc:
        raise DataContractError(
            f"Failed to read frequency dataset: {exc}",
            details={"freq_path": str(f_path), "error": str(exc)},
        ) from exc

    try:
        sev_df = pd.read_csv(s_path)
    except Exception as exc:
        raise DataContractError(
            f"Failed to read severity dataset: {exc}",
            details={"sev_path": str(s_path), "error": str(exc)},
        ) from exc

    # Aggregate severity records by PolicyID (accounting for multi-claim policies)
    if not sev_df.empty and "PolicyID" in sev_df.columns and "ClaimAmount" in sev_df.columns:
        sev_agg = (
            sev_df.groupby("PolicyID")
            .agg(
                sev_claim_count=("ClaimAmount", "count"),
                ClaimAmount=("ClaimAmount", "sum"),
            )
            .reset_index()
        )
    else:
        sev_agg = pd.DataFrame({"PolicyID": [], "sev_claim_count": [], "ClaimAmount": []})

    # Outer join to capture both valid matches and any orphan severity records
    merged = pd.merge(freq_df, sev_agg, on="PolicyID", how="outer")

    # Flag orphan severity records (policies in severity with missing frequency record)
    merged["is_orphan_severity"] = merged["Exposure"].isna()

    # Fill defaults for non-claiming policies
    merged["ClaimAmount"] = merged["ClaimAmount"].fillna(0.0).astype(float)
    merged["sev_claim_count"] = merged["sev_claim_count"].fillna(0).astype(int)

    # Reorder columns to standard schema if all expected columns are present
    ordered_cols = [col for col in RAW_POLICY_COLUMNS if col in merged.columns]
    remaining_cols = [col for col in merged.columns if col not in ordered_cols]
    final_cols = ordered_cols + remaining_cols
    result_df = pd.DataFrame(merged[final_cols])
    return result_df


def ingest_raw_data(
    freq_path: Path | str | None = None,
    sev_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    output_path: Path | str = Path("data/raw/raw_policy_data.csv"),
) -> Path:
    """Executes the full raw intake pipeline: checksum verification and joining.

    Args:
        freq_path: Path to frequency file. Defaults to data/raw/freMTPLfreq.csv.zip.
        sev_path: Path to severity file. Defaults to data/raw/freMTPLsev.csv.
        manifest_path: Path to manifest. Defaults to params.yaml.
        output_path: Path to write joined CSV output.

    Returns:
        Path to the generated raw joined policy CSV.

    Raises:
        DataContractError: If checksum verification fails or joining encounters errors.
    """
    f_path = Path(freq_path or "data/raw/freMTPLfreq.csv.zip")
    s_path = Path(sev_path or "data/raw/freMTPLsev.csv")
    out_path = Path(output_path)

    # 1. Cryptographic checksum verification
    verify_checksums(manifest_path=manifest_path)

    # 2. Policy-level joining
    joined_df = load_and_join_raw_policy_data(freq_path=f_path, sev_path=s_path)

    # 3. Write joined CSV output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_csv(out_path, index=False)
    print(f"[INGEST-RAW] Processed {len(joined_df):,} raw policy records -> {out_path}")
    return out_path
