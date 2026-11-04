"""Artifact version manifest for Phase 2 reproducibility evidence (P2-F10, ADR-015).

ADR-015 makes a tiered reproducibility claim. This module produces the evidence for the
strictest tier: an ordered SHA-256 digest set over every canonical data product, plus a
single combined digest. Deterministic stages must yield an identical combined digest
across consecutive runs; a changed digest localizes the drift to one artifact.

The module performs filesystem reads and one atomic write. It fits no model and reads no
training code, so it can run as the terminal DVC stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aegis.pipelines.causal_artifacts import write_json_atomic
from aegis.schemas.versioning import ArtifactDigest, VersionManifest
from aegis.utils.exceptions import VersioningError

# Declared in pipeline order: raw intake, GX report, features, baseline, segments, DGP,
# training GX, estimator, recovery verification. Treated as a structural identity set,
# not a tunable, so it lives with the command that produces it (ADR-015).
CANONICAL_ARTIFACTS: tuple[str, ...] = (
    "data/versioned/raw_policy_data.csv",
    "data/validated/raw_policy_validation_report.json",
    "data/processed/features_glm.csv",
    "data/processed/features_forest.csv",
    "data/processed/feature_metadata.json",
    "data/processed/glm_fitted_loss_cost.csv",
    "data/processed/glm_baseline_model.json",
    "data/processed/glm_calibration_report.json",
    "data/processed/segment_loss_cost.csv",
    "data/processed/segment_definitions.json",
    "data/processed/policy_segments.csv",
    "data/processed/elasticity_training.csv",
    "data/processed/truth_sidecar.csv",
    "data/processed/dgp_manifest.json",
    "data/validated/elasticity_training_validation_report.json",
    "data/processed/segment_elasticity_estimates.csv",
    "data/processed/causal_estimator_metadata.json",
    "data/validated/causal_validation_report.json",
)


def _digest_file(path: Path) -> ArtifactDigest:
    """Digests one declared artifact, failing closed on a missing or unreadable file.

    Args:
        path: Artifact path as declared in :data:`CANONICAL_ARTIFACTS`.

    Returns:
        Path, SHA-256, and byte size for the artifact.

    Raises:
        VersioningError: If the artifact is absent or cannot be read.
    """
    if not path.is_file():
        raise VersioningError(
            f"Declared pipeline artifact is missing: {path}",
            details={"rule": "missing_artifact", "path": str(path)},
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
    except OSError as exc:
        raise VersioningError(
            f"Failed to digest artifact {path}: {exc}",
            details={"rule": "digest_failed", "path": str(path)},
        ) from exc
    return ArtifactDigest(
        path=path.as_posix(), sha256=digest.hexdigest(), size_bytes=size
    )


def combined_digest(artifacts: Sequence[ArtifactDigest]) -> str:
    """Computes one digest over the ordered (path, sha256) pairs.

    Args:
        artifacts: Digests in declared pipeline order.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    payload = [{"path": item.path, "sha256": item.sha256} for item in artifacts]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_version_manifest(
    paths: Sequence[str] = CANONICAL_ARTIFACTS,
) -> VersionManifest:
    """Builds the reproducibility manifest over the canonical artifacts.

    Args:
        paths: Artifact paths in declared order.

    Returns:
        A validated manifest carrying per-artifact and combined digests.

    Raises:
        VersioningError: If any declared artifact is missing or unreadable.
    """
    artifacts = [_digest_file(Path(item)) for item in paths]
    return VersionManifest(
        combined_sha256=combined_digest(artifacts),
        artifact_count=len(artifacts),
        artifacts=artifacts,
    )


def write_version_manifest(path: Path, manifest: VersionManifest) -> None:
    """Writes the manifest atomically with stable key ordering.

    Args:
        path: Destination JSON path.
        manifest: Validated manifest to persist.
    """
    write_json_atomic(path, manifest.model_dump(mode="json"))


def run_version_artifacts(output_path: Path | None) -> int:
    """CLI implementation for the terminal ``version-artifacts`` DVC stage.

    Args:
        output_path: Manifest destination, or ``None`` for the default location.

    Returns:
        Process exit code: 0 on success, 1 on a fail-closed manifest error.
    """
    dest = output_path or Path("data/versioned/version_manifest.json")
    try:
        manifest = build_version_manifest()
        write_version_manifest(dest, manifest)
    except VersioningError as exc:
        print(f"[BLOCK] Artifact versioning failed: {exc}")
        return 1
    print(f"[SUCCESS] Versioned {manifest.artifact_count} artifacts")
    print(f"[DIGEST] combined_sha256={manifest.combined_sha256}")
    print(f"[REPORT] Version manifest written to: {dest}")
    return 0


def add_versioning_subparsers(subparsers: Any) -> None:
    """Registers the ``version-artifacts`` subcommand on the pipeline CLI.

    Args:
        subparsers: Argparse subparser action from :func:`build_parser`.
    """
    parser = subparsers.add_parser(
        "version-artifacts",
        help="Write the SHA-256 artifact version manifest (P2-F10, ADR-015)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Manifest destination (defaults to data/versioned/version_manifest.json)",
    )
    parser.set_defaults(func=lambda args: run_version_artifacts(args.output_path))
