#!/usr/bin/env python
"""Human-gated promotion and rollback for the AEGIS Tier 1 registry (P2-F09).

This is the only entrypoint that may assign the ``approved`` alias (ADR-014,
P2-D11a). The pipeline cannot self-approve: a reviewer identity, the SHA-256 of the
calibration report that reviewer inspected, and a written rationale are mandatory.

Usage:
    uv run python scripts/promote_model.py promote \\
        --version 2 --reviewer-id <human> --calibration-hash <sha256> \\
        --review-note "<rationale>"

    uv run python scripts/promote_model.py rollback \\
        --version 1 --reviewer-id <human> --reason "<why we are rolling back>"
"""

from __future__ import annotations

import argparse
import sys

from aegis.config.loader import get_config
from aegis.pipelines.registry import promote_to_approved, rollback_approved
from aegis.schemas.registry import ModelPromotionRequest
from aegis.utils.exceptions import ModelNotFoundError, PromotionGateError


def _build_parser() -> argparse.ArgumentParser:
    """Builds the promote/rollback argument parser."""
    parser = argparse.ArgumentParser(
        description="AEGIS Tier 1 registry: human-gated promotion and rollback (P2-F09)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote", help="Assign 'approved' to a candidate")
    promote.add_argument("--version", dest="target_version", required=True)
    promote.add_argument("--reviewer-id", required=True)
    promote.add_argument("--calibration-hash", required=True)
    promote.add_argument("--review-note", required=True)
    promote.add_argument("--model-name", default=None)
    promote.add_argument("--calibration-report", default=None)

    rollback = subparsers.add_parser("rollback", help="Re-point 'approved' at a prior version")
    rollback.add_argument("--version", dest="target_version", required=True)
    rollback.add_argument("--reviewer-id", required=True)
    rollback.add_argument("--reason", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Executes the requested registry action and prints its audit record."""
    args = _build_parser().parse_args(argv)

    if args.command == "promote":
        request = ModelPromotionRequest(
            model_name=args.model_name or get_config().tier1_ml.registry.model_name,
            target_version=str(args.target_version),
            reviewer_id=args.reviewer_id,
            calibration_report_hash=args.calibration_hash,
            review_notes=args.review_note,
        )
        record = promote_to_approved(
            request,
            calibration_report_path=args.calibration_report,
        )
    else:
        record = rollback_approved(
            target_version=str(args.target_version),
            reviewer_id=args.reviewer_id,
            reason=args.reason,
        )

    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PromotionGateError, ModelNotFoundError) as exc:
        print(f"[BLOCK] {exc}", file=sys.stderr)
        sys.exit(1)
