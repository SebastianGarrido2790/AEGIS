"""Feature engineering and data ingestion pipelines for AEGIS (Tier 1).

Provides modular feature transformers, data ingestion routines, and dataset preparation
pipelines conforming to ADR-011, ADR-012, and ADR-019.
"""

from aegis.pipelines.feature.ingest import (
    clean_and_merge_fremtpl2,
    fetch_fremtpl2_data,
    ingest_fremtpl2_pipeline,
)

__all__ = [
    "clean_and_merge_fremtpl2",
    "fetch_fremtpl2_data",
    "ingest_fremtpl2_pipeline",
]
