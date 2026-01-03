"""Utility helpers for downloading artifacts from Google Cloud Storage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

try:
    from google.cloud import storage
except Exception:  # pragma: no cover - optional dependency
    storage = None  # type: ignore

logger = logging.getLogger(__name__)


class GCSError(RuntimeError):
    """Raised when a GCS download fails."""


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    without_scheme = uri[len("gs://") :]
    bucket, _, blob = without_scheme.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def download_from_gcs(uri: str, destination: Path) -> Path:
    """Download a blob from GCS into *destination* if possible."""
    if storage is None:
        raise GCSError("google-cloud-storage is not installed or failed to import")

    bucket_name, blob_name = parse_gcs_uri(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        logger.info("Downloading %s -> %s", uri, destination)
        blob.download_to_filename(str(destination))
        return destination
    except Exception as exc:  # pragma: no cover - network failure
        raise GCSError(f"Failed to download {uri}: {exc}") from exc


__all__ = ["download_from_gcs", "parse_gcs_uri", "GCSError"]
