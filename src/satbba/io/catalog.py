"""Image/RPC catalog loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from satbba.io.rpc import load_rpc_model
from satbba.models.dataset import ImageData

LOGGER = logging.getLogger(__name__)


def _import_rasterio() -> Any:
    try:
        import rasterio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("rasterio is required for image catalog scanning") from exc
    return rasterio


def scan_image_catalog(image_dir: Path, rpc_dir: Path | None = None) -> list[ImageData]:
    """Scan GeoTIFF directory and attach loaded RPC model for each image."""

    rasterio = _import_rasterio()
    records: list[ImageData] = []

    for image_path in sorted(image_dir.glob("*.tif")):
        with rasterio.open(image_path) as src:
            width, height = int(src.width), int(src.height)
        rpc = load_rpc_model(image_path, sidecar_dir=rpc_dir)
        records.append(
            ImageData(
                image_id=image_path.stem,
                image_path=image_path,
                width=width,
                height=height,
                rpc=rpc,
            )
        )

    LOGGER.info("Loaded %d images from %s", len(records), image_dir)
    return records
