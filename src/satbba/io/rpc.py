"""RPC loading and internal model wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satbba.exceptions import DataModelError


@dataclass(slots=True)
class RPCModel:
    """Internal RPC wrapper with project/localize APIs.

    Parameters
    ----------
    rpc_obj:
        Backend object exposing ``projection`` and ``localization`` methods.
    """

    rpc_obj: Any

    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        """Project geodetic coordinate to image pixel (col, row)."""

        col, row = self.rpc_obj.projection(lon, lat, h)
        return float(col), float(row)

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        """Localize image pixel to geodetic coordinate (lon, lat)."""

        lon, lat = self.rpc_obj.localization(col, row, h)
        return float(lon), float(lat)


def _import_rasterio() -> Any:
    try:
        import rasterio  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guarded by runtime
        raise DataModelError("rasterio is required to read RPC from GeoTIFF") from exc
    return rasterio


def _import_rpcm() -> Any:
    try:
        import rpcm  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guarded by runtime
        raise DataModelError("rpcm is required for sidecar RPC parsing") from exc
    return rpcm


def _parse_rpc_tags(rpc_tags: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in rpc_tags.items():
        if value is None:
            continue
        stripped = value.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            numbers = [float(x) for x in stripped[1:-1].replace(",", " ").split()]
            parsed[key] = numbers
            continue
        try:
            parsed[key] = float(stripped)
        except ValueError:
            parsed[key] = stripped
    return parsed


def load_rpc_model(image_path: Path, sidecar_dir: Path | None = None) -> RPCModel:
    """Load RPC model from embedded tags or sidecar files.

    Search order:
    1. GeoTIFF RPC namespace tags
    2. Sidecar files near image (`.RPB`, `.xml`, `.txt`, `.json`)
    """

    rasterio = _import_rasterio()
    rpcm = _import_rpcm()

    with rasterio.open(image_path) as src:
        rpc_tags = src.tags(ns="RPC")

    if rpc_tags:
        rpc_dict = _parse_rpc_tags(rpc_tags)
        return RPCModel(rpcm.RPCModel(rpc_dict))

    lookup_dir = sidecar_dir or image_path.parent
    candidates = [
        lookup_dir / f"{image_path.stem}.RPB",
        lookup_dir / f"{image_path.stem}.rpb",
        lookup_dir / f"{image_path.stem}.xml",
        lookup_dir / f"{image_path.stem}.txt",
        lookup_dir / f"{image_path.stem}.json",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".json":
            content = json.loads(candidate.read_text(encoding="utf-8"))
            return RPCModel(rpcm.RPCModel(content))
        return RPCModel(rpcm.rpc_from_rpc_file(str(candidate)))

    raise DataModelError(f"RPC metadata not found for image: {image_path}")
