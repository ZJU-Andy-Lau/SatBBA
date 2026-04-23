"""Dataset-level models for image, RPC and footprint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RPCProtocol(Protocol):
    """Protocol contract for internal RPC model."""

    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        """Project world coordinate to image coordinate (col, row)."""

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        """Back-project image coordinate and height to (lon, lat)."""


@dataclass(slots=True)
class ImageData:
    """Image input record used by overlap and matching pipeline."""

    image_id: str
    image_path: Path
    width: int
    height: int
    rpc: RPCProtocol
    footprint: object | None = None
