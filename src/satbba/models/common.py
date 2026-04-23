"""Core domain data models for SatBBA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    image_path: Path
    rpc_path: Path | None = None


@dataclass(slots=True)
class KeypointObservation:
    image_id: str
    row: float
    col: float


@dataclass(slots=True)
class PairwiseMatch:
    image_id_a: str
    image_id_b: str
    points_a: list[tuple[float, float]] = field(default_factory=list)
    points_b: list[tuple[float, float]] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


@dataclass(slots=True)
class Track:
    track_id: str
    observations: list[KeypointObservation]


@dataclass(slots=True)
class TiePoint3D:
    track_id: str
    lon: float
    lat: float
    h: float


@dataclass(slots=True)
class AffineCorrection:
    image_id: str
    a11: float = 1.0
    a12: float = 0.0
    a13: float = 0.0
    a21: float = 0.0
    a22: float = 1.0
    a23: float = 0.0
