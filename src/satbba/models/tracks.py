"""Track-level data models for phase-3 processing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Observation:
    """Single image-space observation in a track."""

    obs_id: int
    image_id: int
    col: float
    row: float
    score: float | None = None


@dataclass(slots=True)
class Track:
    """Multi-view tie-point track."""

    track_id: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return len(self.observations)


@dataclass(slots=True)
class TriangulatedPoint:
    """Initial 3D result for one track."""

    track_id: int
    lon_init: float
    lat_init: float
    h_init: float
    n_views: int
    mean_reproj_error: float
    triangulation_success: bool
    failure_reason: str | None = None
