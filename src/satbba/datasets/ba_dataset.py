"""BA input dataset structures for phase-4 consumption."""

from __future__ import annotations

from dataclasses import dataclass, field

from satbba.models.dataset import ImageData
from satbba.models.tracks import Observation, Track, TriangulatedPoint


@dataclass(slots=True)
class BADataset:
    """Standardized bundle-adjustment input dataset."""

    images: list[ImageData]
    tracks: list[Track]
    observations: list[Observation]
    points_init: list[TriangulatedPoint]
    reference_image_id: int
    h_ref: float
    metadata: dict[str, object] = field(default_factory=dict)
