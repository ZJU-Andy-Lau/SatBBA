"""Post-triangulation quality filtering and statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from satbba.models.tracks import TriangulatedPoint


@dataclass(slots=True)
class TriangulationQualityConfig:
    reproj_error_threshold: float = 5.0
    min_success_views: int = 2
    h_min: float = -1000.0
    h_max: float = 10000.0


@dataclass(slots=True)
class TriangulationQualityStats:
    success_count: int
    rejected_count: int
    reject_reasons: dict[str, int]


def filter_triangulated_points(
    points: list[TriangulatedPoint], cfg: TriangulationQualityConfig
) -> tuple[list[TriangulatedPoint], TriangulationQualityStats]:
    """Filter triangulated points by reprojection/height/success criteria."""

    kept: list[TriangulatedPoint] = []
    reasons: Counter[str] = Counter()

    for p in points:
        if not p.triangulation_success:
            reasons["optimizer_failed"] += 1
            continue
        if p.n_views < cfg.min_success_views:
            reasons["insufficient_views"] += 1
            continue
        if p.mean_reproj_error > cfg.reproj_error_threshold:
            reasons["high_reprojection_error"] += 1
            continue
        if p.h_init < cfg.h_min or p.h_init > cfg.h_max:
            reasons["height_out_of_range"] += 1
            continue
        kept.append(p)

    return kept, TriangulationQualityStats(
        success_count=len(kept), rejected_count=len(points) - len(kept), reject_reasons=dict(reasons)
    )
