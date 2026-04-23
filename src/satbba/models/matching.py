"""Matching-related domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Match:
    """Normalized match entity after pairwise filtering pipeline."""

    image_i: int
    image_j: int
    pt_i: tuple[float, float]
    pt_j: tuple[float, float]
    score: float = 1.0


@dataclass(slots=True)
class PairMatchResult:
    """Per-pair matching output ready for serialization."""

    image_i: int
    image_j: int
    matches: list[Match] = field(default_factory=list)
    raw_count: int = 0
    geometric_inliers: int = 0
    geo_filtered: int = 0
