"""Track filtering and reference connectivity pruning."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

from satbba.models.tracks import Track

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackFilterConfig:
    min_views: int = 2
    min_score: float | None = None
    max_track_length: int | None = None
    max_tracks_per_grid_cell: int = 50
    grid_rows: int = 20
    grid_cols: int = 20
    drop_disconnected_from_reference: bool = True


@dataclass(slots=True)
class TrackFilterStats:
    raw_tracks: int
    filtered_tracks: int
    average_length: float
    length_histogram: dict[int, int]
    tracks_per_image: dict[int, int]


def _track_mean_score(track: Track) -> float:
    scores = [o.score for o in track.observations if o.score is not None]
    return float(sum(scores) / len(scores)) if scores else 0.0


def _image_graph(tracks: list[Track]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for t in tracks:
        ids = sorted({o.image_id for o in t.observations})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                graph[ids[i]].add(ids[j])
                graph[ids[j]].add(ids[i])
    return graph


def _connected_component(graph: dict[int, set[int]], start: int) -> set[int]:
    if start not in graph:
        return {start}
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in graph.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def filter_tracks(tracks: list[Track], cfg: TrackFilterConfig, reference_image_id: int) -> tuple[list[Track], TrackFilterStats]:
    """Apply phase-3 track filtering rules."""

    kept: list[Track] = []
    for t in tracks:
        if len(t.observations) < cfg.min_views:
            continue
        if cfg.max_track_length is not None and len(t.observations) > cfg.max_track_length:
            continue
        if cfg.min_score is not None and _track_mean_score(t) < cfg.min_score:
            continue
        kept.append(t)

    if cfg.drop_disconnected_from_reference:
        graph = _image_graph(kept)
        component = _connected_component(graph, reference_image_id)
        kept = [t for t in kept if any(o.image_id in component for o in t.observations)]

    counts = Counter(len(t.observations) for t in kept)
    tracks_per_image: dict[int, int] = Counter()
    for t in kept:
        for o in t.observations:
            tracks_per_image[o.image_id] += 1

    avg = float(sum(len(t.observations) for t in kept) / len(kept)) if kept else 0.0
    stats = TrackFilterStats(
        raw_tracks=len(tracks),
        filtered_tracks=len(kept),
        average_length=avg,
        length_histogram=dict(sorted(counts.items())),
        tracks_per_image=dict(sorted(tracks_per_image.items())),
    )
    LOGGER.info("Track filtering: raw=%d filtered=%d avg_len=%.2f", stats.raw_tracks, stats.filtered_tracks, stats.average_length)
    return kept, stats
