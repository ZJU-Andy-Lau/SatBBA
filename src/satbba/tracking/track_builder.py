"""Build multi-view tracks from pairwise matches."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from satbba.dataio.dataset_loader import MatchingDataset
from satbba.models.tracks import Observation, Track
from satbba.tracking.union_find import UnionFind

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrackBuildResult:
    """Track build outputs and stats."""

    tracks: list[Track]
    observations: list[Observation]
    total_edges: int


class ObservationIndexer:
    """Stable observation identity allocator using coordinate quantization."""

    def __init__(self, quantization: float = 0.25) -> None:
        self.quant = quantization
        self._mapping: dict[tuple[int, int, int], int] = {}
        self._next_id = 0

    def get_or_create(self, image_id: int, col: float, row: float) -> int:
        """Return stable obs_id for quantized coordinate key."""

        qc = int(round(col / self.quant))
        qr = int(round(row / self.quant))
        key = (image_id, qc, qr)
        if key not in self._mapping:
            self._mapping[key] = self._next_id
            self._next_id += 1
        return self._mapping[key]


def _resolve_duplicate_image_observations(track: Track) -> Track:
    """Ensure one observation per image by keeping highest-score entry."""

    best_by_image: dict[int, Observation] = {}
    for obs in track.observations:
        current = best_by_image.get(obs.image_id)
        current_score = current.score if current and current.score is not None else -1.0
        obs_score = obs.score if obs.score is not None else -1.0
        if current is None or obs_score > current_score:
            best_by_image[obs.image_id] = obs

    deduped = list(best_by_image.values())
    deduped.sort(key=lambda x: x.image_id)
    return Track(track_id=track.track_id, observations=deduped)


def build_tracks(dataset: MatchingDataset, quantization: float = 0.25) -> TrackBuildResult:
    """Build tracks from pairwise matches using Union-Find merging."""

    obs_indexer = ObservationIndexer(quantization=quantization)
    uf = UnionFind()
    obs_data: dict[int, Observation] = {}
    total_edges = 0

    for (i, j), pair_record in dataset.pair_matches.items():
        pts_i = pair_record.match.points_a
        pts_j = pair_record.match.points_b
        scores = pair_record.match.confidence or [1.0] * len(pts_i)

        for pti, ptj, score in zip(pts_i, pts_j, scores):
            oi = obs_indexer.get_or_create(i, pti[0], pti[1])
            oj = obs_indexer.get_or_create(j, ptj[0], ptj[1])
            uf.union(oi, oj)
            total_edges += 1

            obs_data[oi] = Observation(obs_id=oi, image_id=i, col=float(pti[0]), row=float(pti[1]), score=float(score))
            obs_data[oj] = Observation(obs_id=oj, image_id=j, col=float(ptj[0]), row=float(ptj[1]), score=float(score))

    groups: dict[int, list[Observation]] = defaultdict(list)
    for obs_id, obs in obs_data.items():
        root = uf.find(obs_id)
        groups[root].append(obs)

    tracks: list[Track] = []
    for track_id, (_, members) in enumerate(groups.items()):
        track = Track(track_id=track_id, observations=members)
        tracks.append(_resolve_duplicate_image_observations(track))

    all_observations = [obs for t in tracks for obs in t.observations]
    LOGGER.info("Track builder produced %d raw tracks from %d edges", len(tracks), total_edges)
    return TrackBuildResult(tracks=tracks, observations=all_observations, total_edges=total_edges)
