"""Unified dataset loader for phase-3 track/triangulation steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from satbba.dataio.matches_loader import PairMatchRecord, load_pair_matches
from satbba.io.catalog import scan_image_catalog
from satbba.models.dataset import ImageData


@dataclass(slots=True)
class MatchingDataset:
    """Standardized access object for phase-3 inputs."""

    images: list[ImageData]
    pairs: list[tuple[int, int]]
    pair_matches: dict[tuple[int, int], PairMatchRecord]


def load_matching_dataset(image_dir: Path, rpc_dir: Path | None, output_dir: Path) -> MatchingDataset:
    """Load images/RPC and phase-2 matching artifacts into one object."""

    images = scan_image_catalog(image_dir, rpc_dir=rpc_dir)
    pairs_json = output_dir / "pairs.json"
    matches_dir = output_dir / "matches"
    pair_records = load_pair_matches(matches_dir=matches_dir, pairs_json_path=pairs_json)
    pair_dict = {(rec.pair.image_i, rec.pair.image_j): rec for rec in pair_records}
    return MatchingDataset(images=images, pairs=list(pair_dict.keys()), pair_matches=pair_dict)
