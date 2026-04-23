"""Load phase-2 pairwise match artifacts from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satbba.models.common import PairwiseMatch


@dataclass(slots=True)
class ImagePair:
    """Image pair key."""

    image_i: int
    image_j: int


@dataclass(slots=True)
class PairMatchRecord:
    """Pair-level matching data loaded from phase-2 outputs."""

    pair: ImagePair
    match: PairwiseMatch


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required to load .npz match files") from exc
    return np


def load_pair_matches(matches_dir: Path, pairs_json_path: Path) -> list[PairMatchRecord]:
    """Load pair matches from `pairs.json` and `matches/*.npz`."""

    np = _import_numpy()
    metadata = json.loads(pairs_json_path.read_text(encoding="utf-8"))

    records: list[PairMatchRecord] = []
    for item in metadata:
        i, j = int(item["pair"][0]), int(item["pair"][1])
        npz_path = matches_dir / item["file"]
        data = np.load(npz_path)
        pts_i = data["pts_i"].tolist()
        pts_j = data["pts_j"].tolist()
        scores = data["scores"].tolist()
        records.append(
            PairMatchRecord(
                pair=ImagePair(image_i=i, image_j=j),
                match=PairwiseMatch(
                    image_id_a=str(i),
                    image_id_b=str(j),
                    points_a=[(float(x), float(y)) for x, y in pts_i],
                    points_b=[(float(x), float(y)) for x, y in pts_j],
                    confidence=[float(s) for s in scores],
                ),
            )
        )
    return records
