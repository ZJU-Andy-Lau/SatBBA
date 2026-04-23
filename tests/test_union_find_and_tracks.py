from pathlib import Path

from satbba.dataio.dataset_loader import MatchingDataset
from satbba.dataio.matches_loader import ImagePair, PairMatchRecord
from satbba.models.common import PairwiseMatch
from satbba.models.dataset import ImageData
from satbba.tracking.track_builder import build_tracks


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


def _images(n: int) -> list[ImageData]:
    return [ImageData(str(i), Path(f"{i}.tif"), 100, 100, _RPC()) for i in range(n)]


def test_union_merge_and_conflict_resolution() -> None:
    pair01 = PairMatchRecord(
        pair=ImagePair(0, 1),
        match=PairwiseMatch(
            image_id_a="0",
            image_id_b="1",
            points_a=[(10.0, 10.0), (10.2, 10.1)],
            points_b=[(20.0, 20.0), (20.1, 20.1)],
            confidence=[0.9, 0.2],
        ),
    )
    pair12 = PairMatchRecord(
        pair=ImagePair(1, 2),
        match=PairwiseMatch(
            image_id_a="1",
            image_id_b="2",
            points_a=[(20.0, 20.0)],
            points_b=[(30.0, 30.0)],
            confidence=[0.8],
        ),
    )
    ds = MatchingDataset(images=_images(3), pairs=[(0, 1), (1, 2)], pair_matches={(0, 1): pair01, (1, 2): pair12})

    out = build_tracks(ds, quantization=0.5)
    assert len(out.tracks) == 1
    track = out.tracks[0]
    assert len(track.observations) == 3
    assert len({o.image_id for o in track.observations}) == 3
