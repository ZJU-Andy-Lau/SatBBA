from pathlib import Path

import satbba.core.matching_pipeline as mp
from satbba.config.settings import AppConfig, IOConfig, LoggingConfig, MatcherConfig, MatchingConfig, RuntimeConfig
from satbba.models.common import PairwiseMatch
from satbba.models.dataset import ImageData


class _FakeRPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


class _FakeMatcher:
    config = MatcherConfig()

    def validate_runtime(self) -> None:
        return None

    def match(self, image_a_path: str, image_b_path: str) -> PairwiseMatch:
        return PairwiseMatch(
            image_id_a="a",
            image_id_b="b",
            points_a=[(1.0, 2.0), (3.0, 4.0)],
            points_b=[(1.1, 2.1), (3.1, 4.1)],
            confidence=[0.9, 0.8],
        )


def test_match_pair_with_monkeypatched_filters(monkeypatch) -> None:
    cfg = AppConfig(
        io=IOConfig(image_dir=Path("."), output_dir=Path("outputs")),
        matching=MatchingConfig(matcher=MatcherConfig(name="sift"), geo_distance_threshold=100.0),
        logging=LoggingConfig(),
        runtime=RuntimeConfig(),
    )

    image_i = ImageData("i", Path("i.tif"), 100, 100, _FakeRPC())
    image_j = ImageData("j", Path("j.tif"), 100, 100, _FakeRPC())

    monkeypatch.setattr(mp, "geometric_filter", lambda a, b, threshold: [True, False])
    monkeypatch.setattr(mp, "geo_distance_filter", lambda *args, **kwargs: [True])

    result = mp.match_pair(0, 1, image_i, image_j, _FakeMatcher(), cfg)
    assert result.raw_count == 2
    assert result.geometric_inliers == 1
    assert result.geo_filtered == 1
