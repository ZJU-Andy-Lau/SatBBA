from pathlib import Path

import pytest

from satbba.models.dataset import ImageData
from satbba.models.tracks import Observation, Track, TriangulatedPoint
from satbba.triangulation.quality import TriangulationQualityConfig, filter_triangulated_points
from satbba.triangulation.triangulator import TriangulationConfig, triangulate_track


class LinearRPC:
    def __init__(self, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy

    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + self.dx + 0.01 * h, lat + self.dy + 0.02 * h

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col - self.dx - 0.01 * h, row - self.dy - 0.02 * h


def test_single_track_triangulation_synthetic() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("scipy missing")

    lon, lat, h = 100.0, 30.0, 50.0
    images = [
        ImageData("0", Path("0.tif"), 100, 100, LinearRPC(0.0, 0.0)),
        ImageData("1", Path("1.tif"), 100, 100, LinearRPC(5.0, 2.0)),
        ImageData("2", Path("2.tif"), 100, 100, LinearRPC(-3.0, 4.0)),
    ]

    obs = []
    for idx, img in enumerate(images):
        c, r = img.rpc.project(lon, lat, h)
        obs.append(Observation(obs_id=idx, image_id=idx, col=c, row=r, score=1.0))

    track = Track(track_id=0, observations=obs)
    p = triangulate_track(track, images, h_ref=40.0, reference_image_id=0, cfg=TriangulationConfig())
    assert p.triangulation_success
    assert p.mean_reproj_error < 1e-2


def test_quality_filter_rejects_bad_points() -> None:
    points = [
        TriangulatedPoint(0, 0, 0, 0, 3, 1.0, True),
        TriangulatedPoint(1, 0, 0, 0, 1, 1.0, True),
        TriangulatedPoint(2, 0, 0, 20000, 3, 1.0, True),
        TriangulatedPoint(3, 0, 0, 0, 3, 20.0, True),
        TriangulatedPoint(4, 0, 0, 0, 3, 1.0, False),
    ]
    kept, stats = filter_triangulated_points(points, TriangulationQualityConfig())
    assert len(kept) == 1
    assert stats.rejected_count == 4
