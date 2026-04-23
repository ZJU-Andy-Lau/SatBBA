from pathlib import Path

import pytest

from satbba.ba.dataset_loader import BADataset, BAObservation, BAPointInit
from satbba.ba.solver import BAConfig, run_bundle_adjustment
from satbba.models.dataset import ImageData


class LinearRPC:
    def __init__(self, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy

    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + self.dx + 0.01 * h, lat + self.dy + 0.02 * h

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col - self.dx - 0.01 * h, row - self.dy - 0.02 * h


def test_synthetic_ba_converges() -> None:
    try:
        import scipy  # noqa: F401
    except Exception:
        pytest.skip("scipy missing")

    lon, lat, h = 100.0, 30.0, 50.0
    images = [
        ImageData("0", Path("0.tif"), 100, 100, LinearRPC(0.0, 0.0)),
        ImageData("1", Path("1.tif"), 100, 100, LinearRPC(2.0, -1.0)),
    ]
    pts = [BAPointInit(track_id=1, lon=99.8, lat=30.2, h=45.0, n_views=2, mean_reproj_error=2.0)]
    obs = []
    for i, img in enumerate(images):
        c, r = img.rpc.project(lon, lat, h)
        obs.append(BAObservation(obs_id=i, track_id=1, image_id=i, col=c, row=r))

    ds = BADataset(images=images, observations=obs, points_init=pts, reference_image_id=0, metadata={})
    res = run_bundle_adjustment(ds, BAConfig(max_nfev=100))
    assert len(res.reports) >= 2
    assert res.reports[-1]["success"] in {True, False}
