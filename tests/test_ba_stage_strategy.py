import pytest
from pathlib import Path

from satbba.ba.dataset_loader import BADataset, BAObservation, BAPointInit
from satbba.ba.solver import BAConfig, run_bundle_adjustment
from satbba.models.dataset import ImageData


class LinearRPC:
    def __init__(self, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy

    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + self.dx, lat + self.dy

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col - self.dx, row - self.dy


def _dataset() -> BADataset:
    images = [ImageData("0", Path("0.tif"), 100, 100, LinearRPC(0, 0)), ImageData("1", Path("1.tif"), 100, 100, LinearRPC(2, -1))]
    pts = [BAPointInit(track_id=1, lon=100.0, lat=30.0, h=0.0, n_views=2, mean_reproj_error=0.0)]
    obs = [BAObservation(0, 1, 0, 100.0, 30.0), BAObservation(1, 1, 1, 102.0, 29.0)]
    return BADataset(images=images, observations=obs, points_init=pts, reference_image_id=0, metadata={})


def test_translation_stage_reports_exist() -> None:
    try:
        import scipy  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        pytest.skip("scipy/numpy missing")

    out = run_bundle_adjustment(_dataset(), BAConfig(max_nfev=20, checkpoint_interval=0))
    stages = [r.get("stage") for r in out.reports]
    assert "translation" in stages
    assert "affine" in stages
    assert "refine" in stages
