import pytest
from pathlib import Path

from satbba.ba.dataset_loader import BADataset, BAObservation, BAPointInit
from satbba.ba.state import BAState
from satbba.models.dataset import ImageData


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


def _dataset() -> BADataset:
    images = [ImageData("0", Path("0.tif"), 100, 100, _RPC()), ImageData("1", Path("1.tif"), 100, 100, _RPC())]
    obs = [BAObservation(0, 10, 0, 1.0, 2.0), BAObservation(1, 10, 1, 1.1, 2.1)]
    pts = [BAPointInit(10, 120.0, 30.0, 100.0, 2, 0.5)]
    return BADataset(images=images, observations=obs, points_init=pts, reference_image_id=0, metadata={})


def test_pack_unpack_point() -> None:
    pytest.importorskip("numpy")
    st = BAState.from_dataset(_dataset())
    x = st.pack()
    lon, lat, h = st.unpack_point(x, 10)
    assert abs(lon - 120.0) < 1e-8
    assert abs(lat - 30.0) < 1e-8
    assert abs(h - 100.0) < 1e-8
