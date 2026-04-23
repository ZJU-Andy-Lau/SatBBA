import pytest
from pathlib import Path

from satbba.ba.dataset_loader import BADataset, BAObservation, BAPointInit
from satbba.ba.residuals import BAResidualConfig, build_jac_sparsity, residual_vector
from satbba.ba.state import BAState
from satbba.models.dataset import ImageData


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + 0.1 * h, lat + 0.1 * h

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col - 0.1 * h, row - 0.1 * h


def _dataset() -> BADataset:
    images = [ImageData("0", Path("0.tif"), 100, 100, _RPC()), ImageData("1", Path("1.tif"), 100, 100, _RPC())]
    pts = [BAPointInit(10, 120.0, 30.0, 100.0, 2, 0.5)]
    obs = [
        BAObservation(0, 10, 0, 130.0, 40.0),
        BAObservation(1, 10, 1, 130.0, 40.0),
    ]
    return BADataset(images=images, observations=obs, points_init=pts, reference_image_id=0, metadata={})


def test_residual_and_sparsity_shape() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    st = BAState.from_dataset(_dataset())
    x = st.pack()
    r = residual_vector(x, st, BAResidualConfig())
    S = build_jac_sparsity(st)
    assert r.shape[0] == S.shape[0]
    assert S.shape[1] == st.n_params
