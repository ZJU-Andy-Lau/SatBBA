from pathlib import Path

import pytest

from satbba.ba.rpc_refit import RPCRefitConfig, export_refined_rpc, refit_rpc
from satbba.models.dataset import ImageData


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return 100.0 + 1000.0 * lon + 10.0 * h, 200.0 + 1000.0 * lat + 5.0 * h

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return (col - 100.0 - 10.0 * h) / 1000.0, (row - 200.0 - 5.0 * h) / 1000.0


def test_rpc_refit_and_export(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    img = ImageData("IMG_A_001", Path("0.tif"), 1000, 1000, _RPC())
    res = refit_rpc(img, (0.1, 0.01, 0.02, -0.1, 0.03, -0.01), RPCRefitConfig(grid_size=4))
    p = export_refined_rpc(res, tmp_path)
    assert p.exists()
    assert p.name == "image_IMG_A_001.rpc"
    assert res.rmse_pixel >= 0.0


def test_rpc_refit_keeps_unique_string_ids(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    img1 = ImageData("IMG_A", Path("a.tif"), 100, 100, _RPC())
    img2 = ImageData("IMG_B", Path("b.tif"), 100, 100, _RPC())
    p1 = export_refined_rpc(refit_rpc(img1, (0, 0, 0, 0, 0, 0), RPCRefitConfig(grid_size=3)), tmp_path)
    p2 = export_refined_rpc(refit_rpc(img2, (0, 0, 0, 0, 0, 0), RPCRefitConfig(grid_size=3)), tmp_path)
    assert p1 != p2
    assert p1.exists() and p2.exists()
