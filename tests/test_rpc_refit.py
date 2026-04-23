from pathlib import Path

import pytest

from satbba.ba.rpc_refit import RPCRefitConfig, export_refined_rpc, refit_rpc
from satbba.models.dataset import ImageData


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return 100.0 + 1000.0 * lon + 10.0 * h, 200.0 + 1000.0 * lat + 5.0 * h


def test_rpc_refit_and_export(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    img = ImageData("0", Path("0.tif"), 1000, 1000, _RPC())
    res = refit_rpc(img, (0.1, 0.01, 0.02, -0.1, 0.03, -0.01), RPCRefitConfig(grid_size=4))
    p = export_refined_rpc(res, tmp_path)
    assert p.exists()
    assert res.rmse_pixel >= 0.0
