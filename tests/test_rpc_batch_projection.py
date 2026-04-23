import pytest

from satbba.io.rpc_accel import project_batch


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + h, lat - h


def test_batch_projection() -> None:
    np = pytest.importorskip("numpy")
    lon = np.array([1.0, 2.0, 3.0])
    lat = np.array([4.0, 5.0, 6.0])
    h = np.array([0.5, 0.5, 0.5])
    out = project_batch(_RPC(), lon, lat, h)
    assert out.cols.shape == (3,)
    assert abs(out.cols[0] - 1.5) < 1e-8
