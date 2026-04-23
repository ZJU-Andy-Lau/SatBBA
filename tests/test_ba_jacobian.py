from satbba.ba.model import project_with_jacobian_numeric


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + 2.0 * lat + 0.5 * h, 3.0 * lon - lat + 0.2 * h


def test_project_with_jacobian_numeric() -> None:
    c, r, j = project_with_jacobian_numeric(_RPC(), lon=1.0, lat=2.0, h=3.0, eps=1e-8)
    assert abs(c - 6.5) < 1e-6
    assert abs(r - 1.6) < 1e-6
    assert len(j) == 2 and len(j[0]) == 3
    assert abs(j[0][0] - 1.0) < 1e-3
    assert abs(j[0][1] - 2.0) < 1e-3
