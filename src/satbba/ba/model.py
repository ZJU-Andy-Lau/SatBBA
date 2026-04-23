"""BA observation model: RPC projection + image affine correction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ImageNorm:
    """Image coordinate normalization parameters."""

    c0: float
    r0: float
    sc: float
    sr: float


def affine_delta(col_rpc: float, row_rpc: float, params: tuple[float, float, float, float, float, float], norm: ImageNorm) -> tuple[float, float]:
    """Compute affine correction delta in image space."""

    a0, a1, a2, b0, b1, b2 = params
    cn = (col_rpc - norm.c0) / norm.sc
    rn = (row_rpc - norm.r0) / norm.sr
    dc = a0 + a1 * cn + a2 * rn
    dr = b0 + b1 * cn + b2 * rn
    return dc, dr


def project_with_jacobian_numeric(rpc: object, lon: float, lat: float, h: float, eps: float = 1e-6) -> tuple[float, float, list[list[float]]]:
    """Project point and return numerical 2x3 Jacobian wrt lon/lat/h."""

    col, row = rpc.project(lon, lat, h)

    def _proj(dl: float, da: float, dh: float) -> tuple[float, float]:
        return rpc.project(lon + dl, lat + da, h + dh)

    c_lon, r_lon = _proj(eps, 0.0, 0.0)
    c_lat, r_lat = _proj(0.0, eps, 0.0)
    c_h, r_h = _proj(0.0, 0.0, eps)

    jac = [
        [(c_lon - col) / eps, (c_lat - col) / eps, (c_h - col) / eps],
        [(r_lon - row) / eps, (r_lat - row) / eps, (r_h - row) / eps],
    ]
    return float(col), float(row), jac
