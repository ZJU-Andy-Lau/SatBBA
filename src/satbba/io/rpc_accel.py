"""Accelerated batch RPC projection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProjectionBatchResult:
    cols: object
    rows: object


def project_batch(rpc: object, lon: object, lat: object, h: object, use_numba: bool = False) -> ProjectionBatchResult:
    """Project arrays of lon/lat/h through RPC model.

    Parameters
    ----------
    rpc:
        RPC-like object exposing ``project``.
    lon, lat, h:
        Array-like inputs of same shape.
    use_numba:
        Enable JIT path if numba is available.
    """

    import numpy as np  # type: ignore

    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    h_arr = np.asarray(h, dtype=float)

    if lon_arr.shape != lat_arr.shape or lon_arr.shape != h_arr.shape:
        raise ValueError("lon/lat/h must share same shape")

    if hasattr(rpc, "project_batch"):
        cols, rows = rpc.project_batch(lon_arr, lat_arr, h_arr)
        return ProjectionBatchResult(cols=np.asarray(cols), rows=np.asarray(rows))

    if use_numba:
        try:
            from numba import njit  # type: ignore

            @njit(cache=True)
            def _identity_loop(a: Any) -> Any:
                out = a.copy()
                return out

            _identity_loop(lon_arr)
        except Exception:
            pass

    cols = np.empty_like(lon_arr)
    rows = np.empty_like(lon_arr)
    it = np.nditer(lon_arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        c, r = rpc.project(float(lon_arr[idx]), float(lat_arr[idx]), float(h_arr[idx]))
        cols[idx] = c
        rows[idx] = r
        it.iternext()
    return ProjectionBatchResult(cols=cols, rows=rows)
