"""Track-level non-linear triangulation with RPC projection residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from satbba.models.dataset import ImageData
from satbba.models.tracks import Track, TriangulatedPoint


def _import_scipy_opt() -> Any:
    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for triangulation optimization") from exc
    return least_squares


@dataclass(slots=True)
class TriangulationConfig:
    method: str = "trf"
    max_nfev: int = 100


def _seed_xyz(track: Track, images: list[ImageData], reference_image_id: int, h_ref: float) -> tuple[float, float, float]:
    """Build initialization from reference view if available."""

    ref_obs = next((o for o in track.observations if o.image_id == reference_image_id), None)
    obs = ref_obs if ref_obs is not None else track.observations[0]
    lon, lat = images[obs.image_id].rpc.localize(obs.col, obs.row, h_ref)
    return lon, lat, h_ref


def triangulate_track(
    track: Track,
    images: list[ImageData],
    h_ref: float,
    reference_image_id: int,
    cfg: TriangulationConfig,
) -> TriangulatedPoint:
    """Estimate initial 3D point for one track via non-linear least squares."""

    least_squares = _import_scipy_opt()

    if len(track.observations) < 2:
        return TriangulatedPoint(
            track_id=track.track_id,
            lon_init=0.0,
            lat_init=0.0,
            h_init=0.0,
            n_views=len(track.observations),
            mean_reproj_error=math.inf,
            triangulation_success=False,
            failure_reason="insufficient_views",
        )

    x0 = _seed_xyz(track, images, reference_image_id, h_ref)

    def residuals(x: list[float]) -> list[float]:
        lon, lat, h = float(x[0]), float(x[1]), float(x[2])
        res: list[float] = []
        for obs in track.observations:
            col_pred, row_pred = images[obs.image_id].rpc.project(lon, lat, h)
            res.append(col_pred - obs.col)
            res.append(row_pred - obs.row)
        return res

    try:
        out = least_squares(residuals, x0=list(x0), method=cfg.method, max_nfev=cfg.max_nfev)
    except Exception as exc:
        return TriangulatedPoint(
            track_id=track.track_id,
            lon_init=x0[0],
            lat_init=x0[1],
            h_init=x0[2],
            n_views=len(track.observations),
            mean_reproj_error=math.inf,
            triangulation_success=False,
            failure_reason=f"optimizer_error:{exc}",
        )

    if out.fun is None or len(out.fun) == 0:
        mean_err = math.inf
    else:
        sq = [float(v) ** 2 for v in out.fun]
        mean_err = float(math.sqrt(sum(sq) / len(sq)))

    return TriangulatedPoint(
        track_id=track.track_id,
        lon_init=float(out.x[0]),
        lat_init=float(out.x[1]),
        h_init=float(out.x[2]),
        n_views=len(track.observations),
        mean_reproj_error=mean_err,
        triangulation_success=bool(out.success),
        failure_reason=None if out.success else str(out.message),
    )
