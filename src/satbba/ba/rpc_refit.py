"""RPC refit from original RPC + solved affine corrections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from satbba.ba.model import ImageNorm, affine_delta
from satbba.io.rpc_accel import project_batch


@dataclass(slots=True)
class RPCRefitConfig:
    """Configuration for RPC refit sampling."""

    grid_size: int = 10


@dataclass(slots=True)
class RPCRefitResult:
    """Refit result payload."""

    image_id: int
    rmse_pixel: float
    coeffs: dict[str, float]


def _sample_world_grid(image: object, grid_size: int) -> tuple[object, object, object]:
    import numpy as np  # type: ignore

    cs = np.linspace(0, max(int(image.width) - 1, 1), grid_size)
    rs = np.linspace(0, max(int(image.height) - 1, 1), grid_size)
    hs = np.linspace(-100.0, 100.0, grid_size)

    c_grid, r_grid, h_grid = np.meshgrid(cs, rs, hs, indexing="ij")
    lon = np.empty_like(c_grid, dtype=float)
    lat = np.empty_like(c_grid, dtype=float)

    it = np.nditer(c_grid, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        lo, la = image.rpc.localize(float(c_grid[idx]), float(r_grid[idx]), float(h_grid[idx]))
        lon[idx], lat[idx] = lo, la
        it.iternext()
    return lon, lat, h_grid


def refit_rpc(
    image: object,
    affine_params: tuple[float, float, float, float, float, float],
    cfg: RPCRefitConfig,
) -> RPCRefitResult:
    """Fit compact correction model approximating RPC+affine output.

    This implementation samples AOI-consistent 3D points by localizing image-grid
    coordinates, then regresses correction terms that approximate
    `RPC(X) + affine(RPC(X))`.
    """

    import numpy as np  # type: ignore

    width, height = int(image.width), int(image.height)
    norm = ImageNorm(
        c0=width / 2.0,
        r0=height / 2.0,
        sc=max(width / 2.0, 1.0),
        sr=max(height / 2.0, 1.0),
    )

    lon, lat, h = _sample_world_grid(image, cfg.grid_size)

    batch = project_batch(image.rpc, lon, lat, h)
    dc, dr = affine_delta(batch.cols, batch.rows, affine_params, norm)
    target_c = batch.cols + dc
    target_r = batch.rows + dr

    X = np.stack(
        [
            np.ones(target_c.size),
            ((batch.cols - norm.c0) / norm.sc).reshape(-1),
            ((batch.rows - norm.r0) / norm.sr).reshape(-1),
        ],
        axis=1,
    )
    yc = (target_c - batch.cols).reshape(-1)
    yr = (target_r - batch.rows).reshape(-1)

    beta_c, *_ = np.linalg.lstsq(X, yc, rcond=None)
    beta_r, *_ = np.linalg.lstsq(X, yr, rcond=None)
    pred_c = X @ beta_c
    pred_r = X @ beta_r
    rmse = float(np.sqrt(np.mean((yc - pred_c) ** 2 + (yr - pred_r) ** 2)))

    coeffs = {
        "a0": float(beta_c[0]),
        "a1": float(beta_c[1]),
        "a2": float(beta_c[2]),
        "b0": float(beta_r[0]),
        "b1": float(beta_r[1]),
        "b2": float(beta_r[2]),
    }
    image_idx = int(image.image_id) if str(image.image_id).isdigit() else 0
    return RPCRefitResult(image_id=image_idx, rmse_pixel=rmse, coeffs=coeffs)


def export_refined_rpc(result: RPCRefitResult, output_dir: Path) -> Path:
    """Write refined RPC sidecar file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"image_{result.image_id:04d}.rpc"
    payload = {
        "image_id": result.image_id,
        "rmse_pixel": result.rmse_pixel,
        "coeffs": result.coeffs,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
