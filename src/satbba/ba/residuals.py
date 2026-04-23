"""Residual and sparsity builders for BA optimization."""

from __future__ import annotations

from dataclasses import dataclass

from satbba.ba.dataset_loader import BADataset
from satbba.ba.model import ImageNorm, affine_delta
from satbba.ba.state import BAState


@dataclass(slots=True)
class BAResidualConfig:
    reproj_sigma: float = 1.0
    affine_reg_weight: float = 1e-4
    point_reg_weight: float = 1e-6


def build_image_norms(dataset: BADataset) -> dict[int, ImageNorm]:
    """Build per-image normalization constants."""

    norms: dict[int, ImageNorm] = {}
    for idx, img in enumerate(dataset.images):
        norms[idx] = ImageNorm(c0=img.width / 2.0, r0=img.height / 2.0, sc=max(img.width / 2.0, 1.0), sr=max(img.height / 2.0, 1.0))
    return norms


def residual_vector(x: "object", state: BAState, cfg: BAResidualConfig, active_obs_mask: list[bool] | None = None) -> "object":
    """Build full residual vector (reprojection + weak regularization)."""

    import numpy as np  # type: ignore

    ds = state.dataset
    norms = build_image_norms(ds)
    point_init_by_track = {p.track_id: p for p in ds.points_init}

    residuals: list[float] = []
    if active_obs_mask is None:
        active_obs_mask = [True] * len(ds.observations)

    for keep, obs in zip(active_obs_mask, ds.observations):
        if not keep or obs.track_id not in state.point_param_by_track:
            continue
        lon, lat, h = state.unpack_point(x, obs.track_id)
        col_rpc, row_rpc = ds.images[obs.image_id].rpc.project(lon, lat, h)
        params = state.get_image_params(x, obs.image_id)
        dc, dr = affine_delta(col_rpc, row_rpc, params, norms[obs.image_id])
        col_pred = col_rpc + dc
        row_pred = row_rpc + dr
        residuals.append((obs.col - col_pred) / cfg.reproj_sigma)
        residuals.append((obs.row - row_pred) / cfg.reproj_sigma)

    reg_a = cfg.affine_reg_weight ** 0.5
    for image_id, idx in state.image_param_by_id.items():
        _ = image_id
        for k in range(6):
            residuals.append(reg_a * float(x[idx + k]))

    reg_p = cfg.point_reg_weight ** 0.5
    for track_id, idx in state.point_param_by_track.items():
        p0 = point_init_by_track[track_id]
        nlon0, nlat0, nh0 = state.normalizer.normalize(p0.lon, p0.lat, p0.h)
        residuals.append(reg_p * (float(x[idx]) - nlon0))
        residuals.append(reg_p * (float(x[idx + 1]) - nlat0))
        residuals.append(reg_p * (float(x[idx + 2]) - nh0))

    return np.asarray(residuals, dtype=float)


def build_jac_sparsity(state: BAState, active_obs_mask: list[bool] | None = None) -> "object":
    """Build sparsity pattern for least_squares numeric Jacobian."""

    import numpy as np  # type: ignore
    from scipy.sparse import lil_matrix  # type: ignore

    ds = state.dataset
    if active_obs_mask is None:
        active_obs_mask = [True] * len(ds.observations)

    n_obs_used = sum(1 for keep, obs in zip(active_obs_mask, ds.observations) if keep and obs.track_id in state.point_param_by_track)
    rows = 2 * n_obs_used + 6 * len(state.image_param_by_id) + 3 * len(state.point_param_by_track)
    S = lil_matrix((rows, state.n_params), dtype=np.int8)

    r = 0
    for keep, obs in zip(active_obs_mask, ds.observations):
        if not keep or obs.track_id not in state.point_param_by_track:
            continue
        pidx = state.point_param_by_track[obs.track_id]
        for rr in [r, r + 1]:
            for k in range(3):
                S[rr, pidx + k] = 1
        if obs.image_id in state.image_param_by_id:
            iidx = state.image_param_by_id[obs.image_id]
            for rr in [r, r + 1]:
                for k in range(6):
                    S[rr, iidx + k] = 1
        r += 2

    for _, iidx in state.image_param_by_id.items():
        for k in range(6):
            S[r, iidx + k] = 1
            r += 1

    for _, pidx in state.point_param_by_track.items():
        for k in range(3):
            S[r, pidx + k] = 1
            r += 1

    return S.tocsr()
