"""Production-oriented staged BA solver (TRF + LSMR)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from satbba.ba.checkpoint import BACheckpoint, load_checkpoint, save_checkpoint
from satbba.ba.dataset_loader import BADataset
from satbba.ba.residuals import BAResidualConfig, build_jac_sparsity, residual_vector
from satbba.ba.state import BAState

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BAConfig:
    """Bundle-adjustment solver options."""

    use_translation_stage: bool = True
    use_affine_stage: bool = True
    robust_loss: str = "soft_l1"
    final_loss: str = "linear"
    reproj_sigma: float = 1.0
    affine_reg_weight: float = 1e-4
    point_reg_weight: float = 1e-6
    outlier_threshold: float = 5.0
    max_nfev: int = 200
    checkpoint_interval: int = 1
    checkpoint_dir: str = "outputs/checkpoints"
    resume: str | None = None


@dataclass(slots=True)
class BAResult:
    """BA solver outputs."""

    x: object
    state: BAState
    active_obs_mask: list[bool]
    reports: list[dict[str, object]]


def _import_numpy_scipy() -> tuple[object, object]:
    import numpy as np  # type: ignore
    from scipy.optimize import least_squares  # type: ignore

    return np, least_squares


def _make_stage_mask(state: BAState, stage_name: str) -> object:
    """Create parameter mask for staged optimization."""

    np, _ = _import_numpy_scipy()
    mask = np.ones(state.n_params, dtype=bool)

    if stage_name == "translation":
        for image_id, idx in state.image_param_by_id.items():
            _ = image_id
            # keep a0,b0 only
            mask[idx + 1 : idx + 3] = False
            mask[idx + 4 : idx + 6] = False
    return mask


def _solve_stage(
    x0: object,
    state: BAState,
    cfg: BAConfig,
    active_obs_mask: list[bool],
    loss: str,
    stage_name: str,
) -> tuple[object, dict[str, object]]:
    """Solve one BA stage with optional frozen parameters."""

    np, least_squares = _import_numpy_scipy()

    r_cfg = BAResidualConfig(
        reproj_sigma=cfg.reproj_sigma,
        affine_reg_weight=cfg.affine_reg_weight,
        point_reg_weight=cfg.point_reg_weight,
    )

    stage_mask = _make_stage_mask(state, stage_name)
    active_idx = np.where(stage_mask)[0]

    def _expand(x_active: object) -> object:
        x_full = np.asarray(x0, dtype=float).copy()
        x_full[active_idx] = x_active
        return x_full

    def fun(x_active: object) -> object:
        x_full = _expand(x_active)
        return residual_vector(x_full, state=state, cfg=r_cfg, active_obs_mask=active_obs_mask)

    sparsity_full = build_jac_sparsity(state, active_obs_mask=active_obs_mask)
    sparsity = sparsity_full[:, active_idx]

    out = least_squares(
        fun,
        np.asarray(x0, dtype=float)[active_idx],
        method="trf",
        tr_solver="lsmr",
        loss=loss,
        x_scale="jac",
        jac_sparsity=sparsity,
        max_nfev=cfg.max_nfev,
    )

    x_out = _expand(out.x)
    residual = fun(out.x)
    obs_count = sum(1 for keep in active_obs_mask if keep)
    mean_err = float(np.mean(np.abs(residual[: 2 * obs_count]))) if len(residual) > 0 else 0.0
    report = {
        "stage": stage_name,
        "success": bool(out.success),
        "nfev": int(out.nfev),
        "cost": float(out.cost),
        "message": str(out.message),
        "mean_abs_residual": mean_err,
    }
    return x_out, report


def _observation_errors(x: object, state: BAState, cfg: BAConfig, active_obs_mask: list[bool]) -> list[float]:
    """Compute per-observation reprojection magnitude from residual vector."""

    r_cfg = BAResidualConfig(cfg.reproj_sigma, cfg.affine_reg_weight, cfg.point_reg_weight)
    r = residual_vector(x, state=state, cfg=r_cfg, active_obs_mask=active_obs_mask)

    out: list[float] = []
    obs_count = sum(
        1
        for keep, obs in zip(active_obs_mask, state.dataset.observations)
        if keep and obs.track_id in state.point_param_by_track
    )
    idx = 0
    for _ in range(obs_count):
        ex = float(r[idx])
        ey = float(r[idx + 1])
        out.append((ex**2 + ey**2) ** 0.5)
        idx += 2
    return out


def _save_stage_checkpoint(
    cfg: BAConfig,
    x: object,
    stage: str,
    iteration: int,
    report: dict[str, object],
) -> None:
    if cfg.checkpoint_interval <= 0:
        return
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt = BACheckpoint(
        x=x,
        stage=stage,
        iteration=iteration,
        residual_stats={k: float(v) for k, v in report.items() if isinstance(v, (int, float))},
        config_snapshot={
            "robust_loss": cfg.robust_loss,
            "final_loss": cfg.final_loss,
            "max_nfev": cfg.max_nfev,
        },
    )
    save_checkpoint(ckpt, ckpt_dir / f"ba_{stage}.pkl")


def run_bundle_adjustment(dataset: BADataset, cfg: BAConfig) -> BAResult:
    """Run translation -> affine -> refine BA stages."""

    _, _ = _import_numpy_scipy()
    state = BAState.from_dataset(dataset)
    active_obs_mask = [obs.track_id in state.point_param_by_track for obs in dataset.observations]
    reports: list[dict[str, object]] = []

    start_stage = "translation"
    if cfg.resume:
        cp = load_checkpoint(Path(cfg.resume))
        x = cp.x
        if cp.stage == "translation":
            start_stage = "affine"
        elif cp.stage == "affine":
            start_stage = "refine"
        reports.append({"stage": "resume", "loaded_from": cfg.resume})
    else:
        x = state.pack()

    if cfg.use_translation_stage and start_stage == "translation":
        x, rep = _solve_stage(x, state, cfg, active_obs_mask, loss=cfg.robust_loss, stage_name="translation")
        reports.append(rep)
        _save_stage_checkpoint(cfg, x, "translation", len(reports), rep)

    if cfg.use_affine_stage and start_stage in {"translation", "affine"}:
        x, rep = _solve_stage(x, state, cfg, active_obs_mask, loss=cfg.robust_loss, stage_name="affine")
        reports.append(rep)
        _save_stage_checkpoint(cfg, x, "affine", len(reports), rep)

    obs_err = _observation_errors(x, state, cfg, active_obs_mask)
    kept_obs: list[bool] = []
    k = 0
    for keep, obs in zip(active_obs_mask, dataset.observations):
        if not keep or obs.track_id not in state.point_param_by_track:
            kept_obs.append(False)
            continue
        kept_obs.append(obs_err[k] <= cfg.outlier_threshold)
        k += 1
    active_obs_mask = kept_obs

    x, rep = _solve_stage(x, state, cfg, active_obs_mask, loss=cfg.final_loss, stage_name="refine")
    rep["removed_outliers"] = int(sum(1 for v in kept_obs if not v))
    reports.append(rep)
    _save_stage_checkpoint(cfg, x, "refine", len(reports), rep)

    return BAResult(x=x, state=state, active_obs_mask=active_obs_mask, reports=reports)


def export_ba_results(result: BAResult, output_dir: Path) -> None:
    """Export affine params, optimized points and report."""

    np, _ = _import_numpy_scipy()
    output_dir.mkdir(parents=True, exist_ok=True)

    affine_rows: list[dict[str, object]] = []
    x = result.x
    for image_id in range(len(result.state.dataset.images)):
        params = list(result.state.get_image_params(x, image_id))
        affine_rows.append(
            {
                "image_id": image_id,
                "params": params,
                "affine_2x3": [
                    [1.0 + params[1], params[2], params[0]],
                    [params[4], 1.0 + params[5], params[3]],
                ],
            }
        )

    (output_dir / "affine_params.json").write_text(json.dumps(affine_rows, indent=2), encoding="utf-8")

    points_rows: list[dict[str, object]] = []
    for p in result.state.dataset.points_init:
        lon, lat, h = result.state.unpack_point(x, p.track_id)
        points_rows.append({"track_id": p.track_id, "lon": lon, "lat": lat, "h": h})

    (output_dir / "points_optimized.json").write_text(json.dumps(points_rows, indent=2), encoding="utf-8")

    residual_cfg = BAResidualConfig()
    rv = residual_vector(x, result.state, residual_cfg, active_obs_mask=result.active_obs_mask)
    mean_abs = float(np.mean(np.abs(rv))) if len(rv) else 0.0
    med_abs = float(np.median(np.abs(rv))) if len(rv) else 0.0

    per_image: dict[int, list[float]] = {}
    obs_used = [
        o
        for keep, o in zip(result.active_obs_mask, result.state.dataset.observations)
        if keep and o.track_id in result.state.point_param_by_track
    ]
    for idx, obs in enumerate(obs_used):
        ex = float(rv[2 * idx])
        ey = float(rv[2 * idx + 1])
        per_image.setdefault(obs.image_id, []).append((ex**2 + ey**2) ** 0.5)

    report = {
        "stages": result.reports,
        "mean_reprojection_error": mean_abs,
        "median_reprojection_error": med_abs,
        "per_image_error": {str(k): float(np.mean(v)) for k, v in per_image.items()},
        "active_observations": int(sum(result.active_obs_mask)),
    }
    (output_dir / "ba_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
