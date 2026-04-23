"""Phase-4 orchestration for bundle adjustment."""

from __future__ import annotations

import json
import logging

from satbba.ba.dataset_loader import load_ba_dataset
from satbba.ba.rpc_refit import RPCRefitConfig, export_refined_rpc, refit_rpc
from satbba.ba.solver import BAConfig, export_ba_results, run_bundle_adjustment
from satbba.config.settings import AppConfig
from satbba.reporting.export import export_summary
from satbba.reporting.plots import generate_plots

LOGGER = logging.getLogger(__name__)


def run_bundle_adjust(config: AppConfig) -> None:
    """Run BA solver from phase-3 exported dataset and write outputs."""

    ba_dir = config.io.output_dir / "ba_results"
    if ba_dir.exists() and not config.runtime.force_recompute and not config.runtime.resume:
        LOGGER.info("BA results already exist. Skipping due to force_recompute=false")
        return

    dataset = load_ba_dataset(config.io.image_dir, config.io.rpc_dir, config.io.output_dir)
    ba_cfg = BAConfig(
        use_translation_stage=config.ba.use_translation_stage,
        use_affine_stage=config.ba.use_affine_stage,
        robust_loss=config.ba.robust_loss,
        final_loss=config.ba.final_loss,
        reproj_sigma=config.ba.reproj_sigma,
        affine_reg_weight=config.ba.affine_reg_weight,
        point_reg_weight=config.ba.point_reg_weight,
        outlier_threshold=config.ba.outlier_threshold,
        max_nfev=config.ba.max_nfev,
        checkpoint_interval=config.runtime.checkpoint_interval,
        checkpoint_dir=str(config.io.output_dir / "checkpoints"),
        resume=config.runtime.resume,
    )
    result = run_bundle_adjustment(dataset, ba_cfg)
    export_ba_results(result, ba_dir)

    if config.rpc_refit.enable:
        affine_rows = json.loads((ba_dir / "affine_params.json").read_text(encoding="utf-8"))
        rpc_out = config.io.output_dir / "rpc_refined"
        for row in affine_rows:
            image_id = int(row["image_id"])
            if image_id >= len(dataset.images):
                continue
            rpc_res = refit_rpc(dataset.images[image_id], tuple(row["params"]), RPCRefitConfig(config.rpc_refit.grid_size))
            export_refined_rpc(rpc_res, rpc_out)

    if config.output.save_plots:
        generate_plots(ba_dir, config.io.output_dir / "plots")

    if config.output.export_report:
        export_summary(ba_dir, config.io.output_dir / "report.json")

    LOGGER.info("bundle-adjust finished")
