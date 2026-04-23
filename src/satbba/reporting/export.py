"""Unified final export helpers."""

from __future__ import annotations

import json
from pathlib import Path


def export_summary(ba_results_dir: Path, out_path: Path) -> None:
    """Generate flattened summary report for downstream dashboards."""

    report = json.loads((ba_results_dir / "ba_report.json").read_text(encoding="utf-8"))
    affine = json.loads((ba_results_dir / "affine_params.json").read_text(encoding="utf-8"))
    points = json.loads((ba_results_dir / "points_optimized.json").read_text(encoding="utf-8"))

    summary = {
        "images": len(affine),
        "points": len(points),
        "mean_reprojection_error": report.get("mean_reprojection_error"),
        "median_reprojection_error": report.get("median_reprojection_error"),
        "active_observations": report.get("active_observations"),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
