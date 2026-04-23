"""Visualization generation for BA outputs."""

from __future__ import annotations

import json
from pathlib import Path


def generate_plots(ba_results_dir: Path, output_dir: Path) -> None:
    """Create residual and height histograms if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = ba_results_dir / "ba_report.json"
    points_path = ba_results_dir / "points_optimized.json"
    if not report_path.exists() or not points_path.exists():
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    points = json.loads(points_path.read_text(encoding="utf-8"))

    per_image = report.get("per_image_error", {})
    if per_image:
        vals = list(per_image.values())
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(vals)), vals)
        plt.title("Per-image reprojection error")
        plt.xlabel("image idx")
        plt.ylabel("error")
        plt.tight_layout()
        plt.savefig(output_dir / "residual_map_summary.png", dpi=150)
        plt.close()

    hs = [p.get("h", 0.0) for p in points]
    plt.figure(figsize=(6, 4))
    plt.hist(hs, bins=20)
    plt.title("Tie point height distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "height_hist.png", dpi=150)
    plt.close()

    stage_cost = [s.get("cost", 0.0) for s in report.get("stages", [])]
    if stage_cost:
        plt.figure(figsize=(6, 4))
        plt.plot(stage_cost, marker="o")
        plt.title("Stage cost progression")
        plt.tight_layout()
        plt.savefig(output_dir / "reproj_hist.png", dpi=150)
        plt.close()
