"""Export and load BA input dataset files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from satbba.datasets.ba_dataset import BADataset
from satbba.models.tracks import Observation, Track, TriangulatedPoint


def export_ba_dataset(dataset: BADataset, output_dir: Path, export_parquet: bool = False) -> None:
    """Write BA dataset to disk (CSV + JSON metadata)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    images_csv = output_dir / "images.csv"
    observations_csv = output_dir / "observations.csv"
    tracks_csv = output_dir / "tracks_points_init.csv"
    metadata_json = output_dir / "metadata.json"

    with images_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "image_path", "width", "height", "is_reference"])
        writer.writeheader()
        for idx, image in enumerate(dataset.images):
            writer.writerow(
                {
                    "image_id": idx,
                    "image_path": str(image.image_path),
                    "width": image.width,
                    "height": image.height,
                    "is_reference": int(idx == dataset.reference_image_id),
                }
            )

    with observations_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["obs_id", "track_id", "image_id", "col", "row", "score"])
        writer.writeheader()
        for tr in dataset.tracks:
            for obs in tr.observations:
                writer.writerow(
                    {
                        "obs_id": obs.obs_id,
                        "track_id": tr.track_id,
                        "image_id": obs.image_id,
                        "col": obs.col,
                        "row": obs.row,
                        "score": obs.score,
                    }
                )

    by_track = {p.track_id: p for p in dataset.points_init}
    with tracks_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "track_id",
                "lon_init",
                "lat_init",
                "h_init",
                "n_views",
                "mean_reproj_error",
                "triangulation_success",
            ],
        )
        writer.writeheader()
        for tr in dataset.tracks:
            p = by_track.get(tr.track_id)
            if p is None:
                continue
            writer.writerow(
                {
                    "track_id": tr.track_id,
                    "lon_init": p.lon_init,
                    "lat_init": p.lat_init,
                    "h_init": p.h_init,
                    "n_views": p.n_views,
                    "mean_reproj_error": p.mean_reproj_error,
                    "triangulation_success": int(p.triangulation_success),
                }
            )

    metadata = dict(dataset.metadata)
    metadata["reference_image_id"] = dataset.reference_image_id
    metadata["h_ref"] = dataset.h_ref
    metadata["counts_summary"] = {
        "images": len(dataset.images),
        "tracks": len(dataset.tracks),
        "observations": sum(len(t.observations) for t in dataset.tracks),
        "points_init": len(dataset.points_init),
    }
    metadata["export_parquet_requested"] = bool(export_parquet)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_exported_observations(path: Path) -> list[Observation]:
    """Load observation rows from exported CSV file."""

    observations: list[Observation] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            observations.append(
                Observation(
                    obs_id=int(row["obs_id"]),
                    image_id=int(row["image_id"]),
                    col=float(row["col"]),
                    row=float(row["row"]),
                    score=float(row["score"]) if row["score"] not in {"", "None"} else None,
                )
            )
    return observations


def load_exported_points(path: Path) -> list[TriangulatedPoint]:
    """Load triangulated points rows from exported CSV file."""

    points: list[TriangulatedPoint] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            points.append(
                TriangulatedPoint(
                    track_id=int(row["track_id"]),
                    lon_init=float(row["lon_init"]),
                    lat_init=float(row["lat_init"]),
                    h_init=float(row["h_init"]),
                    n_views=int(row["n_views"]),
                    mean_reproj_error=float(row["mean_reproj_error"]),
                    triangulation_success=bool(int(row["triangulation_success"])),
                )
            )
    return points
