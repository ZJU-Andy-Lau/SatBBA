"""Load phase-3 exported BA input dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from satbba.io.catalog import scan_image_catalog
from satbba.models.dataset import ImageData


@dataclass(slots=True)
class BAObservation:
    """Observation row for BA solver."""

    obs_id: int
    track_id: int
    image_id: int
    col: float
    row: float
    score: float | None = None


@dataclass(slots=True)
class BAPointInit:
    """Initial tie-point row for BA solver."""

    track_id: int
    lon: float
    lat: float
    h: float
    n_views: int
    mean_reproj_error: float


@dataclass(slots=True)
class BADataset:
    """In-memory BA dataset."""

    images: list[ImageData]
    observations: list[BAObservation]
    points_init: list[BAPointInit]
    reference_image_id: int
    metadata: dict[str, object]


def _read_observations(path: Path) -> list[BAObservation]:
    out: list[BAObservation] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            out.append(
                BAObservation(
                    obs_id=int(row["obs_id"]),
                    track_id=int(row["track_id"]),
                    image_id=int(row["image_id"]),
                    col=float(row["col"]),
                    row=float(row["row"]),
                    score=float(row["score"]) if row.get("score") not in {None, "", "None"} else None,
                )
            )
    return out


def _read_points(path: Path) -> list[BAPointInit]:
    out: list[BAPointInit] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            out.append(
                BAPointInit(
                    track_id=int(row["track_id"]),
                    lon=float(row["lon_init"]),
                    lat=float(row["lat_init"]),
                    h=float(row["h_init"]),
                    n_views=int(row["n_views"]),
                    mean_reproj_error=float(row["mean_reproj_error"]),
                )
            )
    return out


def load_ba_dataset(image_dir: Path, rpc_dir: Path | None, output_dir: Path) -> BADataset:
    """Load phase-3 BA input files and image/RPC catalog."""

    ba_dir = output_dir / "ba_input"
    meta_path = ba_dir / "metadata.json"
    observations_path = ba_dir / "observations.csv"
    points_path = ba_dir / "tracks_points_init.csv"

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    images = scan_image_catalog(image_dir=image_dir, rpc_dir=rpc_dir)
    observations = _read_observations(observations_path)
    points_init = _read_points(points_path)
    return BADataset(
        images=images,
        observations=observations,
        points_init=points_init,
        reference_image_id=int(metadata["reference_image_id"]),
        metadata=metadata,
    )
