"""Phase-3 orchestration: tracks build and triangulation initialization."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from satbba.config.settings import AppConfig
from satbba.dataio.dataset_loader import load_matching_dataset
from satbba.datasets.ba_dataset import BADataset
from satbba.datasets.export import export_ba_dataset
from satbba.models.tracks import Track
from satbba.tracking.track_builder import build_tracks
from satbba.tracking.track_filter import TrackFilterConfig, filter_tracks
from satbba.triangulation.initializer import resolve_h_ref, select_reference_image
from satbba.triangulation.quality import TriangulationQualityConfig, filter_triangulated_points
from satbba.triangulation.triangulator import TriangulationConfig, triangulate_track

LOGGER = logging.getLogger(__name__)


def _tracks_dir(output_dir: Path) -> Path:
    return output_dir / "tracks"


def _triangulation_dir(output_dir: Path) -> Path:
    return output_dir / "ba_input"


def _save_tracks_jsonl(tracks: list[Track], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    obs_path = output_dir / "observations.csv"

    with tracks_path.open("w", encoding="utf-8") as fp:
        for t in tracks:
            payload = {
                "track_id": t.track_id,
                "observations": [
                    {
                        "obs_id": o.obs_id,
                        "image_id": o.image_id,
                        "col": o.col,
                        "row": o.row,
                        "score": o.score,
                    }
                    for o in t.observations
                ],
            }
            fp.write(json.dumps(payload) + "\n")

    with obs_path.open("w", encoding="utf-8") as fp:
        fp.write("obs_id,track_id,image_id,col,row,score\n")
        for t in tracks:
            for o in t.observations:
                fp.write(f"{o.obs_id},{t.track_id},{o.image_id},{o.col},{o.row},{o.score}\n")


def _load_tracks_jsonl(path: Path) -> list[Track]:
    from satbba.models.tracks import Observation

    tracks: list[Track] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        obs = [
            Observation(
                obs_id=int(o["obs_id"]),
                image_id=int(o["image_id"]),
                col=float(o["col"]),
                row=float(o["row"]),
                score=float(o["score"]) if o["score"] is not None else None,
            )
            for o in payload["observations"]
        ]
        tracks.append(Track(track_id=int(payload["track_id"]), observations=obs))
    return tracks


def run_build_tracks(config: AppConfig) -> None:
    """CLI command implementation: build-tracks."""

    dataset = load_matching_dataset(config.io.image_dir, config.io.rpc_dir, config.io.output_dir)
    build_result = build_tracks(dataset, quantization=config.tracks.quantization)

    ref_decision = select_reference_image(
        mode=config.reference_image.mode,
        fixed_image_id=config.reference_image.image_id,
        images=dataset.images,
        tracks=build_result.tracks,
    )

    filtered_tracks, filter_stats = filter_tracks(
        build_result.tracks,
        TrackFilterConfig(
            min_views=config.tracks.min_views,
            min_score=config.tracks.min_score,
            max_track_length=config.tracks.max_track_length,
            max_tracks_per_grid_cell=config.tracks.max_tracks_per_grid_cell,
            grid_rows=config.tracks.grid_rows,
            grid_cols=config.tracks.grid_cols,
            drop_disconnected_from_reference=config.tracks.drop_disconnected_from_reference,
        ),
        reference_image_id=ref_decision.reference_image_id,
    )

    out_dir = _tracks_dir(config.io.output_dir)
    _save_tracks_jsonl(filtered_tracks, out_dir)

    summary = {
        "raw_pairs": len(dataset.pairs),
        "raw_matches": build_result.total_edges,
        "raw_tracks": len(build_result.tracks),
        "filtered_tracks": len(filtered_tracks),
        "average_track_length": filter_stats.average_length,
        "length_histogram": filter_stats.length_histogram,
        "tracks_per_image": filter_stats.tracks_per_image,
        "reference_image": {
            "image_id": ref_decision.reference_image_id,
            "reason": ref_decision.reason,
            "scores": ref_decision.scores,
        },
    }
    (out_dir / "track_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("build-tracks finished: %d tracks kept", len(filtered_tracks))


def run_triangulate_init(config: AppConfig) -> None:
    """CLI command implementation: triangulate-init."""

    dataset = load_matching_dataset(config.io.image_dir, config.io.rpc_dir, config.io.output_dir)
    track_path = _tracks_dir(config.io.output_dir) / "tracks.jsonl"
    tracks = _load_tracks_jsonl(track_path)

    ref_decision = select_reference_image(
        mode=config.reference_image.mode,
        fixed_image_id=config.reference_image.image_id,
        images=dataset.images,
        tracks=tracks,
    )
    h_ref = resolve_h_ref(config.triangulation.h_ref_mode, config.triangulation.h_ref_value, dataset.images)

    tri_cfg = TriangulationConfig(method=config.triangulation.method, max_nfev=config.triangulation.max_nfev)
    points = [
        triangulate_track(t, dataset.images, h_ref=h_ref, reference_image_id=ref_decision.reference_image_id, cfg=tri_cfg)
        for t in tracks
    ]

    points_kept, quality = filter_triangulated_points(
        points,
        TriangulationQualityConfig(
            reproj_error_threshold=config.triangulation.reproj_error_threshold,
            min_success_views=config.triangulation.min_success_views,
            h_min=config.triangulation.h_min,
            h_max=config.triangulation.h_max,
        ),
    )

    ba_dataset = BADataset(
        images=dataset.images,
        tracks=[t for t in tracks if any(p.track_id == t.track_id for p in points_kept)],
        observations=[o for t in tracks for o in t.observations],
        points_init=points_kept,
        reference_image_id=ref_decision.reference_image_id,
        h_ref=h_ref,
        metadata={
            "triangulation": {
                "success": quality.success_count,
                "rejected": quality.rejected_count,
                "reject_reasons": quality.reject_reasons,
            }
        },
    )
    export_ba_dataset(ba_dataset, _triangulation_dir(config.io.output_dir), export_parquet=config.output.export_parquet)
    LOGGER.info("triangulate-init finished: %d/%d points kept", len(points_kept), len(points))
