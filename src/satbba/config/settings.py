"""Application configuration dataclasses and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satbba.exceptions import ConfigError


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_dir: Path = Path("outputs/logs")


@dataclass(slots=True)
class MatcherConfig:
    name: str = "sift"
    max_features: int = 8_000
    contrast_threshold: float = 0.04
    ratio_test: float = 0.6
    ransac_thresh: float = 1.5
    weights_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchingConfig:
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    overlap_threshold: float = 0.1
    geo_distance_threshold: float = 100.0
    fundamental_threshold: float = 1.5
    h_ref: float = 0.0


@dataclass(slots=True)
class ReferenceImageConfig:
    mode: str = "auto"
    image_id: int | None = None


@dataclass(slots=True)
class TracksConfig:
    min_views: int = 2
    max_tracks_per_grid_cell: int = 50
    grid_rows: int = 20
    grid_cols: int = 20
    min_score: float | None = None
    drop_disconnected_from_reference: bool = True
    max_track_length: int | None = None
    quantization: float = 0.25


@dataclass(slots=True)
class TriangulationConfig:
    h_ref_mode: str = "rpc_median_offset"
    h_ref_value: float | None = None
    reproj_error_threshold: float = 5.0
    min_success_views: int = 2
    max_nfev: int = 100
    method: str = "trf"
    h_min: float = -1000.0
    h_max: float = 10000.0


@dataclass(slots=True)
class OutputConfig:
    save_plots: bool = False
    export_parquet: bool = False
    export_rpc: bool = True
    export_report: bool = True




@dataclass(slots=True)
class BAConfig:
    use_translation_stage: bool = True
    use_affine_stage: bool = True
    robust_loss: str = "soft_l1"
    final_loss: str = "linear"
    reproj_sigma: float = 1.0
    affine_reg_weight: float = 1e-4
    point_reg_weight: float = 1e-6
    outlier_threshold: float = 5.0
    max_nfev: int = 200


@dataclass(slots=True)
class OptimizationConfig:
    use_numba: bool = False
    batch_size: int = 10000


@dataclass(slots=True)
class RPCRefitConfig:
    enable: bool = False
    grid_size: int = 10


@dataclass(slots=True)
class RuntimeConfig:
    num_workers: int = 1
    checkpoint_interval: int = 10
    resume: str | None = None
    force_recompute: bool = False


@dataclass(slots=True)
class IOConfig:
    image_dir: Path
    rpc_dir: Path | None = None
    output_dir: Path = Path("outputs")


@dataclass(slots=True)
class AppConfig:
    io: IOConfig
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    reference_image: ReferenceImageConfig = field(default_factory=ReferenceImageConfig)
    tracks: TracksConfig = field(default_factory=TracksConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ba: BAConfig = field(default_factory=BAConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rpc_refit: RPCRefitConfig = field(default_factory=RPCRefitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


PipelineConfig = MatchingConfig


def _as_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ConfigError("PyYAML is required for YAML config files") from exc
        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> AppConfig:
    """Load YAML/JSON configuration into :class:`AppConfig`."""

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = _read_config(path)
    dataset_raw = raw.get("dataset", raw.get("io", {}))
    io = IOConfig(
        image_dir=Path(dataset_raw["image_dir"]),
        rpc_dir=_as_path(dataset_raw.get("rpc_dir")),
        output_dir=Path(raw.get("io", {}).get("output_dir", dataset_raw.get("output_dir", "outputs"))),
    )

    matching_raw = raw.get("matching", {})
    matcher_raw = matching_raw.get("matcher", raw.get("matcher", {}))
    sift_raw = matching_raw.get("sift", {})
    matcher = MatcherConfig(
        name=str(matcher_raw.get("type", matcher_raw.get("name", "sift"))).lower(),
        max_features=int(sift_raw.get("nfeatures", matcher_raw.get("max_features", 8000))),
        contrast_threshold=float(sift_raw.get("contrastThreshold", matcher_raw.get("contrast_threshold", 0.04))),
        ratio_test=float(sift_raw.get("ratio", matcher_raw.get("ratio_test", 0.6))),
        ransac_thresh=float(matcher_raw.get("ransac_thresh", 1.5)),
        weights_path=_as_path(matcher_raw.get("weights_path", matching_raw.get("loftr", {}).get("weights_path"))),
        extra=dict(matcher_raw.get("extra", {})),
    )
    matching = MatchingConfig(
        matcher=matcher,
        overlap_threshold=float(matching_raw.get("overlap_threshold", 0.1)),
        geo_distance_threshold=float(matching_raw.get("geo_distance_threshold", 100.0)),
        fundamental_threshold=float(matching_raw.get("fundamental_threshold", 1.5)),
        h_ref=float(matching_raw.get("h_ref", 0.0)),
    )

    ref_raw = raw.get("reference_image", {})
    tracks_raw = raw.get("tracks", {})
    tri_raw = raw.get("triangulation", {})
    output_raw = raw.get("output", {})
    ba_raw = raw.get("ba", {})
    optimization_raw = raw.get("optimization", {})
    rpc_refit_raw = raw.get("rpc_refit", {})
    logging_raw = raw.get("logging", {})
    runtime_raw = raw.get("runtime", {})

    return AppConfig(
        io=io,
        matching=matching,
        reference_image=ReferenceImageConfig(mode=str(ref_raw.get("mode", "auto")), image_id=ref_raw.get("image_id")),
        tracks=TracksConfig(
            min_views=int(tracks_raw.get("min_views", 2)),
            max_tracks_per_grid_cell=int(tracks_raw.get("max_tracks_per_grid_cell", 50)),
            grid_rows=int(tracks_raw.get("grid_rows", 20)),
            grid_cols=int(tracks_raw.get("grid_cols", 20)),
            min_score=tracks_raw.get("min_score"),
            drop_disconnected_from_reference=bool(tracks_raw.get("drop_disconnected_from_reference", True)),
            max_track_length=tracks_raw.get("max_track_length"),
            quantization=float(tracks_raw.get("quantization", 0.25)),
        ),
        triangulation=TriangulationConfig(
            h_ref_mode=str(tri_raw.get("h_ref_mode", "rpc_median_offset")),
            h_ref_value=tri_raw.get("h_ref_value"),
            reproj_error_threshold=float(tri_raw.get("reproj_error_threshold", 5.0)),
            min_success_views=int(tri_raw.get("min_success_views", 2)),
            max_nfev=int(tri_raw.get("max_nfev", 100)),
            method=str(tri_raw.get("method", "trf")),
            h_min=float(tri_raw.get("h_min", -1000.0)),
            h_max=float(tri_raw.get("h_max", 10000.0)),
        ),
        output=OutputConfig(
            save_plots=bool(output_raw.get("save_plots", False)),
            export_parquet=bool(output_raw.get("export_parquet", False)),
            export_rpc=bool(output_raw.get("export_rpc", True)),
            export_report=bool(output_raw.get("export_report", True)),
        ),
        ba=BAConfig(
            use_translation_stage=bool(ba_raw.get("stages", {}).get("use_translation_stage", True)),
            use_affine_stage=bool(ba_raw.get("stages", {}).get("use_affine_stage", True)),
            robust_loss=str(ba_raw.get("robust_loss", "soft_l1")),
            final_loss=str(ba_raw.get("final_loss", "linear")),
            reproj_sigma=float(ba_raw.get("reproj_sigma", 1.0)),
            affine_reg_weight=float(ba_raw.get("affine_reg_weight", 1e-4)),
            point_reg_weight=float(ba_raw.get("point_reg_weight", 1e-6)),
            outlier_threshold=float(ba_raw.get("outlier_threshold", 5.0)),
            max_nfev=int(ba_raw.get("max_nfev", 200)),
        ),
        optimization=OptimizationConfig(
            use_numba=bool(optimization_raw.get("use_numba", False)),
            batch_size=int(optimization_raw.get("batch_size", 10000)),
        ),
        rpc_refit=RPCRefitConfig(
            enable=bool(rpc_refit_raw.get("enable", False)),
            grid_size=int(rpc_refit_raw.get("grid_size", 10)),
        ),
        logging=LoggingConfig(
            level=str(logging_raw.get("level", "INFO")),
            log_to_file=bool(logging_raw.get("log_to_file", True)),
            log_dir=Path(logging_raw.get("log_dir", "outputs/logs")),
        ),
        runtime=RuntimeConfig(
            num_workers=int(runtime_raw.get("num_workers", 1)),
            checkpoint_interval=int(runtime_raw.get("checkpoint_interval", 10)),
            resume=runtime_raw.get("resume"),
            force_recompute=bool(runtime_raw.get("force_recompute", False)),
        ),
    )
