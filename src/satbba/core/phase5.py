"""Phase-5 orchestration helpers (run-all pipeline)."""

from __future__ import annotations

from satbba.config.settings import AppConfig
from satbba.core.phase3 import run_build_tracks, run_triangulate_init
from satbba.core.phase4 import run_bundle_adjust
from satbba.core.pipeline import run_match_pipeline


def run_all(config: AppConfig) -> None:
    """Execute full pipeline from matching to BA export."""

    run_match_pipeline(config)
    run_build_tracks(config)
    run_triangulate_init(config)
    run_bundle_adjust(config)
