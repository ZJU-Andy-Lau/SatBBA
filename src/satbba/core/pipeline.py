"""Top-level SatBBA pipeline orchestration."""

from __future__ import annotations

import logging

from satbba.config.settings import AppConfig
from satbba.core.matching_pipeline import run_matching_pipeline
from satbba.core.overlap import build_overlap_pairs, compute_footprint
from satbba.io.catalog import scan_image_catalog
from satbba.matching import build_matcher

LOGGER = logging.getLogger(__name__)


def run_pipeline(config: AppConfig) -> None:
    """Compatibility wrapper for stage-1 `run` command."""

    LOGGER.info("Running match pipeline via legacy 'run' command")
    run_match_pipeline(config)


def run_match_pipeline(config: AppConfig) -> None:
    """Execute phase-2 pairwise matching pipeline."""

    images = scan_image_catalog(config.io.image_dir, rpc_dir=config.io.rpc_dir)

    for image in images:
        image.footprint = compute_footprint(image, h_ref=config.matching.h_ref)

    overlap_graph = build_overlap_pairs(images, threshold=config.matching.overlap_threshold)
    matcher = build_matcher(config.matching.matcher)
    summary = run_matching_pipeline(images, overlap_graph.pairs, matcher, config)
    LOGGER.info(
        "Summary total=%d processed=%d dropped=%d avg=%.2f",
        summary.total_pairs,
        summary.processed_pairs,
        summary.dropped_pairs,
        summary.average_matches,
    )
