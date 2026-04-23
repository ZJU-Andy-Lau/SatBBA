"""CLI entrypoints for SatBBA."""

from __future__ import annotations

import argparse
from pathlib import Path

from satbba.config.settings import load_config
from satbba.core.ba_pipeline import run_bundle_adjust
from satbba.core.full_pipeline import run_all
from satbba.core.track_pipeline import run_build_tracks
from satbba.core.triangulation_pipeline import run_triangulate_init
from satbba.core.pipeline import run_match_pipeline, run_pipeline
from satbba.logging.setup import configure_logging
from satbba.matching import list_matchers


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""

    parser = argparse.ArgumentParser(prog="sat_bba", description="Satellite BBA toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd, help_text in [
        ("run", "Legacy alias for match pipeline"),
        ("match", "Run pairwise matching pipeline"),
        ("build-tracks", "Build multi-view tracks from pairwise matches"),
        ("triangulate-init", "Triangulate initial tie points and export BA dataset"),
        ("bundle-adjust", "Run phase-4 bundle adjustment solver"),
        ("run-all", "Run full pipeline match->tracks->triangulate->BA"),
    ]:
        p = subparsers.add_parser(cmd, help=help_text)
        p.add_argument("-c", "--config", type=Path, required=True, help="Path to config file")

    subparsers.add_parser("list-matchers", help="List registered matchers")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main function."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-matchers":
        for name in list_matchers():
            print(name)
        return 0

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if args.command == "match":
        run_match_pipeline(cfg)
        return 0
    if args.command == "run":
        run_pipeline(cfg)
        return 0
    if args.command == "build-tracks":
        run_build_tracks(cfg)
        return 0
    if args.command == "triangulate-init":
        run_triangulate_init(cfg)
        return 0
    if args.command == "bundle-adjust":
        run_bundle_adjust(cfg)
        return 0
    if args.command == "run-all":
        run_all(cfg)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
