"""Pairwise matching pipeline for overlap-selected image pairs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satbba.config.settings import AppConfig
from satbba.exceptions import MatcherError
from satbba.matching.base import BaseMatcher
from satbba.models.dataset import ImageData
from satbba.models.matching import Match, PairMatchResult

LOGGER = logging.getLogger(__name__)


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for match result serialization") from exc
    return np


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for geometric filtering") from exc
    return cv2


def geometric_filter(
    pts_i: list[tuple[float, float]], pts_j: list[tuple[float, float]], threshold: float
) -> list[bool]:
    """Estimate fundamental matrix and return inlier mask."""

    if len(pts_i) < 8:
        return [False] * len(pts_i)

    np = _import_numpy()

    try:
        import pydegensac  # type: ignore

        _, inliers = pydegensac.findFundamentalMatrix(
            np.asarray(pts_i), np.asarray(pts_j), px_th=threshold, conf=0.999, max_iters=10000
        )
        return [bool(x) for x in inliers.reshape(-1)]
    except Exception:
        cv2 = _import_cv2()
        _, mask = cv2.findFundamentalMat(
            np.asarray(pts_i, dtype=np.float64),
            np.asarray(pts_j, dtype=np.float64),
            cv2.FM_RANSAC,
            threshold,
            0.999,
        )
        if mask is None:
            return [False] * len(pts_i)
        return [bool(x) for x in mask.reshape(-1)]


def geo_distance_filter(
    image_i: ImageData,
    image_j: ImageData,
    pts_i: list[tuple[float, float]],
    pts_j: list[tuple[float, float]],
    h_ref: float,
    threshold_m: float,
) -> list[bool]:
    """Filter matches by RPC-localized geographic distance in projected CRS."""

    try:
        from pyproj import CRS, Transformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyproj is required for geo distance filtering") from exc

    mask: list[bool] = []
    for pti, ptj in zip(pts_i, pts_j):
        lon_i, lat_i = image_i.rpc.localize(pti[0], pti[1], h_ref)
        lon_j, lat_j = image_j.rpc.localize(ptj[0], ptj[1], h_ref)

        zone = int((lon_i + 180.0) / 6.0) + 1
        epsg = 32600 + zone if lat_i >= 0 else 32700 + zone
        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
        xi, yi = transformer.transform(lon_i, lat_i)
        xj, yj = transformer.transform(lon_j, lat_j)
        d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
        mask.append(d < threshold_m)
    return mask


def match_pair(
    idx_i: int,
    idx_j: int,
    image_i: ImageData,
    image_j: ImageData,
    matcher: BaseMatcher,
    config: AppConfig,
) -> PairMatchResult:
    """Run matching and filtering on one image pair."""

    pair = matcher.match(str(image_i.image_path), str(image_j.image_path))
    raw_pts_i = list(pair.points_a)
    raw_pts_j = list(pair.points_b)
    scores = pair.confidence or [1.0] * len(raw_pts_i)

    if len(raw_pts_i) != len(raw_pts_j):
        raise MatcherError("Matcher output has inconsistent point arrays")

    geom_mask = geometric_filter(raw_pts_i, raw_pts_j, config.matching.fundamental_threshold)
    geom_pts_i = [p for p, keep in zip(raw_pts_i, geom_mask) if keep]
    geom_pts_j = [p for p, keep in zip(raw_pts_j, geom_mask) if keep]
    geom_scores = [s for s, keep in zip(scores, geom_mask) if keep]

    geo_mask = geo_distance_filter(
        image_i,
        image_j,
        geom_pts_i,
        geom_pts_j,
        h_ref=config.matching.h_ref,
        threshold_m=config.matching.geo_distance_threshold,
    )

    matches: list[Match] = []
    for pti, ptj, score, keep in zip(geom_pts_i, geom_pts_j, geom_scores, geo_mask):
        if keep:
            matches.append(Match(image_i=idx_i, image_j=idx_j, pt_i=pti, pt_j=ptj, score=float(score)))

    return PairMatchResult(
        image_i=idx_i,
        image_j=idx_j,
        matches=matches,
        raw_count=len(raw_pts_i),
        geometric_inliers=len(geom_pts_i),
        geo_filtered=len(matches),
    )


def save_pair_result(output_dir: Path, result: PairMatchResult) -> Path:
    """Save one pair matching result to `.npz` format."""

    np = _import_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pair_{result.image_i:04d}_{result.image_j:04d}.npz"
    pts_i = np.asarray([m.pt_i for m in result.matches], dtype=float)
    pts_j = np.asarray([m.pt_j for m in result.matches], dtype=float)
    scores = np.asarray([m.score for m in result.matches], dtype=float)
    np.savez(out_path, pts_i=pts_i, pts_j=pts_j, scores=scores)
    return out_path


@dataclass(slots=True)
class MatchSummary:
    """Aggregated statistics for full pairwise matching run."""

    total_pairs: int
    processed_pairs: int
    dropped_pairs: int
    average_matches: float


def run_matching_pipeline(
    images: list[ImageData],
    pairs: list[tuple[int, int]],
    matcher: BaseMatcher,
    config: AppConfig,
) -> MatchSummary:
    """Run pairwise matching for all candidate pairs and save outputs."""

    matches_dir = config.io.output_dir / "matches"
    pairs_meta: list[dict[str, Any]] = []

    processed = 0
    total_matches = 0
    for i, j in pairs:
        result = match_pair(i, j, images[i], images[j], matcher, config)
        LOGGER.info(
            "pair=(%d,%d) raw=%d geom=%d geo=%d",
            i,
            j,
            result.raw_count,
            result.geometric_inliers,
            result.geo_filtered,
        )
        if result.geo_filtered == 0:
            continue

        out_path = save_pair_result(matches_dir, result)
        pairs_meta.append({"pair": [i, j], "file": out_path.name, "matches": result.geo_filtered})
        processed += 1
        total_matches += result.geo_filtered

    meta_path = config.io.output_dir / "pairs.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(pairs_meta, indent=2), encoding="utf-8")

    dropped = len(pairs) - processed
    avg = float(total_matches / processed) if processed > 0 else 0.0
    LOGGER.info("matching done: total_pairs=%d processed=%d dropped=%d avg=%.2f", len(pairs), processed, dropped, avg)
    return MatchSummary(len(pairs), processed, dropped, avg)
