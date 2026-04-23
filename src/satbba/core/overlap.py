"""Footprint estimation and overlap pair selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from satbba.models.dataset import ImageData

LOGGER = logging.getLogger(__name__)


def _import_shapely() -> tuple[type, type]:
    try:
        from shapely.geometry import Polygon  # type: ignore
        from shapely.ops import transform  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("shapely is required for footprint computation") from exc
    return Polygon, transform


def _import_pyproj() -> tuple[object, object]:
    try:
        from pyproj import CRS, Transformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyproj is required for footprint projection") from exc
    return CRS, Transformer


def _sample_boundary_points(width: int, height: int) -> list[tuple[float, float]]:
    """Sample boundary points in strict perimeter order (clockwise).

    Returned sequence starts at top-left and walks along top->right->bottom->left edges.
    This ordering avoids self-intersection when constructing polygon from sampled points.
    """

    w = float(width - 1)
    h = float(height - 1)

    top = [(0.0, 0.0), (w * 0.25, 0.0), (w * 0.5, 0.0), (w * 0.75, 0.0), (w, 0.0)]
    right = [(w, h * 0.25), (w, h * 0.5), (w, h * 0.75), (w, h)]
    bottom = [(w * 0.75, h), (w * 0.5, h), (w * 0.25, h), (0.0, h)]
    left = [(0.0, h * 0.75), (0.0, h * 0.5), (0.0, h * 0.25)]

    ordered = top + right + bottom + left
    # Deduplicate potential identical entries on tiny images while preserving order.
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for c, r in ordered:
        key = (int(round(c * 1e6)), int(round(r * 1e6)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((c, r))
    return deduped


def _utm_transformer(lon: float, lat: float) -> object:
    CRS, Transformer = _import_pyproj()
    zone = int((lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def compute_footprint(image: ImageData, h_ref: float) -> object:
    """Compute projected footprint polygon from image boundary and RPC localization."""

    Polygon, transform = _import_shapely()

    geo_points: list[tuple[float, float]] = []
    for col, row in _sample_boundary_points(image.width, image.height):
        lon, lat = image.rpc.localize(col, row, h_ref)
        geo_points.append((lon, lat))

    transformer = _utm_transformer(geo_points[0][0], geo_points[0][1])
    lonlat_polygon = Polygon(geo_points)
    return transform(transformer.transform, lonlat_polygon)


@dataclass(slots=True)
class OverlapGraph:
    """Overlap graph representation for candidate pair construction."""

    adjacency: dict[int, list[int]]
    pairs: list[tuple[int, int]]


def build_overlap_pairs(images: list[ImageData], threshold: float) -> OverlapGraph:
    """Build overlap graph and candidate pair list from footprint intersections."""

    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(images))}
    pairs: list[tuple[int, int]] = []

    for i in range(len(images)):
        fi = images[i].footprint
        if fi is None:
            continue
        for j in range(i + 1, len(images)):
            fj = images[j].footprint
            if fj is None:
                continue
            inter = fi.intersection(fj).area
            min_area = min(fi.area, fj.area)
            overlap = (inter / min_area) if min_area > 0 else 0.0
            if overlap >= threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
                pairs.append((i, j))

    LOGGER.info("Overlap selection produced %d candidate pairs", len(pairs))
    return OverlapGraph(adjacency=adjacency, pairs=pairs)
