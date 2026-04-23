"""Reference image selection and initial XYZ seed helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from satbba.models.dataset import ImageData
from satbba.models.tracks import Track


@dataclass(slots=True)
class ReferenceSelectionResult:
    """Reference image decision and rationale."""

    reference_image_id: int
    scores: dict[int, float]
    reason: str


def select_reference_image(
    mode: str,
    fixed_image_id: int | None,
    images: list[ImageData],
    tracks: list[Track],
) -> ReferenceSelectionResult:
    """Select reference image by fixed or auto strategy."""

    if mode == "fixed":
        if fixed_image_id is None:
            raise ValueError("reference image mode is fixed but image_id is missing")
        return ReferenceSelectionResult(fixed_image_id, {fixed_image_id: 1.0}, "fixed-config")

    scores: dict[int, float] = {idx: 0.0 for idx in range(len(images))}
    for t in tracks:
        views = {o.image_id for o in t.observations}
        for vid in views:
            scores[vid] += len(views)

    selected = min(sorted(scores.keys()), key=lambda idx: (-scores[idx], idx))
    return ReferenceSelectionResult(selected, scores, "auto-max-track-connectivity")


def resolve_h_ref(mode: str, fixed_value: float | None, images: list[ImageData]) -> float:
    """Resolve initial height reference from RPC metadata or fixed value."""

    if mode == "fixed":
        if fixed_value is None:
            raise ValueError("triangulation.h_ref_mode=fixed requires h_ref_value")
        return float(fixed_value)

    offsets: list[float] = []
    for img in images:
        rpc_obj = getattr(img.rpc, "rpc_obj", None)
        if rpc_obj is None:
            continue
        for key in ("HEIGHT_OFF", "height_off", "height_offset"):
            if hasattr(rpc_obj, "__dict__") and key in rpc_obj.__dict__:
                offsets.append(float(rpc_obj.__dict__[key]))
                break
    return float(median(offsets)) if offsets else 0.0
