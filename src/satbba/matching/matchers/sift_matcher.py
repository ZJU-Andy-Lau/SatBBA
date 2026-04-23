"""SIFT matcher implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from satbba.config.settings import MatcherConfig
from satbba.matching.base import BaseMatcher
from satbba.matching.registry import register_matcher
from satbba.models.common import PairwiseMatch


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for SIFT matcher") from exc
    return cv2


@register_matcher("sift")
class SIFTMatcher(BaseMatcher):
    """OpenCV SIFT matcher with ratio-test filtering."""

    def validate_runtime(self) -> None:
        """Validate OpenCV SIFT support is available."""

        cv2 = _import_cv2()
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Current OpenCV build does not provide SIFT")

    def _read_gray(self, path: str) -> Any:
        cv2 = _import_cv2()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def match(self, image_a_path: str, image_b_path: str) -> PairwiseMatch:
        """Run SIFT + BF knn + Lowe ratio test."""

        cv2 = _import_cv2()
        image_id_a = Path(image_a_path).stem
        image_id_b = Path(image_b_path).stem

        img_a = self._read_gray(image_a_path)
        img_b = self._read_gray(image_b_path)

        sift = cv2.SIFT_create(
            nfeatures=int(self.config.max_features),
            contrastThreshold=float(self.config.contrast_threshold),
        )

        kp_a, des_a = sift.detectAndCompute(img_a, None)
        kp_b, des_b = sift.detectAndCompute(img_b, None)
        if des_a is None or des_b is None:
            return PairwiseMatch(image_id_a=image_id_a, image_id_b=image_id_b)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(des_a, des_b, k=2)

        pts_a: list[tuple[float, float]] = []
        pts_b: list[tuple[float, float]] = []
        scores: list[float] = []

        ratio = float(self.config.ratio_test)
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                pa = kp_a[m.queryIdx].pt
                pb = kp_b[m.trainIdx].pt
                pts_a.append((float(pa[0]), float(pa[1])))
                pts_b.append((float(pb[0]), float(pb[1])))
                scores.append(float(1.0 / (1e-6 + m.distance)))

        return PairwiseMatch(
            image_id_a=image_id_a,
            image_id_b=image_id_b,
            points_a=pts_a,
            points_b=pts_b,
            confidence=scores,
        )
