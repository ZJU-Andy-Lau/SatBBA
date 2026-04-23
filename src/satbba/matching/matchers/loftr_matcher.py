"""LoFTR matcher implementation with local weight loading only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from satbba.config.settings import MatcherConfig
from satbba.exceptions import WeightFileNotFoundError
from satbba.matching.base import BaseMatcher
from satbba.matching.registry import register_matcher
from satbba.models.common import PairwiseMatch


@register_matcher("loftr")
class LoFTRMatcher(BaseMatcher):
    """LoFTR wrapper requiring local checkpoint path."""

    def __init__(self, config: MatcherConfig) -> None:
        super().__init__(config)
        self.weights_path = config.weights_path

    def validate_runtime(self) -> None:
        """Ensure local checkpoint exists and LoFTR dependency is present."""

        if self.weights_path is None:
            raise WeightFileNotFoundError(
                "LoFTR requires matcher.weights_path to point to a local checkpoint file"
            )
        if not self.weights_path.exists() or not self.weights_path.is_file():
            raise WeightFileNotFoundError(f"LoFTR weights not found: {self.weights_path}")

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("LoFTR matcher requires PyTorch to be installed") from exc

    def _load_backend(self) -> Any:
        try:
            from kornia.feature import LoFTR  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LoFTR dependency missing. Install kornia and related packages."
            ) from exc
        return LoFTR

    def match(self, image_a_path: str, image_b_path: str) -> PairwiseMatch:
        """Run LoFTR inference if dependencies are available.

        Notes
        -----
        This stage provides production-ready integration points and strict weight-policy
        checks. Full acceleration/device policies are deferred to next phase.
        """

        image_id_a = Path(image_a_path).stem
        image_id_b = Path(image_b_path).stem

        # Backend check only; detailed tensor preprocessing/inference is TODO for phase-3.
        self._load_backend()
        raise NotImplementedError(
            "LoFTR forward inference pipeline is not yet implemented in phase-2. "
            "Weight loading policy and runtime checks are implemented."
        )
