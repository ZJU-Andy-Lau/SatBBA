"""Matcher abstraction interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from satbba.config.settings import MatcherConfig
from satbba.models.common import PairwiseMatch


class BaseMatcher(ABC):
    """Common interface for all pluggable feature matchers."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config

    @abstractmethod
    def match(self, image_a_path: str, image_b_path: str) -> PairwiseMatch:
        """Run feature matching for one image pair."""

    @abstractmethod
    def validate_runtime(self) -> None:
        """Validate runtime prerequisites (dependencies, weights, etc.)."""
