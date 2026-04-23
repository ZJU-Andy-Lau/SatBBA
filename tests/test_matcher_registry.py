from pathlib import Path

import pytest

from satbba.config.settings import MatcherConfig
from satbba.exceptions import MatcherError, WeightFileNotFoundError
from satbba.matching import build_matcher, list_matchers


def test_matcher_registry_contains_expected_plugins() -> None:
    names = list_matchers()
    assert "sift" in names
    assert "loftr" in names


def test_unknown_matcher_raises() -> None:
    with pytest.raises(MatcherError):
        build_matcher(MatcherConfig(name="unknown"))


def test_loftr_requires_local_weight_path() -> None:
    cfg = MatcherConfig(name="loftr", weights_path=Path("/tmp/missing.ckpt"))
    with pytest.raises(WeightFileNotFoundError):
        build_matcher(cfg)
