import pytest

from satbba.config.settings import MatcherConfig
from satbba.matching.matchers.sift_matcher import SIFTMatcher


def test_sift_matcher_runtime_check() -> None:
    try:
        matcher = SIFTMatcher(MatcherConfig(name="sift"))
        matcher.validate_runtime()
    except RuntimeError:
        pytest.skip("opencv SIFT not available in current environment")
