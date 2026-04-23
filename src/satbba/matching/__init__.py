"""Matcher package with plugin registration side effects."""

from satbba.matching.matchers.loftr_matcher import LoFTRMatcher
from satbba.matching.matchers.sift_matcher import SIFTMatcher
from satbba.matching.registry import build_matcher, list_matchers

__all__ = ["SIFTMatcher", "LoFTRMatcher", "build_matcher", "list_matchers"]
