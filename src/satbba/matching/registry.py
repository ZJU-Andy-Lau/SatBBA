"""Registry for matcher plugin discovery."""

from __future__ import annotations

from collections.abc import Callable

from satbba.config.settings import MatcherConfig
from satbba.exceptions import MatcherError
from satbba.matching.base import BaseMatcher

MatcherFactory = Callable[[MatcherConfig], BaseMatcher]

_MATCHER_REGISTRY: dict[str, MatcherFactory] = {}


def register_matcher(name: str) -> Callable[[MatcherFactory], MatcherFactory]:
    """Decorator used by matcher implementations to self-register."""

    def decorator(factory: MatcherFactory) -> MatcherFactory:
        if name in _MATCHER_REGISTRY:
            raise MatcherError(f"Matcher '{name}' already registered")
        _MATCHER_REGISTRY[name] = factory
        return factory

    return decorator


def build_matcher(config: MatcherConfig) -> BaseMatcher:
    """Instantiate matcher from registry by configured name."""

    key = config.name.lower()
    if key not in _MATCHER_REGISTRY:
        available = ", ".join(sorted(_MATCHER_REGISTRY))
        raise MatcherError(f"Unknown matcher '{config.name}'. Available: [{available}]")
    matcher = _MATCHER_REGISTRY[key](config)
    matcher.validate_runtime()
    return matcher


def list_matchers() -> list[str]:
    """List registered matcher names."""

    return sorted(_MATCHER_REGISTRY.keys())
