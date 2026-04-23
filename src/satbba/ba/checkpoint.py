"""Checkpoint save/load for BA optimization progress."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BACheckpoint:
    """Serializable BA checkpoint payload."""

    x: object
    stage: str
    iteration: int
    residual_stats: dict[str, float]
    config_snapshot: dict[str, object]


def save_checkpoint(checkpoint: BACheckpoint, path: Path) -> None:
    """Persist checkpoint to pickle + metadata JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(checkpoint, fp)
    meta = {
        "stage": checkpoint.stage,
        "iteration": checkpoint.iteration,
        "residual_stats": checkpoint.residual_stats,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> BACheckpoint:
    """Load checkpoint payload from disk."""

    with path.open("rb") as fp:
        obj = pickle.load(fp)
    if not isinstance(obj, BACheckpoint):
        raise TypeError("invalid checkpoint payload")
    return obj
