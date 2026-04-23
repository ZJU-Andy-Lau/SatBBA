from pathlib import Path

import pytest

from satbba.ba.checkpoint import BACheckpoint, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    ck = BACheckpoint(x=np.array([1.0, 2.0]), stage="affine", iteration=2, residual_stats={"cost": 1.2}, config_snapshot={"a": 1})
    path = tmp_path / "ck.pkl"
    save_checkpoint(ck, path)
    loaded = load_checkpoint(path)
    assert loaded.stage == "affine"
    assert loaded.iteration == 2
