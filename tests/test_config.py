from pathlib import Path

import pytest

from satbba.config.settings import load_config
from satbba.exceptions import ConfigError


def test_load_config_json_success(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"dataset": {"image_dir": "data/images"}, "matching": {"matcher": {"type": "sift"}}}',
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.matching.matcher.name == "sift"


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(ConfigError):
        load_config(Path("/not/exist/config.json"))
