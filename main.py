"""Direct project entrypoint without `pip install -e .`.

Usage
-----
python main.py <command> [options]

Example
-------
python main.py bundle-adjust --config configs/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _bootstrap_src_path()
    from satbba.cli.main import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
