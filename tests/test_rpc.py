from pathlib import Path

import pytest

from satbba.io.rpc import RPCModel, load_rpc_model


class _FakeRPCBackend:
    def projection(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon + h, lat + h

    def localization(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col - h, row - h


class _FakeDataset:
    def tags(self, ns: str) -> dict[str, str]:
        assert ns == "RPC"
        return {"LINE_OFF": "0", "SAMP_OFF": "0"}

    def __enter__(self) -> "_FakeDataset":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeRasterio:
    def open(self, path: Path) -> _FakeDataset:
        return _FakeDataset()


class _FakeRPCM:
    class RPCModel:
        def __init__(self, rpc_dict: dict[str, object]) -> None:
            self.rpc_dict = rpc_dict

        def projection(self, lon: float, lat: float, h: float) -> tuple[float, float]:
            return lon, lat

        def localization(self, col: float, row: float, h: float) -> tuple[float, float]:
            return col, row


def test_rpc_model_project_localize() -> None:
    model = RPCModel(_FakeRPCBackend())
    assert model.project(1.0, 2.0, 3.0) == (4.0, 5.0)
    assert model.localize(4.0, 5.0, 3.0) == (1.0, 2.0)


def test_load_rpc_model_from_tags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import satbba.io.rpc as rpc_module

    monkeypatch.setattr(rpc_module, "_import_rasterio", lambda: _FakeRasterio())
    monkeypatch.setattr(rpc_module, "_import_rpcm", lambda: _FakeRPCM())

    image_path = tmp_path / "a.tif"
    image_path.write_text("x", encoding="utf-8")
    model = load_rpc_model(image_path)
    assert model.localize(12.0, 34.0, 0.0) == (12.0, 34.0)
