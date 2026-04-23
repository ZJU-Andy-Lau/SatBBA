from pathlib import Path

from satbba.models.dataset import ImageData
from satbba.models.tracks import Observation, Track
from satbba.triangulation.initializer import select_reference_image


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


def test_reference_fixed() -> None:
    images = [ImageData(str(i), Path(f"{i}.tif"), 100, 100, _RPC()) for i in range(3)]
    result = select_reference_image("fixed", 2, images, [])
    assert result.reference_image_id == 2


def test_reference_auto() -> None:
    images = [ImageData(str(i), Path(f"{i}.tif"), 100, 100, _RPC()) for i in range(3)]
    tracks = [
        Track(0, [Observation(0, 0, 1, 1), Observation(1, 1, 2, 2)]),
        Track(1, [Observation(2, 0, 1, 1), Observation(3, 2, 2, 2)]),
    ]
    result = select_reference_image("auto", None, images, tracks)
    assert result.reference_image_id == 0
