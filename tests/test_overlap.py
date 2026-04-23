from dataclasses import dataclass
from pathlib import Path

from satbba.core.overlap import build_overlap_pairs
from satbba.models.dataset import ImageData


@dataclass(slots=True)
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection(self, other: "_Rect") -> "_Rect":
        return _Rect(max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1))


class _FakeRPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


def _image(idx: int, rect: _Rect) -> ImageData:
    image = ImageData(str(idx), Path(f"{idx}.tif"), 100, 100, _FakeRPC())
    image.footprint = rect
    return image


def test_build_overlap_pairs() -> None:
    images = [
        _image(0, _Rect(0, 0, 10, 10)),
        _image(1, _Rect(5, 5, 15, 15)),
        _image(2, _Rect(20, 20, 30, 30)),
    ]
    graph = build_overlap_pairs(images, threshold=0.2)
    assert graph.pairs == [(0, 1)]
    assert graph.adjacency[0] == [1]
