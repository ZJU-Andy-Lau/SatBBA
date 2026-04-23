from pathlib import Path

from satbba.datasets.ba_dataset import BADataset
from satbba.datasets.export import export_ba_dataset, load_exported_observations, load_exported_points
from satbba.models.dataset import ImageData
from satbba.models.tracks import Observation, Track, TriangulatedPoint


class _RPC:
    def project(self, lon: float, lat: float, h: float) -> tuple[float, float]:
        return lon, lat

    def localize(self, col: float, row: float, h: float) -> tuple[float, float]:
        return col, row


def test_export_and_reload(tmp_path: Path) -> None:
    images = [ImageData("0", Path("0.tif"), 100, 100, _RPC())]
    track = Track(0, [Observation(0, 0, 10.0, 20.0, 0.9)])
    point = TriangulatedPoint(0, 100.0, 30.0, 50.0, 1, 0.1, True)
    ds = BADataset(images=images, tracks=[track], observations=track.observations, points_init=[point], reference_image_id=0, h_ref=0.0)

    export_ba_dataset(ds, tmp_path)
    obs = load_exported_observations(tmp_path / "observations.csv")
    pts = load_exported_points(tmp_path / "tracks_points_init.csv")
    assert len(obs) == 1
    assert len(pts) == 1
