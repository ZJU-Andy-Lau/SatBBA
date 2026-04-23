"""State vector management for BA optimization."""

from __future__ import annotations

from dataclasses import dataclass

from satbba.ba.dataset_loader import BADataset


@dataclass(slots=True)
class PointNormalizer:
    """Normalization transform for lon/lat/h point parameters."""

    lon0: float
    lat0: float
    h0: float
    slon: float
    slat: float
    sh: float

    def normalize(self, lon: float, lat: float, h: float) -> tuple[float, float, float]:
        return (lon - self.lon0) / self.slon, (lat - self.lat0) / self.slat, (h - self.h0) / self.sh

    def denormalize(self, xlon: float, xlat: float, xh: float) -> tuple[float, float, float]:
        return self.lon0 + xlon * self.slon, self.lat0 + xlat * self.slat, self.h0 + xh * self.sh


@dataclass(slots=True)
class BAState:
    """Pack/unpack helper with index mappings."""

    dataset: BADataset
    normalizer: PointNormalizer
    image_param_by_id: dict[int, int]
    point_param_by_track: dict[int, int]
    n_image_params: int
    n_point_params: int

    @classmethod
    def from_dataset(cls, dataset: BADataset) -> "BAState":
        points = dataset.points_init
        lon_vals = [p.lon for p in points] or [0.0]
        lat_vals = [p.lat for p in points] or [0.0]
        h_vals = [p.h for p in points] or [0.0]

        def _scale(vals: list[float]) -> float:
            mn, mx = min(vals), max(vals)
            return max(mx - mn, 1e-6)

        normalizer = PointNormalizer(
            lon0=sum(lon_vals) / len(lon_vals),
            lat0=sum(lat_vals) / len(lat_vals),
            h0=sum(h_vals) / len(h_vals),
            slon=_scale(lon_vals),
            slat=_scale(lat_vals),
            sh=max(_scale(h_vals), 10.0),
        )

        image_param_by_id: dict[int, int] = {}
        offset = 0
        for image_id in range(len(dataset.images)):
            if image_id == dataset.reference_image_id:
                continue
            image_param_by_id[image_id] = offset
            offset += 6

        point_param_by_track: dict[int, int] = {}
        p_offset = offset
        for p in dataset.points_init:
            point_param_by_track[p.track_id] = p_offset
            p_offset += 3

        return cls(
            dataset=dataset,
            normalizer=normalizer,
            image_param_by_id=image_param_by_id,
            point_param_by_track=point_param_by_track,
            n_image_params=offset,
            n_point_params=p_offset - offset,
        )

    @property
    def n_params(self) -> int:
        return self.n_image_params + self.n_point_params

    def pack(self) -> "object":
        import numpy as np  # type: ignore

        x = np.zeros(self.n_params, dtype=float)
        for p in self.dataset.points_init:
            idx = self.point_param_by_track[p.track_id]
            nl, na, nh = self.normalizer.normalize(p.lon, p.lat, p.h)
            x[idx : idx + 3] = [nl, na, nh]
        return x

    def unpack_point(self, x: "object", track_id: int) -> tuple[float, float, float]:
        idx = self.point_param_by_track[track_id]
        nl, na, nh = float(x[idx]), float(x[idx + 1]), float(x[idx + 2])
        return self.normalizer.denormalize(nl, na, nh)

    def get_image_params(self, x: "object", image_id: int) -> tuple[float, float, float, float, float, float]:
        if image_id == self.dataset.reference_image_id:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        idx = self.image_param_by_id[image_id]
        return tuple(float(v) for v in x[idx : idx + 6])  # type: ignore[return-value]
