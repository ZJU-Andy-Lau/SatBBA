from satbba.core.overlap import _sample_boundary_points


def test_sample_boundary_points_clockwise_perimeter() -> None:
    pts = _sample_boundary_points(100, 50)

    # Starts at top-left.
    assert pts[0] == (0.0, 0.0)

    # Points should walk perimeter without jumping back to top after reaching bottom.
    # Find first bottom-edge point.
    bottom_idx = next(i for i, (_, r) in enumerate(pts) if r == 49.0)
    # After entering bottom edge, no point should go back to top edge (r == 0).
    assert all(r != 0.0 for _, r in pts[bottom_idx + 1 :])

    # Ensure there are at least 8 perimeter samples.
    assert len(pts) >= 8
