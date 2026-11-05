"""Numerical, boundary and resource regressions for the shared analysis engine."""

from itertools import combinations
import tracemalloc

import cv2
import numpy as np
import pytest

from src.core.analyzer_core import CNTAnalyzer
from src.core.workflow import DetectionSettings


@pytest.mark.parametrize("shape", [(1, 1), (1, 11), (11, 1), (3, 3), (8, 13)])
def test_full_foreground_skeleton_terminates_with_background_border(shape, monkeypatch):
    erode = cv2.erode
    calls = 0

    def bounded_erode(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert calls <= max(shape) + 1, "erosion must shrink a full foreground mask"
        return erode(*args, **kwargs)

    monkeypatch.setattr(cv2, "erode", bounded_erode)
    result = CNTAnalyzer()._skeletonize(np.full(shape, 255, np.uint8))
    assert result.shape == shape
    assert result.dtype == np.uint8
    assert np.any(result)
    np.testing.assert_array_equal(result, result[::-1, ::-1])
    if shape == (3, 3):
        np.testing.assert_array_equal(result, [[255, 0, 255], [0, 255, 0], [255, 0, 255]])


def test_coverage_is_union_for_overlapping_nested_and_duplicate_contours():
    contours = [
        np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.int32)
        for x1, y1, x2, y2 in [(1, 1, 7, 7), (4, 1, 10, 7), (2, 2, 3, 3), (1, 1, 7, 7)]
    ]
    expected = np.zeros((10, 12), np.uint8)
    expected[1:8, 1:11] = 255
    analyzer = CNTAnalyzer()
    mask = analyzer._build_coverage_mask(contours, 12, 10)
    np.testing.assert_array_equal(mask, expected)
    coverage = analyzer._build_coverage_ratio_grid(contours, 12, 10, 1)
    assert coverage[0, 0] == pytest.approx(70 / 120)


def test_full_foreground_width_is_finite_and_uses_exterior_background():
    analyzer = CNTAnalyzer()
    binary = np.full((3, 13), 255, np.uint8)
    skeleton = np.zeros_like(binary)
    skeleton[1, 2:11] = 255
    assert analyzer._measure_width(skeleton, binary) == {"mean": 4.0, "median": 4.0, "iqr": 0.0}


@pytest.mark.parametrize("count", [0, 1, 2, 255, 256, 257, 600])
def test_nearest_neighbor_matches_exhaustive_distances(count):
    points = np.random.default_rng(20261105).uniform(-100, 100, (count, 2))
    if count > 256:
        points[256] = points[0]  # A duplicate in a different tile is a valid zero distance.
    result = CNTAnalyzer()._calculate_nearest_neighbor_stats(points, 200, 200)
    if count < 2:
        assert all(value == 0.0 for value in result.values())
        return
    distances = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    expected_mean = 0.5 / np.sqrt(count / 40000)
    assert result == {
        "nearest_neighbor_mean_px": float(nearest.mean()),
        "nearest_neighbor_std_px": float(nearest.std()),
        "nearest_neighbor_cv": float(nearest.std() / nearest.mean()),
        "nearest_neighbor_expected_mean_px": expected_mean,
        "nearest_neighbor_index": float(nearest.mean() / expected_mean),
    }


def test_nearest_neighbor_temporary_memory_is_bounded():
    points = np.random.default_rng(7).uniform(0, 2000, (2000, 2))
    tracemalloc.start()
    try:
        CNTAnalyzer()._calculate_nearest_neighbor_stats(points, 2000, 2000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The former full distance matrix needs over 100 MB for this input.
    assert peak < 8 * 1024 * 1024


def _line_candidate(x, y, *, vertical=False):
    skeleton = np.zeros((3, 13), np.uint8)
    skeleton[1, 1:12] = 255
    contour = np.array([[[0, 0]], [[12, 0]], [[12, 2]], [[0, 2]]], np.int32)
    if vertical:
        skeleton = skeleton.T.copy()
        contour = contour[..., ::-1].copy()
    return {
        "skeleton": skeleton,
        "contour_local": contour + np.array([x, y], np.int32),
        "skeleton_bbox_local": (x, y),
        "length_pixels": 10.0,
        "area": 24.0,
        "width_median_px": 2.0,
    }


def test_endpoint_index_keeps_all_close_pairs_and_pair_order():
    rng = np.random.default_rng(13)
    candidates = [_line_candidate(int(x), int(y)) for x, y in rng.integers(-100, 100, (70, 2))]
    # Exact distance boundary, a transitive chain, and intersecting orientations.
    candidates += [_line_candidate(0, 0), _line_candidate(15, 0), _line_candidate(30, 0),
                   _line_candidate(15, 0, vertical=True), _line_candidate(15, 0)]
    analyzer = CNTAnalyzer()
    indexed = list(analyzer._iter_nearby_candidate_pairs(candidates, 5.0))
    close = {
        (i, j) for i, j in combinations(range(len(candidates)), 2)
        if analyzer._candidate_pair_metrics(candidates[i], candidates[j])["distance"] <= 5.0
    }
    assert close.issubset(set(indexed))
    assert indexed == sorted(set(indexed))
    assert len(indexed) < len(candidates) * (len(candidates) - 1) // 4


def test_indexed_merge_matches_exhaustive_groups_and_measurements(monkeypatch):
    candidates = [_line_candidate(x, y, vertical=vertical) for x, y, vertical in [
        (0, 0, False), (15, 0, False), (30, 0, False), (15, 0, True),
        (100, 40, False), (115, 40, False), (-100, 100, True), (80, -30, False),
    ]]
    analyzer = CNTAnalyzer()
    exhaustive = CNTAnalyzer()
    monkeypatch.setattr(exhaustive, "_iter_nearby_candidate_pairs",
                        lambda items, distance: combinations(range(len(items)), 2))
    expected = exhaustive._merge_nearby_cnt_candidates(candidates, 5.0, "balanced")
    actual = analyzer._merge_nearby_cnt_candidates(candidates, 5.0, "balanced")
    assert len(actual) == len(expected)
    assert len(actual) < len(candidates)
    for result, reference in zip(actual, expected):
        assert result.keys() == reference.keys()
        for key in result:
            np.testing.assert_equal(result[key], reference[key])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scale_rejects_nonfinite_values_without_changing_calibration(value):
    analyzer = CNTAnalyzer()
    analyzer.set_scale(10.0, 2.0)
    for pixels, micrometers in [(value, 2.0), (10.0, value)]:
        with pytest.raises(ValueError, match="有限正数"):
            analyzer.set_scale(pixels, micrometers)
        assert analyzer.scale_um_per_pixel == 0.2


@pytest.mark.parametrize("field", ["min_length_um", "max_length_um", "min_slenderness", "merge_distance_px"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_detection_settings_reject_nonfinite_values(field, value):
    with pytest.raises(ValueError, match="有限数值"):
        DetectionSettings(**{field: value}).as_kwargs()
