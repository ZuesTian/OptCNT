import cv2
import numpy as np
import pytest
from types import SimpleNamespace

import src.core.analyzer_core as analyzer_core_module
from src.core.analyzer_core import CNTAnalyzer
from src.core.models import CNTMeasurement


def _make_measurement(measurement_id: int, length_um: float) -> CNTMeasurement:
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
    return CNTMeasurement(
        id=measurement_id,
        length_pixels=length_um,
        length_um=length_um,
        contour=contour,
    )


def _make_square_measurement(measurement_id: int, length_um: float, center_x: int, center_y: int) -> CNTMeasurement:
    contour = np.array(
        [
            [[center_x - 2, center_y - 2]],
            [[center_x + 2, center_y - 2]],
            [[center_x + 2, center_y + 2]],
            [[center_x - 2, center_y + 2]],
        ],
        dtype=np.int32,
    )
    return CNTMeasurement(
        id=measurement_id,
        length_pixels=length_um,
        length_um=length_um,
        contour=contour,
    )


def test_parse_scale_value_accepts_numeric_text_and_rejects_out_of_range_values():
    analyzer = CNTAnalyzer()

    assert analyzer._parse_scale_value("Scale 10.5 um") == 10.5
    assert analyzer._parse_scale_value("0.05 um") is None
    assert analyzer._parse_scale_value("10000 um") is None


def test_record_manual_scale_builds_exclusion_rect_from_selected_line():
    analyzer = CNTAnalyzer()
    analyzer.original_image = np.zeros((240, 320, 3), dtype=np.uint8)

    analyzer.record_manual_scale(
        60.0,
        10.0,
        selection_line=((220.0, 190.0), (280.0, 190.0)),
    )

    rect = analyzer.scale_exclusion_rect
    assert rect is not None
    x1, y1, x2, y2 = rect
    assert x1 <= 220
    assert x2 >= 280
    assert y1 < 190
    assert y2 > 220
    assert analyzer.get_scale_status()["exclusion_enabled"] is True

    mask = analyzer.get_scale_exclusion_mask()
    assert mask is not None
    assert mask[200, 250] == 255


def test_build_scale_exclusion_rect_expands_to_cover_text_rect():
    analyzer = CNTAnalyzer()

    rect = analyzer._build_scale_exclusion_rect(
        (240, 320, 3),
        248,
        184,
        42,
        4,
        text_rect=(176, 190, 302, 226),
    )

    x1, y1, x2, y2 = rect
    assert x1 <= 176
    assert x2 >= 302
    assert y1 <= 190
    assert y2 >= 226


def test_get_statistics_uses_shared_length_distribution_bins():
    analyzer = CNTAnalyzer()
    analyzer.measurements = [
        _make_measurement(0, 4.9),
        _make_measurement(1, 5.0),
        _make_measurement(2, 14.9),
        _make_measurement(3, 15.0),
        _make_measurement(4, 29.9),
        _make_measurement(5, 30.0),
    ]
    analyzer.analyze_spatial_distribution = lambda roi=None: {"mock": True}

    stats = analyzer.get_statistics()

    assert stats["count"] == 6
    assert stats["length_distribution"] == {
        "<5μm": 1,
        "5-15μm": 2,
        "15-30μm": 2,
        ">30μm": 1,
    }
    assert stats["spatial_distribution"] == {"mock": True}


def test_recognize_characters_stops_after_high_confidence_template_match(monkeypatch):
    analyzer = CNTAnalyzer()
    analyzer.ocr_templates = {
        "1": [np.zeros((28, 28), dtype=np.uint8), np.ones((28, 28), dtype=np.uint8)],
        "2": [np.full((28, 28), 2, dtype=np.uint8)],
    }

    calls = {"count": 0}

    def fake_match_template(*args, **kwargs):
        calls["count"] += 1
        score = 0.95 if calls["count"] == 1 else 0.1
        return np.array([[score]], dtype=np.float32)

    monkeypatch.setattr("src.core.analyzer_core.cv2.matchTemplate", fake_match_template)

    binary = np.full((10, 10), 255, dtype=np.uint8)
    result = analyzer._recognize_characters(binary, [(0, 0, 10, 10)])

    assert result == "1"
    assert calls["count"] == 1


def test_suggest_preprocess_params_uses_configurable_noise_thresholds(monkeypatch):
    analyzer = CNTAnalyzer()
    gray = np.full((8, 8), 128, dtype=np.uint8)
    valid_mask = np.ones((8, 8), dtype=bool)

    monkeypatch.setattr(analyzer, "_get_analysis_region", lambda roi=None: (0, 8, 0, 8))
    monkeypatch.setattr(analyzer, "_get_analysis_gray_image", lambda: gray)
    monkeypatch.setattr(analyzer, "_get_valid_analysis_mask", lambda roi=None: valid_mask)
    monkeypatch.setattr(
        analyzer,
        "_suggest_adaptive_binary",
        lambda gray, blur_kernel, adaptive_block, adaptive_c: np.zeros_like(gray),
    )

    baseline = analyzer.suggest_preprocess_params()
    assert baseline["blur_kernel"] == 7

    monkeypatch.setattr(analyzer_core_module, "PREPROCESS_NOISE_LOW_THRESHOLD", -1.0)
    customized = analyzer.suggest_preprocess_params()
    assert customized["blur_kernel"] == 9


def test_preprocess_can_skip_skeleton_generation(monkeypatch):
    analyzer = CNTAnalyzer()
    analyzer.image = np.zeros((16, 16, 3), dtype=np.uint8)
    analyzer.analysis_gray_image = np.full((16, 16), 128, dtype=np.uint8)
    analyzer.skeleton_image = np.ones((4, 4), dtype=np.uint8)
    analyzer.skeleton_overlay = np.ones((16, 16, 3), dtype=np.uint8)

    calls = {"skeletonize": 0, "overlay": 0}

    def fake_skeletonize(binary):
        calls["skeletonize"] += 1
        return np.ones_like(binary)

    def fake_overlay(*args, **kwargs):
        calls["overlay"] += 1

    monkeypatch.setattr(analyzer, "_skeletonize", fake_skeletonize)
    monkeypatch.setattr(analyzer, "_generate_skeleton_overlay", fake_overlay)

    binary = analyzer.preprocess(generate_skeleton=False)

    assert binary.shape == (16, 16)
    assert calls == {"skeletonize": 0, "overlay": 0}
    assert analyzer.skeleton_image is None
    assert analyzer.skeleton_overlay is None


def test_skeletonize_prefers_opencv_thinning_when_available(monkeypatch):
    analyzer = CNTAnalyzer()
    binary = np.array(
        [
            [0, 255, 0],
            [255, 255, 255],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )
    expected = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    calls = {"count": 0}

    def fake_thinning(work):
        calls["count"] += 1
        assert work.dtype == np.uint8
        return expected

    monkeypatch.setattr(analyzer_core_module.cv2, "ximgproc", SimpleNamespace(thinning=fake_thinning), raising=False)

    result = analyzer._skeletonize(binary)

    assert calls["count"] == 1
    assert np.array_equal(result, expected)


def test_trace_skeleton_path_does_not_loop_on_cycle_graph():
    analyzer = CNTAnalyzer()
    neighbors = {
        (0, 0): [(0, 1), (1, 0)],
        (0, 1): [(0, 0), (1, 1)],
        (1, 1): [(0, 1), (1, 0)],
        (1, 0): [(1, 1), (0, 0)],
    }

    path = analyzer._trace_skeleton_path((0, 0), neighbors, 160.0)

    assert len(path) == len(set(path))
    assert len(path) <= len(neighbors)


def test_find_graph_diameter_path_prefers_longest_branch_to_branch_path():
    analyzer = CNTAnalyzer()
    neighbors = {
        (0, 0): [(0, 1)],
        (0, 1): [(0, 0), (0, 2)],
        (0, 2): [(0, 1), (0, 3), (1, 2)],
        (0, 3): [(0, 2)],
        (1, 2): [(0, 2), (2, 2)],
        (2, 2): [(1, 2), (3, 2)],
        (3, 2): [(2, 2)],
    }

    path = analyzer._find_graph_diameter_path(neighbors)

    assert {path[0], path[-1]} == {(0, 0), (3, 2)}
    assert analyzer._calculate_path_length(path) == pytest.approx(5.0)


def test_detect_cnts_hybrid_skips_candidates_whose_upper_bound_is_below_min_length(monkeypatch):
    analyzer = CNTAnalyzer()
    analyzer.image = np.zeros((16, 16, 3), dtype=np.uint8)
    analyzer.analysis_image = analyzer.image.copy()
    analyzer.binary_image = np.zeros((16, 16), dtype=np.uint8)
    analyzer.binary_image[4:7, 4:7] = 255
    analyzer.scale_um_per_pixel = 1.0

    calls = {"count": 0}

    def fake_build(*args, **kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(analyzer, "_build_cnt_candidate", fake_build)

    measurements = analyzer.detect_cnts_hybrid(min_length_um=10.0)

    assert measurements == []
    assert calls["count"] == 0


def test_build_cnt_candidate_uses_contour_correction_for_sinuous_shape():
    analyzer = CNTAnalyzer()
    cnt_binary = np.zeros((220, 420), dtype=np.uint8)
    points = []
    for x in range(20, 390, 6):
        y = int(90 + 45 * np.sin((x - 20) / 55.0))
        points.append((x, y))
    for start, end in zip(points, points[1:]):
        cv2.line(cnt_binary, start, end, 255, 9)

    candidate = analyzer._build_cnt_candidate(
        cnt_binary,
        x_offset=0,
        y_offset=0,
        profile="balanced",
        bypass_endpoint_filter=True,
    )

    assert candidate is not None
    assert candidate["length_pixels"] > 350.0
    assert candidate["width_median_px"] > 0.0


def test_analyze_spatial_distribution_returns_aggregation_scores(monkeypatch):
    analyzer = CNTAnalyzer()
    measurements = [_make_measurement(0, 10.0), _make_measurement(1, 12.0)]

    monkeypatch.setattr(
        analyzer,
        "_get_local_measurements",
        lambda roi=None: (measurements, 0, 0, 100, 80),
    )
    monkeypatch.setattr(
        analyzer,
        "_extract_spatial_distribution_inputs",
        lambda measurements, offset_x, offset_y: (
            [(10.0, 12.0), (48.0, 36.0)],
            [np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)],
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_calculate_nearest_neighbor_stats",
        lambda centroid_array, width, height: {
            "nearest_neighbor_mean": 4.0,
            "nearest_neighbor_std": 1.0,
            "nearest_neighbor_cv": 0.25,
            "nearest_neighbor_index": 0.92,
            "expected_nearest_neighbor": 4.3,
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_build_centroid_count_grid",
        lambda centroids, width, height, grid_size: np.ones((grid_size, grid_size), dtype=float),
    )
    monkeypatch.setattr(
        analyzer,
        "_build_coverage_ratio_grid",
        lambda local_contours, width, height, grid_size: np.full((grid_size, grid_size), 0.25, dtype=float),
    )
    monkeypatch.setattr(
        analyzer,
        "_build_shadow_density_grid",
        lambda offset_x, offset_y, width, height, grid_size, roi=None: np.full((grid_size, grid_size), 0.4, dtype=float),
    )

    summary_values = iter([
        {"mean": 1.0, "std": 0.5, "cv": 0.5, "entropy": 0.8, "occupancy_ratio": 0.6, "dispersion_index": 1.1},
        {"mean": 0.25, "std": 0.1, "cv": 0.4, "entropy": 0.7, "occupancy_ratio": 0.5, "dispersion_index": 0.9},
        {"mean": 0.4, "std": 0.2, "cv": 0.3, "entropy": 0.65, "occupancy_ratio": 0.55, "dispersion_index": 0.8},
    ])
    monkeypatch.setattr(analyzer, "_summarize_density_grid", lambda grid: next(summary_values))
    monkeypatch.setattr(analyzer, "_calculate_grid_morans_i", lambda grid: 0.2)

    result = analyzer.analyze_spatial_distribution(grid_size=4)

    assert "aggregation_scores" in result
    assert result["shadow_density_mean"] == pytest.approx(0.4)
    assert np.allclose(np.array(result["shadow_density_grid"], dtype=float), np.full((4, 4), 0.4))
    uniformity_scores = result["uniformity_scores"]
    aggregation_scores = result["aggregation_scores"]
    for key in ("nearest_neighbor", "grid_density", "moran", "overall"):
        assert aggregation_scores[key] == pytest.approx(100.0 - uniformity_scores[key])


def test_get_dispersed_statistics_filters_hotspots_and_supports_strictness(monkeypatch):
    analyzer = CNTAnalyzer()
    measurements = [
        _make_square_measurement(0, 10.0, 25, 25),
        _make_square_measurement(1, 20.0, 75, 25),
        _make_square_measurement(2, 30.0, 75, 75),
    ]
    analyzer.measurements = measurements

    monkeypatch.setattr(analyzer, "analyze_spatial_distribution", lambda roi=None: {"grid_size": 2})
    monkeypatch.setattr(
        analyzer,
        "_build_spatial_hotspot_masks",
        lambda spatial_distribution: {
            "hotspot_mask": np.array([[True, True], [False, False]], dtype=bool),
            "severe_mask": np.array([[True, False], [False, False]], dtype=bool),
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_get_local_measurements",
        lambda roi=None: (measurements, 0, 0, 100, 100),
    )

    stats = analyzer.get_dispersed_statistics()

    assert stats["strictness"] == "all_hotspots"
    assert stats["dispersed_count"] == 1
    assert stats["agglomerated_count"] == 2
    assert [m.id for m in stats["dispersed_measurements"]] == [2]
    assert stats["dispersed_length_stats"]["length_mean"] == pytest.approx(30.0)

    severe_only = analyzer.get_dispersed_statistics(strictness="severe_only")

    assert severe_only["strictness"] == "severe_only"
    assert severe_only["dispersed_count"] == 2
    assert severe_only["agglomerated_count"] == 1
    assert [m.id for m in severe_only["agglomerated_measurements"]] == [0]


def test_get_dispersed_statistics_marks_contour_overlap_as_agglomerated(monkeypatch):
    analyzer = CNTAnalyzer()
    measurement = CNTMeasurement(
        id=0,
        length_pixels=12.0,
        length_um=12.0,
        contour=np.array(
            [[[40, 10]], [[52, 10]], [[52, 22]], [[40, 22]]],
            dtype=np.int32,
        ),
    )
    analyzer.measurements = [measurement]

    monkeypatch.setattr(analyzer, "analyze_spatial_distribution", lambda roi=None: {"grid_size": 2})
    monkeypatch.setattr(
        analyzer,
        "_build_spatial_hotspot_masks",
        lambda spatial_distribution: {
            "hotspot_mask": np.array([[False, True], [False, False]], dtype=bool),
            "severe_mask": np.array([[False, True], [False, False]], dtype=bool),
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_get_local_measurements",
        lambda roi=None: ([measurement], 0, 0, 100, 100),
    )

    stats = analyzer.get_dispersed_statistics()

    assert stats["dispersed_count"] == 0
    assert stats["agglomerated_count"] == 1
    assert [m.id for m in stats["agglomerated_measurements"]] == [0]


def test_get_dispersed_statistics_returns_all_dispersed_when_masks_have_no_hotspots(monkeypatch):
    analyzer = CNTAnalyzer()
    measurements = [
        _make_square_measurement(0, 10.0, 20, 20),
        _make_square_measurement(1, 15.0, 80, 80),
    ]
    analyzer.measurements = measurements

    monkeypatch.setattr(analyzer, "analyze_spatial_distribution", lambda roi=None: {"grid_size": 2})
    monkeypatch.setattr(
        analyzer,
        "_build_spatial_hotspot_masks",
        lambda spatial_distribution: {
            "hotspot_mask": np.zeros((2, 2), dtype=bool),
            "severe_mask": np.zeros((2, 2), dtype=bool),
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_get_local_measurements",
        lambda roi=None: (measurements, 0, 0, 100, 100),
    )

    stats = analyzer.get_dispersed_statistics()

    assert stats["dispersed_count"] == 2
    assert stats["agglomerated_count"] == 0
    assert [m.id for m in stats["dispersed_measurements"]] == [0, 1]
