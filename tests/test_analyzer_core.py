import numpy as np
import pytest

from analyzer_core import CNTAnalyzer
from models import CNTMeasurement


def _make_measurement(measurement_id: int, length_um: float) -> CNTMeasurement:
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
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

    monkeypatch.setattr("analyzer_core.cv2.matchTemplate", fake_match_template)

    binary = np.full((10, 10), 255, dtype=np.uint8)
    result = analyzer._recognize_characters(binary, [(0, 0, 10, 10)])

    assert result == "1"
    assert calls["count"] == 1


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

    summary_values = iter([
        {"mean": 1.0, "std": 0.5, "cv": 0.5, "entropy": 0.8, "occupancy_ratio": 0.6, "dispersion_index": 1.1},
        {"mean": 0.25, "std": 0.1, "cv": 0.4, "entropy": 0.7, "occupancy_ratio": 0.5, "dispersion_index": 0.9},
    ])
    monkeypatch.setattr(analyzer, "_summarize_density_grid", lambda grid: next(summary_values))
    monkeypatch.setattr(analyzer, "_calculate_grid_morans_i", lambda grid: 0.2)

    result = analyzer.analyze_spatial_distribution(grid_size=4)

    assert "aggregation_scores" in result
    uniformity_scores = result["uniformity_scores"]
    aggregation_scores = result["aggregation_scores"]
    for key in ("nearest_neighbor", "grid_density", "moran", "overall"):
        assert aggregation_scores[key] == pytest.approx(100.0 - uniformity_scores[key])
