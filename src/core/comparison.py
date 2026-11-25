"""Headless group-comparison helpers shared by desktop and web frontends."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .stats_compat import mannwhitneyu, ttest_ind

logger = logging.getLogger(__name__)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def grade_core_metric(metric_key: str, value: Optional[float]) -> str:
    numeric = _safe_float(value, default=float("nan"))
    if not np.isfinite(numeric):
        return "未评"
    if metric_key == "dispersed_ratio":
        return "优" if numeric > 0.80 else "中" if numeric >= 0.60 else "差"
    if metric_key == "grid_density_cv":
        return "优" if numeric < 0.60 else "中" if numeric <= 1.00 else "差"
    if metric_key == "agglomerated_area_ratio":
        return "优" if numeric < 0.10 else "中" if numeric <= 0.20 else "差"
    if metric_key == "width_p90_um":
        return "优" if numeric < 1.20 else "中" if numeric <= 1.50 else "差"
    return "未评"


def build_core_metric_snapshot(
    stats: Optional[dict],
    dispersed_stats: Optional[dict] = None,
    framework: Optional[dict] = None,
) -> dict:
    stats = stats or {}
    dispersed_stats = dispersed_stats or {}
    framework = framework or {}
    spatial = stats.get("spatial_distribution") or {}
    thick_bundle = framework.get("thick_bundle") or {}
    agglomeration = framework.get("agglomeration") or {}
    long_cnt = framework.get("long_cnt") or {}
    uniformity = framework.get("uniformity") or {}

    total_count = _safe_int(stats.get("count", dispersed_stats.get("total_count", 0)))
    dispersed_count = _safe_int(dispersed_stats.get("dispersed_count", total_count), total_count)
    agglomerated_count = _safe_int(
        dispersed_stats.get("agglomerated_count", max(0, total_count - dispersed_count)),
        max(0, total_count - dispersed_count),
    )
    dispersed_ratio = _safe_float(
        dispersed_stats.get(
            "dispersed_ratio",
            dispersed_count / total_count if total_count else 0.0,
        )
    )
    grid_density_cv = _safe_float(spatial.get("grid_density_cv", 0.0))
    agglomerated_area_ratio = _safe_float(
        agglomeration.get("agglomerated_area_ratio", spatial.get("hotspot_area_ratio", 0.0))
    )
    width_p90_um = _safe_float(thick_bundle.get("width_p90_um", 0.0))
    uniformity_score = _safe_float(
        (spatial.get("uniformity_scores") or {}).get("overall", uniformity.get("score", 0.0))
    )
    dispersed_length_mean = _safe_float(
        (dispersed_stats.get("dispersed_length_stats") or {}).get(
            "length_mean", stats.get("length_mean", 0.0)
        )
    )
    skeleton_length_mean_um = _safe_float(
        long_cnt.get("skeleton_length_mean_um", dispersed_length_mean)
    )
    long_thick_count = _safe_float(long_cnt.get("long_thick_count", 0.0))
    long_thick_ratio = _safe_float(long_cnt.get("long_thick_ratio", 0.0))
    return {
        "total_count": total_count,
        "dispersed_count": dispersed_count,
        "agglomerated_count": agglomerated_count,
        "dispersed_ratio": dispersed_ratio,
        "grid_density_cv": grid_density_cv,
        "agglomerated_area_ratio": agglomerated_area_ratio,
        "width_p90_um": width_p90_um,
        "uniformity_score": uniformity_score,
        "skeleton_length_mean_um": skeleton_length_mean_um,
        "long_thick_count": long_thick_count,
        "long_thick_ratio": long_thick_ratio,
        "dispersed_ratio_grade": grade_core_metric("dispersed_ratio", dispersed_ratio),
        "grid_density_cv_grade": grade_core_metric("grid_density_cv", grid_density_cv),
        "agglomerated_area_ratio_grade": grade_core_metric(
            "agglomerated_area_ratio", agglomerated_area_ratio
        ),
        "width_p90_um_grade": grade_core_metric("width_p90_um", width_p90_um),
    }


def summarize_numeric_series(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "var": float(np.var(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "total": float(np.sum(array)),
    }


def aggregate_density_grids(
    density_grids: Iterable[np.ndarray], fallback_shape: Tuple[int, int] = (10, 10)
) -> np.ndarray:
    grids = []
    target_rows = target_cols = 0
    for raw in density_grids:
        grid = np.asarray(raw, dtype=float)
        if grid.ndim != 2 or not grid.size:
            continue
        target_rows = max(target_rows, int(grid.shape[0]))
        target_cols = max(target_cols, int(grid.shape[1]))
        grids.append(grid)
    if not grids:
        return np.zeros((max(1, fallback_shape[0]), max(1, fallback_shape[1])), dtype=float)
    resized = [
        grid
        if grid.shape == (target_rows, target_cols)
        else cv2.resize(grid.astype(np.float32), (target_cols, target_rows), interpolation=cv2.INTER_LINEAR)
        for grid in grids
    ]
    return np.mean(np.stack(resized, axis=0), axis=0)


def resolve_group_role(role: Optional[str], label: Optional[str] = None) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in {"base", "control", "baseline"}:
        return "base"
    if normalized in {"experiment", "exp", "variant", "treatment"}:
        return "experiment"
    if normalized in {"left", "right", "generic"}:
        return normalized
    label_text = str(label or "").strip().lower()
    if "base" in label_text or "基准" in label_text or "对照" in label_text:
        return "base"
    if "实验" in label_text or "exp" in label_text or "试验" in label_text:
        return "experiment"
    return "generic"


def group_role_names(role: Optional[str], label: Optional[str] = None) -> Tuple[str, str]:
    resolved = resolve_group_role(role, label)
    names = {
        "base": ("base组", "base"),
        "experiment": ("实验组", "实验"),
        "left": ("左侧对象", "左侧"),
        "right": ("右侧对象", "右侧"),
    }
    fallback = str(label or "").strip() or "对象"
    return names.get(resolved, (fallback, fallback))


def _normalize_hotspot_grid(grid: np.ndarray, percentile: float) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if not grid.size:
        return np.zeros((0, 0), dtype=float)
    positive = grid[grid > 0]
    if not positive.size:
        return np.zeros_like(grid, dtype=float)
    upper = float(np.percentile(positive, percentile))
    upper = max(upper, float(np.max(positive)), 1e-6) if upper <= 0 else upper
    return np.clip(grid / upper, 0.0, 1.0)


def _shadow_aggregation_score(spatial: dict) -> float:
    uniformity = spatial.get("uniformity_scores") or {}
    aggregation = spatial.get("aggregation_scores") or {}
    fallback = _safe_float(aggregation.get("overall", 100.0 - _safe_float(uniformity.get("overall", 0.0))))
    point = np.asarray(spatial.get("point_density_grid") or spatial.get("density_grid") or [], dtype=float)
    coverage = np.asarray(spatial.get("coverage_density_grid") or [], dtype=float)
    shadow = np.asarray(spatial.get("shadow_density_grid") or [], dtype=float)
    if not point.size and not coverage.size and not shadow.size:
        return fallback

    if not point.size:
        point = np.zeros_like(coverage if coverage.size else shadow, dtype=float)
    if not coverage.size:
        coverage = np.zeros_like(point, dtype=float)
    if not shadow.size:
        shadow = np.zeros_like(point, dtype=float)

    point_norm = _normalize_hotspot_grid(point, 86.0)
    coverage_norm = _normalize_hotspot_grid(coverage, 82.0)
    shadow_norm = _normalize_hotspot_grid(shadow, 84.0)
    hotspot_grid = np.clip(point_norm * 0.46 + coverage_norm * 0.20 + shadow_norm * 0.34, 0.0, 1.0)
    active_mask = (point > 0) | (coverage > 0) | (shadow > 0)
    active_scores = hotspot_grid[active_mask]
    if not active_scores.size or float(np.max(active_scores) - np.min(active_scores)) < 0.08:
        hotspot_mask = np.zeros_like(hotspot_grid, dtype=bool)
    else:
        percentile = 78.0 if active_scores.size >= 6 else 65.0
        threshold = float(np.percentile(active_scores, percentile))
        hotspot_mask = active_mask & (hotspot_grid >= threshold) & (hotspot_grid > 0)

    shadow_mean = _safe_float(spatial.get("shadow_density_mean", 0.0))
    shadow_support = float(np.mean(shadow_norm[hotspot_mask])) if np.any(hotspot_mask) else 0.0
    total = float(np.sum(point))
    peak_share = float(np.max(point) / total) if total > 0 else 0.0
    hotspot_mass = float(np.sum(point[hotspot_mask]) / total) if total > 0 and hotspot_mask.size else 0.0
    shadow_available = bool(np.any(shadow)) or shadow_mean > 0.0
    if not shadow_available:
        return fallback
    return float(np.clip(fallback * 0.25 + shadow_support * 38.0 + hotspot_mass * 23.0 + peak_share * 8.0 + shadow_mean * 18.0, 0, 100))


def summarize_group_results(group_label: str, results: List[dict], group_role: Optional[str] = None) -> dict:
    series = {key: [] for key in (
        "count", "dispersed_count", "agglomerated_count", "dispersed_ratio", "length_mean",
        "dispersed_length_mean", "skeleton_length_mean_um", "agglomerated_area_ratio",
        "width_p90_um", "nearest_neighbor_cv", "nearest_neighbor_index", "grid_density_cv",
        "morans_i", "grid_entropy", "occupancy_ratio", "uniformity_nn_score",
        "uniformity_grid_score", "uniformity_moran_score", "uniformity_score",
        "aggregation_nn_score", "aggregation_grid_score", "aggregation_moran_score",
        "aggregation_score", "shadow_aggregation_score", "hybrid_score", "hybrid_uniformity",
        "hybrid_thick_bundle", "long_cnt_score", "hybrid_agglomeration", "long_thick_count",
        "long_thick_ratio",
    )}
    grids = []
    details = []
    for result in results:
        stats = result.get("stats") or {}
        dispersed = result.get("dispersed_stats") or {}
        framework = result.get("evaluation_framework") or {}
        spatial = stats.get("spatial_distribution") or {}
        uniformity = spatial.get("uniformity_scores") or {}
        aggregation = spatial.get("aggregation_scores") or {}
        core = build_core_metric_snapshot(stats, dispersed, framework)
        long_metrics = framework.get("long_cnt") or {}
        values = {
            "count": _safe_float(stats.get("count", 0)),
            "dispersed_count": _safe_float(dispersed.get("dispersed_count", stats.get("count", 0))),
            "agglomerated_count": _safe_float(dispersed.get("agglomerated_count", 0)),
            "dispersed_ratio": _safe_float(dispersed.get("dispersed_ratio", 0)),
            "length_mean": _safe_float(stats.get("length_mean", 0)),
            "dispersed_length_mean": _safe_float((dispersed.get("dispersed_length_stats") or {}).get("length_mean", stats.get("length_mean", 0))),
            "skeleton_length_mean_um": _safe_float(core.get("skeleton_length_mean_um", 0)),
            "agglomerated_area_ratio": _safe_float(core.get("agglomerated_area_ratio", 0)),
            "width_p90_um": _safe_float(core.get("width_p90_um", 0)),
            "nearest_neighbor_cv": _safe_float(spatial.get("nearest_neighbor_cv", 0)),
            "nearest_neighbor_index": _safe_float(spatial.get("nearest_neighbor_index", 0)),
            "grid_density_cv": _safe_float(spatial.get("grid_density_cv", 0)),
            "morans_i": _safe_float(spatial.get("morans_i", 0)),
            "grid_entropy": _safe_float(spatial.get("grid_entropy", 0)),
            "occupancy_ratio": _safe_float(spatial.get("occupancy_ratio", 0)),
            "uniformity_nn_score": _safe_float(uniformity.get("nearest_neighbor", 0)),
            "uniformity_grid_score": _safe_float(uniformity.get("grid_density", 0)),
            "uniformity_moran_score": _safe_float(uniformity.get("moran", 0)),
            "uniformity_score": _safe_float(uniformity.get("overall", 0)),
            "aggregation_nn_score": _safe_float(aggregation.get("nearest_neighbor", 100 - _safe_float(uniformity.get("nearest_neighbor", 0)))),
            "aggregation_grid_score": _safe_float(aggregation.get("grid_density", 100 - _safe_float(uniformity.get("grid_density", 0)))),
            "aggregation_moran_score": _safe_float(aggregation.get("moran", 100 - _safe_float(uniformity.get("moran", 0)))),
            "aggregation_score": _safe_float(aggregation.get("overall", 100 - _safe_float(uniformity.get("overall", 0)))),
            "shadow_aggregation_score": _shadow_aggregation_score(spatial),
            "hybrid_score": _safe_float(framework.get("hybrid_score", 0)),
            "hybrid_uniformity": _safe_float((framework.get("uniformity") or {}).get("score", 0)),
            "hybrid_thick_bundle": _safe_float((framework.get("thick_bundle") or {}).get("score", 0)),
            "long_cnt_score": _safe_float(long_metrics.get("score", 0)),
            "hybrid_agglomeration": _safe_float((framework.get("agglomeration") or {}).get("score", 0)),
            "long_thick_count": _safe_float(long_metrics.get("long_thick_count", 0)),
            "long_thick_ratio": _safe_float(long_metrics.get("long_thick_ratio", 0)),
        }
        for key, value in values.items():
            series[key].append(value)
        details.append({"name": result.get("name", "image"), **values})
        grids.append(np.asarray(spatial.get("density_grid") or np.zeros((10, 10)), dtype=float))

    role = resolve_group_role(group_role, group_label)
    display, short = group_role_names(role, group_label)
    summary = {
        "label": group_label,
        "role": role,
        "role_display_name": display,
        "role_short_name": short,
        "image_count": len(results),
        "results": results,
        "file_details": details,
        "mean_density_grid": aggregate_density_grids(grids).tolist(),
    }
    top_level_map = {
        "count_stats": "count", "dispersed_count_stats": "dispersed_count",
        "agglomerated_count_stats": "agglomerated_count", "dispersed_ratio_stats": "dispersed_ratio",
        "length_mean_stats": "length_mean", "dispersed_length_mean_stats": "dispersed_length_mean",
        "skeleton_length_mean_stats": "skeleton_length_mean_um", "grid_density_cv_stats": "grid_density_cv",
        "agglomerated_area_ratio_stats": "agglomerated_area_ratio", "width_p90_um_stats": "width_p90_um",
        "uniformity_score_stats": "uniformity_score", "hybrid_score_stats": "hybrid_score",
        "long_thick_count_stats": "long_thick_count", "long_thick_ratio_stats": "long_thick_ratio",
    }
    summary.update({target: summarize_numeric_series(series[source]) for target, source in top_level_map.items()})
    spatial_keys = (
        "nearest_neighbor_cv", "nearest_neighbor_index", "grid_density_cv", "morans_i",
        "grid_entropy", "occupancy_ratio", "aggregation_nn_score", "aggregation_grid_score",
        "aggregation_moran_score", "aggregation_score", "shadow_aggregation_score",
        "uniformity_nn_score", "uniformity_grid_score", "uniformity_moran_score", "uniformity_score",
    )
    summary["spatial_stats"] = {key: summarize_numeric_series(series[key]) for key in spatial_keys}
    summary["hybrid_component_stats"] = {
        "uniformity": summarize_numeric_series(series["hybrid_uniformity"]),
        "thick_bundle": summarize_numeric_series(series["hybrid_thick_bundle"]),
        "long_cnt": summarize_numeric_series(series["long_cnt_score"]),
        "agglomeration": summarize_numeric_series(series["hybrid_agglomeration"]),
    }
    return summary


def compute_two_group_tests(base_values: Iterable[float], experiment_values: Iterable[float]) -> dict:
    base = np.asarray(list(base_values), dtype=float)
    experiment = np.asarray(list(experiment_values), dtype=float)
    result = {"t_stat": None, "t_pvalue": None, "mw_stat": None, "mw_pvalue": None}
    if not base.size or not experiment.size:
        return result
    try:
        statistic, pvalue = ttest_ind(base, experiment, equal_var=False, nan_policy="omit")
        result["t_stat"] = float(statistic) if np.isfinite(statistic) else None
        result["t_pvalue"] = float(pvalue) if np.isfinite(pvalue) else None
    except (TypeError, ValueError, FloatingPointError):
        pass
    try:
        statistic, pvalue = mannwhitneyu(base, experiment, alternative="two-sided")
        result["mw_stat"] = float(statistic) if np.isfinite(statistic) else None
        result["mw_pvalue"] = float(pvalue) if np.isfinite(pvalue) else None
    except (TypeError, ValueError, FloatingPointError):
        pass
    return result


def compute_group_comparison_tests(base_group: dict, experiment_group: dict) -> dict:
    keys = (
        "count", "dispersed_count", "dispersed_ratio", "hybrid_score", "long_thick_count",
        "long_thick_ratio", "agglomerated_area_ratio", "width_p90_um",
        "skeleton_length_mean_um", "nearest_neighbor_cv", "grid_density_cv",
        "uniformity_nn_score", "uniformity_grid_score", "uniformity_moran_score",
        "uniformity_score", "aggregation_nn_score", "aggregation_grid_score",
        "aggregation_moran_score", "aggregation_score", "shadow_aggregation_score",
    )
    return {
        key: compute_two_group_tests(
            (_safe_float(item.get(key, 0)) for item in base_group.get("file_details", [])),
            (_safe_float(item.get(key, 0)) for item in experiment_group.get("file_details", [])),
        )
        for key in keys
    }
