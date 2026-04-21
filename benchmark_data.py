"""
基于 DATA 样本的 CNT 检测 benchmark 与参数校准脚本。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.core.analyzer_core import CNTAnalyzer
from src.core.stats_compat import mannwhitneyu, ttest_ind
from src.core.utils import (
    SCALE_BAR_DEFAULT_UM,
    CALIBRATED_BLUR_KERNEL,
    CALIBRATED_ADAPTIVE_BLOCK,
    CALIBRATED_ADAPTIVE_C,
)


FILTERS = {
    "min_length_um": 4.0,
    "max_length_um": 200.0,
    "min_slenderness": 10.0,
    "detection_profile": "balanced",
    "split_mode": "off",
}

PARAM_GRID = {
    "blur_kernel": [7, 9, 11],
    "adaptive_block": [15, 17, 19, 21],
    "adaptive_c": [1, 2, 3],
}

REPRESENTATIVE_FILES = {
    "23-0-60-500-1.jpg",
    "23-0-60-500-10.jpg",
    "22-0-60-500-1.jpg",
    "22-0-60-500-6.jpg",
    "22-0-60-500-10.jpg",
    "22-0-50-500-16.jpg",
}

# Spatial-first group-separation metrics: designed for the use case where the
# experiment group is expected to be more dispersed / uniform than the base
# group, while structural thickness metrics stay as secondary penalties.
SEPARATION_METRIC_CONFIG = {
    "dispersed_ratio": {"direction": "high", "weight": 0.20},
    "grid_density_cv": {"direction": "low", "weight": 0.25},
    "uniformity_score": {"direction": "high", "weight": 0.25},
    "agglomerated_area_ratio": {"direction": "low", "weight": 0.15},
    "width_p90_um": {"direction": "low", "weight": 0.10},
    "long_thick_ratio": {"direction": "low", "weight": 0.05},
}


def list_data_images(data_root: Path) -> List[Path]:
    return sorted(p for p in data_root.rglob("*.jpg") if p.is_file())


def legacy_suggest_params(original_image: np.ndarray, detection_profile: str = "balanced") -> dict:
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_mask = gray > otsu_thresh

    bright_mean = float(gray[bright_mask].mean()) if bright_mask.any() else float(gray.mean())
    dark_mean = float(gray[~bright_mask].mean()) if (~bright_mask).any() else float(gray.mean())
    delta = abs(bright_mean - dark_mean)

    profile = (detection_profile or "balanced").lower()
    if delta < 30:
        blur_kernel = 7
        adaptive_block = 21
        adaptive_c = 1
    elif delta < 60:
        blur_kernel = 9
        adaptive_block = 19
        adaptive_c = 2
    else:
        blur_kernel = 9
        adaptive_block = 15
        adaptive_c = 2

    if profile == "precision":
        blur_kernel += 2
        adaptive_block += 2
        adaptive_c += 1
    elif profile == "recall":
        blur_kernel -= 2
        adaptive_block -= 2
        adaptive_c -= 1

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if adaptive_block % 2 == 0:
        adaptive_block += 1

    blur_kernel = max(7, min(15, blur_kernel))
    adaptive_block = max(11, min(31, adaptive_block))
    adaptive_c = max(1, min(5, adaptive_c))
    return {
        "blur_kernel": blur_kernel,
        "adaptive_block": adaptive_block,
        "adaptive_c": adaptive_c,
    }


def foreground_ratio_score(value: float, low: float = 0.015, high: float = 0.08) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, 1.0 - (low - value) / low)
    return max(0.0, 1.0 - (value - high) / high)


def stability_score(base_count: int, shifted_count: int) -> float:
    delta = abs(int(shifted_count) - int(base_count))
    denom = max(10, int(base_count))
    return max(0.0, 1.0 - min(1.0, delta / float(denom)))


def summarize_counts(records: List[dict]) -> dict:
    counts = [item["count"] for item in records]
    if not counts:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    return {
        "n": len(counts),
        "mean": round(mean(counts), 2),
        "std": round(pstdev(counts), 2),
        "min": min(counts),
        "max": max(counts),
    }


def summarize_metric(records: List[dict], key: str) -> dict:
    values = [float(item.get(key, 0.0)) for item in records]
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def resolve_group_role(group_name: str) -> str:
    normalized = str(group_name or "").strip().lower()
    return "base" if normalized.startswith("base") else "experiment"


def resolve_group_bucket(group_name: str) -> str:
    label = str(group_name or "").strip()
    return label.rsplit("-", 1)[-1] if "-" in label else label


def compute_cohens_d(exp_values: List[float], base_values: List[float]) -> Optional[float]:
    exp = np.array(exp_values, dtype=float)
    base = np.array(base_values, dtype=float)
    if exp.size < 2 or base.size < 2:
        return None

    exp_var = float(np.var(exp, ddof=1))
    base_var = float(np.var(base, ddof=1))
    pooled_den = exp.size + base.size - 2
    if pooled_den <= 0:
        return None

    pooled_var = (((exp.size - 1) * exp_var) + ((base.size - 1) * base_var)) / pooled_den
    pooled_std = float(np.sqrt(max(pooled_var, 1e-12)))
    return float((float(np.mean(exp)) - float(np.mean(base))) / pooled_std)


def build_group_separation_report(records: List[dict], metric_config: Optional[Dict[str, dict]] = None) -> dict:
    config = metric_config or SEPARATION_METRIC_CONFIG
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for item in records:
        bucket = resolve_group_bucket(item.get("group", ""))
        role = resolve_group_role(item.get("group", ""))
        grouped.setdefault(bucket, {}).setdefault(role, []).append(item)

    pair_reports = {}
    pair_scores = []

    for bucket, bucket_groups in sorted(grouped.items()):
        base_records = bucket_groups.get("base", [])
        exp_records = bucket_groups.get("experiment", [])
        if not base_records or not exp_records:
            continue

        metric_reports = {}
        ranking = []
        weighted_effect_sum = 0.0
        total_weight = 0.0

        for metric_name, metric_meta in config.items():
            direction = str(metric_meta.get("direction", "high")).lower()
            weight = float(metric_meta.get("weight", 1.0) or 0.0)
            base_values = [float(item.get(metric_name, 0.0)) for item in base_records]
            exp_values = [float(item.get(metric_name, 0.0)) for item in exp_records]
            base_summary = summarize_metric(base_records, metric_name)
            exp_summary = summarize_metric(exp_records, metric_name)
            diff = float(exp_summary["mean"] - base_summary["mean"])
            oriented_gap = diff if direction == "high" else -diff

            try:
                _, t_pvalue = ttest_ind(
                    np.array(base_values, dtype=float),
                    np.array(exp_values, dtype=float),
                    equal_var=False,
                    nan_policy="omit",
                )
                if not np.isfinite(t_pvalue):
                    t_pvalue = None
            except (TypeError, ValueError, FloatingPointError):
                t_pvalue = None

            try:
                _, mw_pvalue = mannwhitneyu(
                    np.array(base_values, dtype=float),
                    np.array(exp_values, dtype=float),
                    alternative="two-sided",
                )
                if not np.isfinite(mw_pvalue):
                    mw_pvalue = None
            except (TypeError, ValueError, FloatingPointError):
                mw_pvalue = None

            cohens_d = compute_cohens_d(exp_values, base_values)
            oriented_effect = None
            if cohens_d is not None:
                oriented_effect = cohens_d if direction == "high" else -cohens_d
                weighted_effect_sum += weight * float(np.clip(oriented_effect, -3.0, 3.0))
                total_weight += weight

            best_pvalue = min(
                value for value in (t_pvalue, mw_pvalue)
                if value is not None
            ) if any(value is not None for value in (t_pvalue, mw_pvalue)) else None

            metric_reports[metric_name] = {
                "direction_for_experiment_better": direction,
                "weight": weight,
                "base_mean": round(base_summary["mean"], 4),
                "exp_mean": round(exp_summary["mean"], 4),
                "diff_exp_minus_base": round(diff, 4),
                "oriented_gap_for_exp_better": round(oriented_gap, 4),
                "cohens_d_exp_vs_base": None if cohens_d is None else round(cohens_d, 4),
                "oriented_effect_for_exp_better": None if oriented_effect is None else round(oriented_effect, 4),
                "t_pvalue": None if t_pvalue is None else round(float(t_pvalue), 6),
                "mw_pvalue": None if mw_pvalue is None else round(float(mw_pvalue), 6),
            }
            ranking.append(
                {
                    "metric": metric_name,
                    "direction_for_experiment_better": direction,
                    "weight": weight,
                    "oriented_gap_for_exp_better": round(oriented_gap, 4),
                    "oriented_effect_for_exp_better": None if oriented_effect is None else round(oriented_effect, 4),
                    "best_pvalue": None if best_pvalue is None else round(float(best_pvalue), 6),
                }
            )

        ranking.sort(
            key=lambda item: (
                item["oriented_gap_for_exp_better"] <= 0,
                1.0 if item["best_pvalue"] is None else item["best_pvalue"],
                -abs(item["oriented_effect_for_exp_better"] or 0.0),
            )
        )
        pair_score = weighted_effect_sum / total_weight if total_weight > 0 else 0.0
        pair_scores.append(pair_score)
        pair_reports[bucket] = {
            "base_group": next((item.get("group") for item in base_records), f"base-{bucket}"),
            "exp_group": next((item.get("group") for item in exp_records), f"experiment-{bucket}"),
            "pair_score": round(pair_score, 5),
            "metrics": metric_reports,
            "ranked_metrics": ranking,
        }

    overall_score = float(mean(pair_scores)) if pair_scores else 0.0
    return {
        "metric_config": config,
        "overall_score": round(overall_score, 5),
        "pairs": pair_reports,
    }


def collect_candidate_metrics(analyzer: CNTAnalyzer) -> List[dict]:
    metrics: List[dict] = []
    binary = analyzer.binary_image
    if binary is None or binary.size == 0:
        return metrics

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1:
            continue

        x_min = int(contour[:, 0, 0].min())
        x_max = int(contour[:, 0, 0].max())
        y_min = int(contour[:, 0, 1].min())
        y_max = int(contour[:, 0, 1].max())

        x_min = max(0, x_min - 1)
        y_min = max(0, y_min - 1)
        x_max = min(binary.shape[1], x_max + 2)
        y_max = min(binary.shape[0], y_max + 2)

        cnt_region = binary[y_min:y_max, x_min:x_max]
        relative_contour = contour - np.array([x_min, y_min])
        mask = np.zeros(cnt_region.shape, dtype=np.uint8)
        cv2.drawContours(mask, [relative_contour], 0, 255, -1)
        cnt_binary = cnt_region & mask

        candidate = analyzer._build_cnt_candidate(
            cnt_binary,
            x_offset=x_min,
            y_offset=y_min,
            profile="balanced",
            bypass_endpoint_filter=True,
        )
        if candidate is None:
            continue

        area = float(candidate.get("area", 0.0))
        bbox_area = float(max(1, cnt_binary.shape[0] * cnt_binary.shape[1]))
        fill_ratio = area / bbox_area
        slenderness = float(candidate.get("slenderness") or 999.0)

        metrics.append(
            {
                "endpoint_count": int(candidate.get("endpoint_count", 0)),
                "fill_ratio": float(fill_ratio),
                "slenderness": float(slenderness),
            }
        )

    return metrics


def run_current_pipeline(image_path: Path, params: dict) -> dict:
    analyzer = CNTAnalyzer()
    analyzer.load_image(str(image_path))
    scale_result = analyzer.apply_detected_scale(default_micrometers=SCALE_BAR_DEFAULT_UM)
    analyzer.preprocess(**params)
    analyzer.detect_cnts_hybrid(**FILTERS)
    stats = analyzer.get_statistics()
    dispersed_stats = analyzer.get_dispersed_statistics()
    spatial = stats.get("spatial_distribution") or {}
    aggregation_scores = spatial.get("aggregation_scores") or {}
    framework = analyzer.get_evaluation_framework(stats=stats, dispersed_stats=dispersed_stats)
    uniformity = framework.get("uniformity") or {}
    thick_bundle = framework.get("thick_bundle") or {}
    long_cnt = framework.get("long_cnt") or {}
    agglomeration = framework.get("agglomeration") or {}

    valid_mask = analyzer._get_valid_analysis_mask()
    fg_ratio = 0.0
    if analyzer.binary_image is not None and np.any(valid_mask):
        fg_ratio = float(np.count_nonzero((analyzer.binary_image > 0) & valid_mask) / max(1, np.count_nonzero(valid_mask)))

    candidate_metrics = collect_candidate_metrics(analyzer)
    shifted_params = dict(params)
    shifted_params["adaptive_c"] = int(params["adaptive_c"]) + 1
    shifted = CNTAnalyzer()
    shifted.load_image(str(image_path))
    shifted.apply_detected_scale(default_micrometers=SCALE_BAR_DEFAULT_UM)
    shifted.preprocess(**shifted_params)
    shifted.detect_cnts_hybrid(**FILTERS)

    return {
        "file": image_path.name,
        "group": image_path.parent.name,
        "params": dict(params),
        "scale_detected": bool(scale_result.get("applied")),
        "scale_status": analyzer.get_scale_status(),
        "scale_pixels": None if not analyzer.scale_bar_info else analyzer.scale_bar_info.get("pixels"),
        "count": int(stats.get("count", 0)),
        "length_mean": round(float(stats.get("length_mean", 0.0)), 3) if stats else 0.0,
        "fg_ratio": round(fg_ratio, 5),
        "candidate_metrics": candidate_metrics,
        "shifted_count": int(len(shifted.measurements)),
        "dispersed_ratio": round(float(dispersed_stats.get("dispersed_ratio", 0.0)), 5),
        "agglomerated_count": int(dispersed_stats.get("agglomerated_count", 0)),
        "shadow_aggregation_score": round(float(aggregation_scores.get("overall", 0.0)), 3),
        "grid_density_cv": round(float(spatial.get("grid_density_cv", 0.0) or 0.0), 5),
        "morans_i": round(float(spatial.get("morans_i", 0.0) or 0.0), 5),
        "uniformity_score": round(float(uniformity.get("score", 0.0) or 0.0), 3),
        "agglomerated_area_ratio": round(float(agglomeration.get("agglomerated_area_ratio", 0.0) or 0.0), 5),
        "width_p90_um": round(float(thick_bundle.get("width_p90_um", 0.0) or 0.0), 5),
        "hybrid_score": round(float(framework.get("hybrid_score", 0.0) or 0.0), 3),
        "skeleton_length_mean_um": round(float(long_cnt.get("skeleton_length_mean_um", stats.get("length_mean", 0.0)) or 0.0), 3),
        "long_thick_ratio": round(float(long_cnt.get("long_thick_ratio", 0.0) or 0.0), 5),
    }


def run_legacy_pipeline(image_path: Path) -> dict:
    analyzer = CNTAnalyzer()
    analyzer.load_image(str(image_path))
    analyzer.apply_detected_scale(default_micrometers=SCALE_BAR_DEFAULT_UM)

    legacy_params = legacy_suggest_params(analyzer.original_image, detection_profile="balanced")
    legacy_gray = cv2.cvtColor(analyzer.original_image, cv2.COLOR_BGR2GRAY)
    analyzer.analysis_gray_image = legacy_gray
    analyzer.analysis_image = cv2.cvtColor(legacy_gray, cv2.COLOR_GRAY2BGR)
    analyzer._set_scale_exclusion_rect(None)

    analyzer.preprocess(**legacy_params)
    analyzer.detect_cnts_hybrid(**FILTERS)
    stats = analyzer.get_statistics()
    return {
        "file": image_path.name,
        "group": image_path.parent.name,
        "params": legacy_params,
        "count": int(stats.get("count", 0)),
        "length_mean": round(float(stats.get("length_mean", 0.0)), 3) if stats else 0.0,
    }


def score_records(records: List[dict]) -> dict:
    fg_scores = []
    skeleton_scores = []
    shape_scores = []
    stability_scores = []

    for item in records:
        fg_scores.append(foreground_ratio_score(item["fg_ratio"]))

        metrics = item["candidate_metrics"]
        if metrics:
            bad_endpoint_ratio = sum(m["endpoint_count"] > 3 for m in metrics) / len(metrics)
            bad_shape_ratio = sum((m["fill_ratio"] > 0.55 and m["slenderness"] < 8.0) for m in metrics) / len(metrics)
        else:
            bad_endpoint_ratio = 1.0
            bad_shape_ratio = 1.0

        skeleton_scores.append(max(0.0, 1.0 - bad_endpoint_ratio))
        shape_scores.append(max(0.0, 1.0 - bad_shape_ratio))
        stability_scores.append(stability_score(item["count"], item["shifted_count"]))

    fg_mean = float(mean(fg_scores)) if fg_scores else 0.0
    skeleton_mean = float(mean(skeleton_scores)) if skeleton_scores else 0.0
    shape_mean = float(mean(shape_scores)) if shape_scores else 0.0
    stability_mean = float(mean(stability_scores)) if stability_scores else 0.0
    total = (
        0.35 * fg_mean +
        0.30 * skeleton_mean +
        0.20 * shape_mean +
        0.15 * stability_mean
    )
    return {
        "foreground_score": round(fg_mean, 5),
        "skeleton_score": round(skeleton_mean, 5),
        "shape_score": round(shape_mean, 5),
        "stability_score": round(stability_mean, 5),
        "total_score": round(total, 5),
    }


def build_dataset_report(current_records: List[dict], legacy_records: List[dict], best_params: dict) -> dict:
    legacy_map = {item["file"]: item for item in legacy_records}
    merged = []
    abnormal = []
    for current in current_records:
        before = legacy_map.get(current["file"], {})
        delta = current["count"] - int(before.get("count", 0))
        merged_item = {
            "file": current["file"],
            "group": current["group"],
            "legacy_count": int(before.get("count", 0)),
            "current_count": int(current["count"]),
            "count_delta": int(delta),
            "scale_detected": bool(current["scale_detected"]),
            "scale_status": current["scale_status"],
            "scale_pixels": current["scale_pixels"],
            "length_mean": current["length_mean"],
            "dispersed_ratio": current["dispersed_ratio"],
            "agglomerated_count": current["agglomerated_count"],
            "shadow_aggregation_score": current["shadow_aggregation_score"],
            "params": best_params,
        }
        merged.append(merged_item)

        if (
            not current["scale_detected"] or
            abs(delta) >= max(12, int(max(1, before.get("count", 0)) * 0.5)) or
            current["file"] in REPRESENTATIVE_FILES
        ):
            abnormal.append(merged_item)

    by_group = {}
    for group_name in sorted({item["group"] for item in merged}):
        group_records = [item for item in merged if item["group"] == group_name]
        by_group[group_name] = {
            "count_summary": summarize_counts(group_records),
            "dispersed_ratio_summary": summarize_metric(group_records, "dispersed_ratio"),
            "agglomerated_count_summary": summarize_metric(group_records, "agglomerated_count"),
            "shadow_aggregation_score_summary": summarize_metric(group_records, "shadow_aggregation_score"),
            "grid_density_cv_summary": summarize_metric(group_records, "grid_density_cv"),
            "morans_i_summary": summarize_metric(group_records, "morans_i"),
            "uniformity_score_summary": summarize_metric(group_records, "uniformity_score"),
            "agglomerated_area_ratio_summary": summarize_metric(group_records, "agglomerated_area_ratio"),
            "width_p90_um_summary": summarize_metric(group_records, "width_p90_um"),
            "hybrid_score_summary": summarize_metric(group_records, "hybrid_score"),
            "skeleton_length_mean_um_summary": summarize_metric(group_records, "skeleton_length_mean_um"),
            "long_thick_ratio_summary": summarize_metric(group_records, "long_thick_ratio"),
        }

    return {
        "best_params": best_params,
        "group_summary": by_group,
        "group_separation": build_group_separation_report(merged),
        "records": merged,
        "abnormal_files": abnormal,
    }


def main():
    parser = argparse.ArgumentParser(description="基于 DATA 样本进行 CNT benchmark 与参数校准")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parent / "DATA"))
    parser.add_argument("--output", default="", help="可选：将 JSON 结果写入文件")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    image_paths = list_data_images(data_root)
    if not image_paths:
        raise SystemExit(f"未在 {data_root} 找到 JPG 图像")

    grid_results = []
    for blur_kernel, adaptive_block, adaptive_c in itertools.product(
        PARAM_GRID["blur_kernel"],
        PARAM_GRID["adaptive_block"],
        PARAM_GRID["adaptive_c"],
    ):
        params = {
            "blur_kernel": int(blur_kernel),
            "adaptive_block": int(adaptive_block),
            "adaptive_c": int(adaptive_c),
            "threshold_invert": True,
        }
        records = [run_current_pipeline(path, params) for path in image_paths]
        score = score_records(records)
        separation = build_group_separation_report(records)
        grid_results.append(
            {
                "params": {
                    "blur_kernel": blur_kernel,
                    "adaptive_block": adaptive_block,
                    "adaptive_c": adaptive_c,
                },
                "score": score,
                "group_separation": separation,
            }
        )

    grid_results.sort(key=lambda item: item["score"]["total_score"], reverse=True)
    separation_sorted_results = sorted(
        grid_results,
        key=lambda item: float((item.get("group_separation") or {}).get("overall_score", 0.0)),
        reverse=True,
    )
    best_params = dict(grid_results[0]["params"])
    current_records = [run_current_pipeline(path, {**best_params, "threshold_invert": True}) for path in image_paths]
    legacy_records = [run_legacy_pipeline(path) for path in image_paths]

    report = {
        "data_root": str(data_root),
        "image_count": len(image_paths),
        "current_defaults": {
            "blur_kernel": CALIBRATED_BLUR_KERNEL,
            "adaptive_block": CALIBRATED_ADAPTIVE_BLOCK,
            "adaptive_c": CALIBRATED_ADAPTIVE_C,
        },
        "top_grid_results": grid_results[:5],
        "top_group_separation_results": separation_sorted_results[:5],
        "dataset_report": build_dataset_report(current_records, legacy_records, best_params),
    }

    output_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
