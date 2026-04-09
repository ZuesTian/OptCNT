"""
基于 DATA 样本的 CNT 检测 benchmark 与参数校准脚本。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import cv2
import numpy as np

from src.core.analyzer_core import CNTAnalyzer
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
        }

    return {
        "best_params": best_params,
        "group_summary": by_group,
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
        grid_results.append(
            {
                "params": {
                    "blur_kernel": blur_kernel,
                    "adaptive_block": adaptive_block,
                    "adaptive_c": adaptive_c,
                },
                "score": score,
            }
        )

    grid_results.sort(key=lambda item: item["score"]["total_score"], reverse=True)
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
        "dataset_report": build_dataset_report(current_records, legacy_records, best_params),
    }

    output_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
