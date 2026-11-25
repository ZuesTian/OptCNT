"""JSON and export serializers for OptCNT web responses."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from ..core.models import CNTMeasurement, ParticleMeasurement


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, CNTMeasurement):
        return measurement_dict(value)
    if isinstance(value, ParticleMeasurement):
        return particle_dict(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def measurement_dict(measurement: CNTMeasurement) -> dict:
    contour = np.asarray(measurement.contour, dtype=np.int32)
    centroid = None
    bbox = None
    if contour.size:
        x, y, width, height = cv2.boundingRect(contour)
        bbox = {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}
        moments = cv2.moments(contour)
        if moments["m00"]:
            centroid = {
                "x": float(moments["m10"] / moments["m00"]),
                "y": float(moments["m01"] / moments["m00"]),
            }
    return {
        "id": int(measurement.id),
        "length_pixels": float(measurement.length_pixels),
        "length_um": float(measurement.length_um),
        "width_mean_um": _optional_float(measurement.width_mean_um),
        "width_median_um": _optional_float(measurement.width_median_um),
        "width_iqr_um": _optional_float(measurement.width_iqr_um),
        "slenderness": _optional_float(measurement.slenderness),
        "centroid": centroid,
        "bbox": bbox,
    }


def particle_dict(particle: ParticleMeasurement) -> dict:
    x, y, width, height = cv2.boundingRect(np.asarray(particle.contour, dtype=np.int32))
    return {
        "id": int(particle.id),
        "centroid": {"x": float(particle.centroid[0]), "y": float(particle.centroid[1])},
        "bbox": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
        "area_pixels": float(particle.area_pixels),
        "area_um2": float(particle.area_um2),
        "equivalent_diameter_pixels": float(particle.equivalent_diameter_pixels),
        "equivalent_diameter_um": float(particle.equivalent_diameter_um),
        "major_axis_um": float(particle.major_axis_um),
        "minor_axis_um": float(particle.minor_axis_um),
        "circularity": float(particle.circularity),
        "solidity": float(particle.solidity),
        "aspect_ratio": float(particle.aspect_ratio),
        "mean_local_contrast": float(particle.mean_local_contrast),
        "peak_local_contrast": float(particle.peak_local_contrast),
        "confidence": float(particle.confidence),
    }


def _optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _public_dispersed_stats(stats: dict) -> dict:
    result = {}
    for key, value in (stats or {}).items():
        if key == "dispersed_measurements":
            result["dispersed_ids"] = [int(item.id) for item in value]
        elif key == "agglomerated_measurements":
            result["agglomerated_ids"] = [int(item.id) for item in value]
        else:
            result[key] = json_safe(value)
    return result


def public_analysis_result(result: dict, *, include_measurements: bool = True) -> dict:
    payload = {
        "name": str(result.get("name", "image")),
        "stats": json_safe(result.get("stats") or {}),
        "dispersed_stats": _public_dispersed_stats(result.get("dispersed_stats") or {}),
        "evaluation_framework": json_safe(result.get("evaluation_framework") or {}),
        "core_metrics": json_safe(result.get("core_metrics") or {}),
        "measurement_count": int(len(result.get("measurements") or [])),
        "particle_statistics": json_safe(result.get("particle_statistics") or {}),
        "particle_measurements": [particle_dict(item) for item in result.get("particle_measurements") or []],
        "scale_status": json_safe(result.get("scale_status") or {}),
        "analysis_roi": json_safe(result.get("analysis_roi")),
        "preprocess_settings": json_safe(result.get("preprocess_settings") or {}),
        "detection_settings": json_safe(result.get("detection_settings") or {}),
    }
    if include_measurements:
        payload["measurements"] = [measurement_dict(item) for item in result.get("measurements") or []]
    return payload


def public_measurements(result: dict) -> dict:
    """Per-CNT detail page payload, fetched lazily by the browser."""
    return {
        "measurement_count": int(len(result.get("measurements") or [])),
        "measurements": [measurement_dict(item) for item in result.get("measurements") or []],
    }


def public_group_summary(summary: dict) -> dict:
    return json_safe({key: value for key, value in summary.items() if key != "results"})


def analysis_json_bytes(result: dict) -> bytes:
    return json.dumps(public_analysis_result(result), ensure_ascii=False, indent=2).encode("utf-8")


def analysis_csv_bytes(result: dict) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["ID", "长度(μm)", "宽度均值(μm)", "宽度中位数(μm)", "宽度IQR(μm)", "长宽比"])
    for measurement in result.get("measurements") or []:
        writer.writerow([
            int(measurement.id),
            f"{float(measurement.length_um):.4f}",
            _csv_number(measurement.width_mean_um),
            _csv_number(measurement.width_median_um),
            _csv_number(measurement.width_iqr_um),
            _csv_number(measurement.slenderness),
        ])
    particles = result.get("particle_measurements") or []
    if particles:
        writer.writerow([])
        writer.writerow(["颗粒ID", "中心X(px)", "中心Y(px)", "面积(μm²)", "等效直径(μm)", "圆度", "实心度", "置信度"])
        for particle in particles:
            writer.writerow([
                int(particle.id),
                f"{float(particle.centroid[0]):.4f}",
                f"{float(particle.centroid[1]):.4f}",
                f"{float(particle.area_um2):.4f}",
                f"{float(particle.equivalent_diameter_um):.4f}",
                f"{float(particle.circularity):.4f}",
                f"{float(particle.solidity):.4f}",
                f"{float(particle.confidence):.4f}",
            ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def _csv_number(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def analysis_report_bytes(result: dict) -> bytes:
    stats = result.get("stats") or {}
    spatial = stats.get("spatial_distribution") or {}
    framework = result.get("evaluation_framework") or {}
    dispersed = result.get("dispersed_stats") or {}
    particle_stats = result.get("particle_statistics") or {}
    roi = result.get("analysis_roi")
    roi_name = roi.get("name", "ROI") if isinstance(roi, dict) else "全图"
    score = (spatial.get("uniformity_scores") or {}).get("overall", 0.0)
    lines = [
        "========================================",
        "    CNT 图像分析报告",
        "========================================",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"分析图像: {result.get('name', 'image')}",
        f"分析区域: {roi_name}",
        f"比例尺: {(result.get('scale_status') or {}).get('um_per_pixel', 0.0):.6f} μm/pixel",
        "",
        "统计摘要",
        f"  CNT 数量: {int(stats.get('count', 0))}",
        f"  分散数量: {int(dispersed.get('dispersed_count', 0))}",
        f"  团聚数量: {int(dispersed.get('agglomerated_count', 0))}",
        f"  分散比例: {float(dispersed.get('dispersed_ratio', 0.0)):.2%}",
        f"  平均长度: {float(stats.get('length_mean', 0.0)):.3f} μm",
        f"  长度标准差: {float(stats.get('length_std', 0.0)):.3f} μm",
        "",
        "颗粒识别（独立口径）",
        f"  颗粒候选数量: {int(particle_stats.get('count', 0))}",
        f"  颗粒面积占比: {float(particle_stats.get('area_ratio', 0.0)):.3%}",
        f"  等效直径均值: {float(particle_stats.get('equivalent_diameter_mean_um', 0.0)):.3f} μm",
        f"  候选平均置信度: {float(particle_stats.get('confidence_mean', 0.0)):.1%}",
        "",
        "空间分布",
        f"  综合均匀性: {float(score):.2f} / 100",
        f"  最近邻 CV: {float(spatial.get('nearest_neighbor_cv', 0.0)):.4f}",
        f"  最近邻指数 NNI: {float(spatial.get('nearest_neighbor_index', 0.0)):.4f}",
        f"  网格 CV: {float(spatial.get('grid_density_cv', 0.0)):.4f}",
        f"  空间熵: {float(spatial.get('grid_entropy', 0.0)):.4f}",
        f"  Moran's I: {float(spatial.get('morans_i', 0.0)):.4f}",
        f"  网格占用率: {float(spatial.get('occupancy_ratio', 0.0)):.2%}",
        "",
        "四维评价框架",
        f"  A 均匀性: {float((framework.get('uniformity') or {}).get('score', 0.0)):.2f}",
        f"  B 粗管/束化: {float((framework.get('thick_bundle') or {}).get('score', 0.0)):.2f}",
        f"  C 长管: {float((framework.get('long_cnt') or {}).get('score', 0.0)):.2f}",
        f"  D 团聚: {float((framework.get('agglomeration') or {}).get('score', 0.0)):.2f}",
        f"  混合得分: {float(framework.get('hybrid_score', 0.0)):.2f}",
        "",
        "详细测量",
        "ID\t长度(μm)\t宽度均值(μm)\t宽度中位数(μm)\t长宽比",
    ]
    for item in result.get("measurements") or []:
        lines.append(
            f"{item.id}\t{item.length_um:.4f}\t{_text_number(item.width_mean_um)}\t"
            f"{_text_number(item.width_median_um)}\t{_text_number(item.slenderness)}"
        )
    lines.extend(["", "========================================", "            报告结束", "========================================"])
    return "\n".join(lines).encode("utf-8")


def _text_number(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def comparison_csv_bytes(result: dict) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["指标", "base组均值", "base组标准差", "实验组均值", "实验组标准差", "Welch p", "Mann-Whitney p", "方向"])
    for metric in result.get("metrics") or []:
        writer.writerow([
            metric["label"],
            metric["base"]["mean"],
            metric["base"]["std"],
            metric["experiment"]["mean"],
            metric["experiment"]["std"],
            metric.get("t_pvalue"),
            metric.get("mw_pvalue"),
            metric["direction"],
        ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")
