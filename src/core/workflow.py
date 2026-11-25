"""Headless OptCNT workflows used by desktop orchestration and the web API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .analyzer_core import CNTAnalyzer
from .comparison import build_core_metric_snapshot
from .models import ROIRegion
from .utils import (
    CALIBRATED_ADAPTIVE_BLOCK,
    CALIBRATED_ADAPTIVE_C,
    CALIBRATED_BLUR_KERNEL,
    CNT_BRIDGE_STRENGTH_DEFAULT,
    CNT_MERGE_DISTANCE_DEFAULT_PX,
    SCALE_BAR_DEFAULT_UM,
)


@dataclass(frozen=True)
class PreprocessSettings:
    blur_kernel: int = CALIBRATED_BLUR_KERNEL
    adaptive_block: int = CALIBRATED_ADAPTIVE_BLOCK
    adaptive_c: int = CALIBRATED_ADAPTIVE_C
    bridge_strength: int = CNT_BRIDGE_STRENGTH_DEFAULT
    threshold_invert: bool = True

    def as_kwargs(self, *, generate_skeleton: bool = True) -> dict:
        return {
            "blur_kernel": _odd(self.blur_kernel, 1, 31),
            "adaptive_block": _odd(self.adaptive_block, 3, 99),
            "adaptive_c": int(np.clip(self.adaptive_c, -30, 30)),
            "bridge_strength": int(np.clip(self.bridge_strength, 0, 10)),
            "threshold_invert": bool(self.threshold_invert),
            "generate_skeleton": bool(generate_skeleton),
        }


@dataclass(frozen=True)
class DetectionSettings:
    min_length_um: float = 5.0
    max_length_um: float = 1000.0
    min_slenderness: float = 4.0
    detection_profile: str = "balanced"
    split_mode: str = "off"
    merge_distance_px: float = CNT_MERGE_DISTANCE_DEFAULT_PX

    def as_kwargs(self) -> dict:
        numeric_values = (self.min_length_um, self.max_length_um, self.min_slenderness, self.merge_distance_px)
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("检测参数必须为有限数值")
        profile = str(self.detection_profile).lower()
        split_mode = str(self.split_mode).lower()
        if profile not in {"precision", "balanced", "recall"}:
            raise ValueError("detection_profile 必须是 precision、balanced 或 recall")
        if split_mode not in {"off", "conservative", "aggressive"}:
            raise ValueError("split_mode 必须是 off、conservative 或 aggressive")
        if self.min_length_um < 0 or self.max_length_um <= 0:
            raise ValueError("长度范围必须为有效正数")
        if self.max_length_um < self.min_length_um:
            raise ValueError("最大长度不能小于最小长度")
        if self.min_slenderness < 0 or self.merge_distance_px < 0:
            raise ValueError("长宽比和合并距离不能小于 0")
        return {
            "min_length_um": float(self.min_length_um),
            "max_length_um": float(self.max_length_um),
            "min_slenderness": float(self.min_slenderness),
            "detection_profile": profile,
            "split_mode": split_mode,
            "merge_distance_px": float(self.merge_distance_px),
        }


@dataclass(frozen=True)
class BatchSettings:
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    scale_um: float = SCALE_BAR_DEFAULT_UM
    manual_scale_pixels: float = 0.0
    center_fraction: float = 0.75
    recognize_scale_text: bool = False


def _odd(value: int, minimum: int, maximum: int) -> int:
    resolved = int(np.clip(int(value), minimum, maximum))
    if resolved % 2 == 0:
        resolved = resolved + 1 if resolved < maximum else resolved - 1
    return resolved


def make_roi(
    analyzer: CNTAnalyzer,
    *,
    name: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> ROIRegion:
    if analyzer.image is None:
        raise ValueError("请先加载图像")
    image_height, image_width = analyzer.image.shape[:2]
    x = int(np.clip(x, 0, max(0, image_width - 1)))
    y = int(np.clip(y, 0, max(0, image_height - 1)))
    width = int(np.clip(width, 1, image_width - x))
    height = int(np.clip(height, 1, image_height - y))
    if width < 10 or height < 10:
        raise ValueError("ROI 宽高不能小于 10 像素")
    return ROIRegion(name=str(name), x=x, y=y, width=width, height=height)


def make_center_roi(analyzer: CNTAnalyzer, fraction: float, name: str = "中心区域") -> Optional[ROIRegion]:
    if analyzer.image is None:
        raise ValueError("请先加载图像")
    fraction = float(np.clip(fraction, 0.1, 1.0))
    if fraction >= 0.999:
        return None
    image_height, image_width = analyzer.image.shape[:2]
    width = max(1, int(round(image_width * fraction)))
    height = max(1, int(round(image_height * fraction)))
    return ROIRegion(
        name=name,
        x=(image_width - width) // 2,
        y=(image_height - height) // 2,
        width=width,
        height=height,
    )


def run_analysis(
    analyzer: CNTAnalyzer,
    *,
    name: str,
    preprocess: PreprocessSettings,
    detection: DetectionSettings,
    roi: Optional[ROIRegion] = None,
) -> dict:
    analyzer.preprocess(roi=roi, **preprocess.as_kwargs(generate_skeleton=True))
    measurements = analyzer.detect_cnts_hybrid(roi=roi, **detection.as_kwargs())
    particles = analyzer.detect_particles(roi=roi)
    stats, dispersed = analyzer.get_statistics_bundle(roi)
    framework = analyzer.get_evaluation_framework(roi, stats=stats, dispersed_stats=dispersed)
    return {
        "name": str(name),
        "stats": stats,
        "dispersed_stats": dispersed,
        "evaluation_framework": framework,
        "core_metrics": build_core_metric_snapshot(stats, dispersed, framework),
        "measurements": list(measurements),
        "particle_measurements": list(particles),
        "particle_statistics": analyzer.get_particle_statistics(roi),
        "scale_status": analyzer.get_scale_status(),
        "analysis_roi": None if roi is None else {
            "name": roi.name,
            "x": int(roi.x),
            "y": int(roi.y),
            "width": int(roi.width),
            "height": int(roi.height),
        },
        "preprocess_settings": preprocess.as_kwargs(generate_skeleton=True),
        "detection_settings": detection.as_kwargs(),
    }


def analyze_image_path(path: str, settings: BatchSettings, *, include_visualization: bool = False) -> dict:
    analyzer = CNTAnalyzer()
    analyzer.load_image(path)
    scale_result = analyzer.apply_detected_scale(
        default_micrometers=float(settings.scale_um),
        recognize_text=bool(settings.recognize_scale_text),
    )
    if not scale_result.get("applied") and settings.manual_scale_pixels > 0:
        analyzer.record_manual_scale(
            float(settings.manual_scale_pixels),
            float(settings.scale_um),
            source="batch_manual",
            confidence="high",
        )
    roi = make_center_roi(analyzer, settings.center_fraction, name="中部分析区域")
    result = run_analysis(
        analyzer,
        name=Path(path).name,
        preprocess=settings.preprocess,
        detection=settings.detection,
        roi=roi,
    )
    result["path"] = str(path)
    if include_visualization:
        result["visualization"] = analyzer.get_visualization_with_skeleton(roi)
        result["particle_visualization"] = analyzer.get_particle_visualization(roi)
    return result
