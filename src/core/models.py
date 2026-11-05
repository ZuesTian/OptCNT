"""
数据模型模块 - 定义所有数据类和数据结构
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ROIRegion:
    """存储ROI区域信息"""
    name: str
    x: int
    y: int
    width: int
    height: int
    color: tuple = (0, 255, 255)  # Default Cyan
    measurements: List['CNTMeasurement'] = field(default_factory=list)
    particle_measurements: List['ParticleMeasurement'] = field(default_factory=list)


@dataclass
class CNTMeasurement:
    """CNT测量结果数据类"""
    id: int
    length_pixels: float
    length_um: float
    contour: np.ndarray
    skeleton: Optional[np.ndarray] = None
    skeleton_bbox: tuple = (0, 0)
    width_mean_um: Optional[float] = None
    width_median_um: Optional[float] = None  # 宽度中位数（鲁棒统计）
    width_iqr_um: Optional[float] = None     # 宽度四分位距（鲁棒统计）
    slenderness: Optional[float] = None
    filter_reason: Optional[str] = None      # 被过滤原因（用于调试）


@dataclass
class ParticleMeasurement:
    """光学图像中的深色紧凑颗粒测量结果。"""
    id: int
    contour: np.ndarray
    centroid: tuple
    area_pixels: float
    equivalent_diameter_pixels: float
    major_axis_pixels: float
    minor_axis_pixels: float
    area_um2: float
    equivalent_diameter_um: float
    major_axis_um: float
    minor_axis_um: float
    circularity: float
    solidity: float
    aspect_ratio: float
    mean_local_contrast: float
    peak_local_contrast: float
    confidence: float
