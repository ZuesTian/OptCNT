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
