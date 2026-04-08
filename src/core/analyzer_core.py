"""
图像分析核心模块 - 包含CNTAnalyzer类及其所有图像处理方法
"""
import logging
import os
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 如果 numba 不可用，创建一个空装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .models import ROIRegion, CNTMeasurement
from .utils import (
    SCALE_BAR_BLUE_THRESHOLD, SCALE_BAR_BLUE_SCORE_MIN, SCALE_BAR_MIN_SPAN_PX,
    SCALE_BAR_BGR_DIST_MAX, SCALE_BAR_ROI_X_RATIO, SCALE_BAR_ROI_Y_RATIO,
    SCALE_BAR_ASPECT_RATIO_MIN, SCALE_BAR_ASPECT_RATIO_STRICT,
    SCALE_BAR_OCR_MATCH_THRESHOLD, SCALE_BAR_OCR_EARLY_STOP_SCORE,
    SCALE_BAR_VALUE_RANGE, SCALE_BAR_DEFAULT_UM,
    ANALYSIS_BLACKHAT_KERNEL, CALIBRATED_BLUR_KERNEL,
    CALIBRATED_ADAPTIVE_BLOCK, CALIBRATED_ADAPTIVE_C,
    PREPROCESS_NOISE_LOW_THRESHOLD, PREPROCESS_NOISE_HIGH_THRESHOLD,
    SKELETON_ANGLE_THRESHOLDS,
    SKELETON_WALK_ANGLE_DEG,
    CNT_MERGE_MAX_ANGLE_DIFF_DEG, CNT_MERGE_MAX_ALIGNMENT_DEG,
    LENGTH_DISTRIBUTION_BINS_UM, LENGTH_DISTRIBUTION_LABELS,
    SPATIAL_HOTSPOT_POINT_PERCENTILE, SPATIAL_HOTSPOT_COVERAGE_PERCENTILE,
    SPATIAL_HOTSPOT_SHADOW_PERCENTILE, SPATIAL_HOTSPOT_POINT_WEIGHT,
    SPATIAL_HOTSPOT_COVERAGE_WEIGHT, SPATIAL_HOTSPOT_SHADOW_WEIGHT,
    SPATIAL_HOTSPOT_ACTIVE_SCORE_RANGE_MIN, SPATIAL_HOTSPOT_MASK_PERCENTILE_LARGE,
    SPATIAL_HOTSPOT_MASK_PERCENTILE_SMALL, SPATIAL_HOTSPOT_SEVERE_PERCENTILE_LARGE,
    SPATIAL_HOTSPOT_SEVERE_PERCENTILE_SMALL, SPATIAL_HOTSPOT_OVERLAP_RATIO_THRESHOLD,
    SPATIAL_HOTSPOT_MASK_UPSAMPLE,
)

logger = logging.getLogger(__name__)


class CNTAnalyzer:
    """CNT图像分析核心类"""

    def __init__(self):
        self.original_image = None
        self.analysis_image = None
        self.analysis_gray_image = None
        self.image = None
        self.processed_image = None
        self.binary_image = None
        self.skeleton_image = None
        self.skeleton_overlay = None
        self.scale_um_per_pixel = 0.1
        self.measurements: List[CNTMeasurement] = []
        self.rois: List[ROIRegion] = []
        self.ocr_templates = None
        self.auto_enhance_enabled = True
        self.scale_bar_info: Optional[dict] = None
        self.scale_exclusion_rect: Optional[tuple] = None
        self.scale_exclusion_mask = None
        self.scale_status = {
            'source': 'unset',
            'confidence': 'low',
            'pixels': None,
            'micrometers': None,
            'ocr_micrometers': None,
            'um_per_pixel': float(self.scale_um_per_pixel),
            'exclusion_enabled': False,
        }

    def _ensure_ocr_templates(self):
        """延迟初始化OCR模板"""
        if self.ocr_templates is not None:
            return

        self.ocr_templates = {}
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        thicknesses = [1, 2, 3]
        for d in range(10):
            key = str(d)
            self.ocr_templates[key] = []
            for f in fonts:
                for s in scales:
                    for t in thicknesses:
                        temp = np.zeros((28, 28), dtype=np.uint8)
                        cv2.putText(temp, key, (2, 24), f, s, 255, t, cv2.LINE_AA)
                        self.ocr_templates[key].append(temp)

    def _auto_enhance_image(self, image: np.ndarray) -> np.ndarray:
        """自动增强图像对比度与亮度（CLAHE + 轻微亮度提升）"""
        if image is None or image.size == 0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        # 轻微提升亮度，避免暗图细节丢失
        l_eq = cv2.convertScaleAbs(l_eq, alpha=1.0, beta=6)

        enhanced_lab = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _prepare_analysis_image(self, image: np.ndarray) -> np.ndarray:
        """生成供参数推荐、预处理和检测统一使用的分析图"""
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel_size = max(9, int(ANALYSIS_BLACKHAT_KERNEL))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        corrected = cv2.subtract(enhanced, blackhat)

        self.analysis_gray_image = corrected
        self.analysis_image = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        return self.analysis_image

    def _reset_scale_state(self):
        """重置比例尺检测与应用状态"""
        self.scale_um_per_pixel = 0.1
        self.scale_bar_info = None
        self.scale_exclusion_rect = None
        self.scale_exclusion_mask = None
        self.scale_status = {
            'source': 'unset',
            'confidence': 'low',
            'pixels': None,
            'micrometers': None,
            'ocr_micrometers': None,
            'um_per_pixel': float(self.scale_um_per_pixel),
            'exclusion_enabled': False,
        }

    def _update_scale_status(self,
                             source: str,
                             confidence: str,
                             pixels: Optional[float],
                             micrometers: Optional[float],
                             ocr_micrometers: Optional[float] = None):
        """记录当前比例尺应用状态"""
        self.scale_status = {
            'source': source,
            'confidence': confidence,
            'pixels': None if pixels is None else float(pixels),
            'micrometers': None if micrometers is None else float(micrometers),
            'ocr_micrometers': None if ocr_micrometers is None else float(ocr_micrometers),
            'um_per_pixel': float(self.scale_um_per_pixel),
            'exclusion_enabled': bool(self.scale_exclusion_rect is not None),
        }

    def _set_scale_exclusion_rect(self, rect: Optional[tuple]):
        """更新比例尺排除区域与对应掩码"""
        self.scale_exclusion_rect = rect
        self.scale_exclusion_mask = None

        if rect is None or self.original_image is None:
            return

        x1, y1, x2, y2 = rect
        height, width = self.original_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 255
        self.scale_exclusion_mask = mask

    def _build_scale_exclusion_rect(self,
                                    image_shape: tuple,
                                    bar_x: int,
                                    bar_y: int,
                                    bar_w: int,
                                    bar_h: int,
                                    text_rect: Optional[tuple] = None) -> tuple:
        """根据比例尺条位置构建固定排除区域"""
        height, width = image_shape[:2]
        # 增大padding以完全覆盖比例尺边缘及形态学操作的扩散效应
        pad_x = max(20, int(bar_w * 0.8))
        pad_top = max(40, int(max(bar_h * 12, bar_w * 0.9)))
        # 比例尺文字更常出现在横线下方，因此下方排除区比上方更宽裕
        pad_bottom = max(50, int(max(bar_h * 18, bar_w * 1.2)))

        x1 = max(0, bar_x - pad_x)
        x2 = min(width, bar_x + bar_w + pad_x)
        y1 = max(0, bar_y - pad_top)
        y2 = min(height, bar_y + bar_h + pad_bottom)

        if text_rect is not None:
            tx1, ty1, tx2, ty2 = text_rect
            text_pad_x = max(10, int(max(1, tx2 - tx1) * 0.10))
            text_pad_y = max(8, int(max(1, ty2 - ty1) * 0.18))
            x1 = max(0, min(x1, tx1 - text_pad_x))
            x2 = min(width, max(x2, tx2 + text_pad_x))
            y1 = max(0, min(y1, ty1 - text_pad_y))
            y2 = min(height, max(y2, ty2 + text_pad_y))
        return (int(x1), int(y1), int(x2), int(y2))

    def _find_scale_text_rect(self,
                              image: np.ndarray,
                              bar_x: int,
                              bar_y: int,
                              bar_w: int,
                              bar_h: int) -> Optional[tuple]:
        """查找比例尺附近的文字区域，并返回全图坐标。"""
        h, w = image.shape[:2]
        search_x1 = max(0, bar_x - int(bar_w * 1.8))
        search_x2 = min(w, bar_x + bar_w + int(bar_w * 1.2))
        search_y1 = max(0, bar_y - int(bar_h * 14) - 24)
        search_y2 = min(h, bar_y + int(bar_h * 14) + 28)

        if search_y2 <= search_y1 or search_x2 <= search_x1:
            return None

        search_roi = image[search_y1:search_y2, search_x1:search_x2]
        hsv_t = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        white_mask = (hsv_t[:, :, 1] <= 96) & (hsv_t[:, :, 2] >= 148)
        white_mask = (white_mask.astype(np.uint8) * 255)
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9)),
        )
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_white = None
        best_score = None
        for c in white_contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 40:
                continue
            if cw < max(14, int(bar_w * 0.18)) or ch < max(8, int(bar_h * 1.2)):
                continue

            contour_center_y = search_y1 + y + ch / 2.0
            # 文字可能位于比例尺上方或下方，但通常不会离横线太远
            dist_y = abs(contour_center_y - (bar_y + bar_h / 2.0))
            score = area / (1.0 + dist_y * 0.08)
            if best_score is None or score > best_score:
                best_score = score
                best_white = (x, y, cw, ch)

        if best_white is not None:
            x, y, cw, ch = best_white
            return (
                int(search_x1 + x),
                int(search_y1 + y),
                int(search_x1 + x + cw),
                int(search_y1 + y + ch),
            )

        # 白底文字框未被稳定识别时，退回到更保守的宽区域，优先覆盖比例尺下方文字
        fallback_x1 = max(0, bar_x - int(bar_w * 1.6))
        fallback_x2 = min(w, bar_x + bar_w + int(bar_w * 0.9))
        fallback_y1 = max(0, bar_y - int(bar_h * 8) - 12)
        fallback_y2 = min(h, bar_y + int(bar_h * 16) + 28)
        if fallback_x2 <= fallback_x1 or fallback_y2 <= fallback_y1:
            return None
        return (int(fallback_x1), int(fallback_y1), int(fallback_x2), int(fallback_y2))

    def _set_scale_exclusion_from_line(self,
                                       start: Tuple[float, float],
                                       end: Tuple[float, float]) -> Optional[tuple]:
        """根据手动选中的比例尺线段构建排除区域。"""
        if self.original_image is None:
            return None

        x1, y1 = start
        x2, y2 = end
        min_x = int(round(min(x1, x2)))
        min_y = int(round(min(y1, y2)))
        length = max(1, int(round(np.hypot(x2 - x1, y2 - y1))))
        thickness = max(2, int(round(abs(y2 - y1))) + 2)
        text_rect = self._find_scale_text_rect(self.original_image, min_x, min_y, length, thickness)
        rect = self._build_scale_exclusion_rect(
            self.original_image.shape,
            min_x,
            min_y,
            length,
            thickness,
            text_rect=text_rect,
        )
        self._set_scale_exclusion_rect(rect)
        return rect

    def get_scale_exclusion_mask(self, roi: Optional[ROIRegion] = None) -> Optional[np.ndarray]:
        """返回全图或 ROI 内的比例尺排除掩码"""
        if self.scale_exclusion_mask is None:
            return None
        if roi is None:
            return self.scale_exclusion_mask.copy()
        return self.scale_exclusion_mask[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width].copy()

    def get_scale_status(self) -> dict:
        """返回比例尺状态快照"""
        return dict(self.scale_status)

    def set_image(self, image: np.ndarray) -> np.ndarray:
        """直接设置图像（用于剪贴板粘贴等场景）"""
        if image is None or image.size == 0:
            raise ValueError("无效图像数据")

        self.original_image = image.copy()
        self.image = self._auto_enhance_image(image) if self.auto_enhance_enabled else image.copy()
        self._prepare_analysis_image(self.original_image)
        self._reset_scale_state()
        return self.image

    def _get_analysis_image(self) -> np.ndarray:
        """获取用于检测/测量的分析图像"""
        if self.analysis_image is not None:
            return self.analysis_image
        if self.original_image is not None:
            return self._prepare_analysis_image(self.original_image)
        if self.image is not None:
            return self._prepare_analysis_image(self.image)
        raise ValueError("请先加载图像")

    def _get_analysis_gray_image(self) -> np.ndarray:
        """获取用于检测/测量的灰度分析图"""
        if self.analysis_gray_image is not None:
            return self.analysis_gray_image
        analysis_image = self._get_analysis_image()
        self.analysis_gray_image = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
        return self.analysis_gray_image

    def load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        image = None
        if os.name == 'nt':
            try:
                image_data = np.fromfile(path, dtype=np.uint8)
                if image_data.size > 0:
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            except (OSError, ValueError, cv2.error) as exc:
                logger.warning("Windows path image load fallback failed for %s: %s", path, exc)
                image = None
        if image is None:
            image = cv2.imread(path)
        if image is None:
            raise ValueError(f"无法加载图像: {path}")
        return self.set_image(image)

    def set_scale(self, pixels: float, micrometers: float):
        """设置比例尺"""
        if pixels <= 0 or micrometers <= 0:
            raise ValueError("像素数和微米数必须大于0")
        self.scale_um_per_pixel = micrometers / pixels

    def record_manual_scale(self,
                            pixels: float,
                            micrometers: float,
                            source: str = "manual",
                            confidence: str = "high",
                            selection_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None):
        """记录手动或界面触发的比例尺应用状态"""
        self.set_scale(pixels, micrometers)
        if selection_line is not None:
            self._set_scale_exclusion_from_line(selection_line[0], selection_line[1])
        ocr_hint = None if not self.scale_bar_info else self.scale_bar_info.get('micrometers')
        self._update_scale_status(source, confidence, pixels, micrometers, ocr_hint)

    def apply_detected_scale(self,
                             default_micrometers: float = SCALE_BAR_DEFAULT_UM,
                             recognize_text: bool = True) -> dict:
        """自动检测并按默认物理长度应用比例尺"""
        scale_info = self.detect_scale_bar(recognize_text=recognize_text)
        if scale_info is not None and float(scale_info.get('pixels', 0)) > 0:
            pixels = float(scale_info['pixels'])
            self.set_scale(pixels, default_micrometers)
            self._update_scale_status(
                source='auto_detected',
                confidence='high',
                pixels=pixels,
                micrometers=default_micrometers,
                ocr_micrometers=scale_info.get('micrometers'),
            )
            return {
                'applied': True,
                'scale_info': scale_info,
                'status': self.get_scale_status(),
            }

        self._update_scale_status(
            source='fallback_default',
            confidence='low',
            pixels=None,
            micrometers=default_micrometers,
            ocr_micrometers=None,
        )
        self.scale_bar_info = None
        self._set_scale_exclusion_rect(None)
        return {
            'applied': False,
            'scale_info': scale_info,
            'status': self.get_scale_status(),
        }

    # ===== 比例尺检测方法 =====
    def _detect_scale_bar_blue(self, roi: np.ndarray) -> Optional[tuple]:
        """通过蓝色通道检测比例尺"""
        b, g, r = cv2.split(roi)
        blue_score = b.astype(np.int16) - ((g.astype(np.int16) + r.astype(np.int16)) // 2)
        blue_row_mask = (b > SCALE_BAR_BLUE_THRESHOLD) & (blue_score > SCALE_BAR_BLUE_SCORE_MIN)
        row_sum = blue_row_mask.sum(axis=1)

        if row_sum.size == 0:
            return None

        y = int(row_sum.argmax())
        xs = np.where(blue_row_mask[y])[0]
        if xs.size == 0:
            return None

        spans = []
        start = xs[0]
        prev = xs[0]
        for x in xs[1:]:
            if x - prev > 2:
                spans.append((start, prev))
                start = x
            prev = x
        spans.append((start, prev))
        spans = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)
        span = spans[0]
        length = int(span[1] - span[0] + 1)

        if length < SCALE_BAR_MIN_SPAN_PX:
            return None

        return (int(span[0]), int(y), int(length), 2)

    def _detect_scale_bar_mask(self, roi: np.ndarray, gray_roi: np.ndarray) -> Optional[tuple]:
        """通过蓝色掩码检测比例尺"""
        b, g, r = cv2.split(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_mask = (b > 80) & ((b.astype(np.int16) - g.astype(np.int16)) > 15) & ((b.astype(np.int16) - r.astype(np.int16)) > 15)
        blue_mask_hsv = (hsv[:, :, 0] >= 70) & (hsv[:, :, 0] <= 170) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 30)
        bgr_dist = (b.astype(np.int16) - 255) ** 2 + (g.astype(np.int16)) ** 2 + (r.astype(np.int16)) ** 2
        blue_mask_dist = bgr_dist < SCALE_BAR_BGR_DIST_MAX
        mask = ((blue_mask | blue_mask_hsv | blue_mask_dist).astype(np.uint8) * 255)

        kernel_w = max(15, roi.shape[1] // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_w = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 30 or ch < 2:
                continue
            if cw / max(1, ch) < SCALE_BAR_ASPECT_RATIO_MIN:
                continue
            if cw > best_w:
                best_w = cw
                best = (x, y, cw, ch)

        return best

    def _detect_scale_bar_gray(self, roi: np.ndarray, gray_roi: np.ndarray) -> Optional[tuple]:
        """通过灰度阈值检测比例尺"""
        _, bin1 = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float((bin1 > 0).mean()) > 0.5:
            bin1 = cv2.bitwise_not(bin1)

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, roi.shape[1] // 8), 1))
        horizontal = cv2.morphologyEx(bin1, cv2.MORPH_OPEN, hor_kernel)
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_w = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < max(60, roi.shape[1] // 8) or ch < 1:
                continue
            if cw / max(1, ch) < SCALE_BAR_ASPECT_RATIO_STRICT:
                continue
            if cw > best_w:
                best_w = cw
                best = (x, y, cw, ch)

        return best

    def _detect_scale_bar_hough(self, roi: np.ndarray, gray_roi: np.ndarray) -> Optional[tuple]:
        """通过Hough线检测比例尺"""
        edges = cv2.Canny(gray_roi, 40, 140)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=max(40, roi.shape[1] // 8),
                                maxLineGap=15)
        if lines is None:
            return None

        best_len = 0
        best = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) > 4:
                continue
            length = abs(x2 - x1)
            if length < best_len:
                continue
            if max(y1, y2) < int(roi.shape[0] * 0.4):
                continue
            best_len = length
            x = min(x1, x2)
            y = min(y1, y2)
            best = (x, y, length, 2)

        return best

    def _extract_text_roi(self, image: np.ndarray, bar_x: int, bar_y: int, bar_w: int, bar_h: int) -> Optional[np.ndarray]:
        """提取比例尺文字区域"""
        text_rect = self._find_scale_text_rect(image, bar_x, bar_y, bar_w, bar_h)
        if text_rect is None:
            return None
        x1, y1, x2, y2 = text_rect
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def _preprocess_ocr_image(self, text_roi: np.ndarray) -> np.ndarray:
        """OCR图像预处理"""
        gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float((binary > 0).mean()) > 0.5:
            binary = cv2.bitwise_not(binary)

        kernel_t = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_t)

        mask = binary > 0
        row_sum = mask.sum(axis=1)
        row_thresh = max(1, int(mask.shape[1] * 0.05))
        rows = np.where(row_sum > row_thresh)[0]
        if rows.size > 0:
            binary = binary[rows[0]:rows[-1] + 1, :]
        return binary

    def _segment_characters(self, binary: np.ndarray) -> List[tuple]:
        """字符分割"""
        mask = binary > 0
        col_sum = mask.sum(axis=0)
        col_thresh = max(1, int(mask.shape[0] * 0.1))
        cols = np.where(col_sum > col_thresh)[0]
        boxes = []
        
        if cols.size > 0:
            start = cols[0]
            prev = cols[0]
            for c in cols[1:]:
                if c - prev > 2:
                    boxes.append((start, 0, prev - start + 1, binary.shape[0]))
                    start = c
                prev = c
            boxes.append((start, 0, prev - start + 1, binary.shape[0]))
        else:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch
                if ch < 6 or cw < 2 or area < 12:
                    continue
                boxes.append((x, y, cw, ch))
        return sorted(boxes, key=lambda b: b[0])

    def _recognize_characters(self, binary: np.ndarray, boxes: List[tuple]) -> str:
        """识别字符"""
        if not boxes:
            return ""
            
        heights = [b[3] for b in boxes]
        median_h = float(np.median(heights))
        
        self._ensure_ocr_templates()
        
        def normalize(img):
            ih, iw = img.shape[:2]
            size = max(ih, iw)
            canvas = np.zeros((size, size), dtype=np.uint8)
            oy = (size - ih) // 2
            ox = (size - iw) // 2
            canvas[oy:oy + ih, ox:ox + iw] = img
            return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

        tokens = []
        for x, y, cw, ch in boxes:
            if ch < median_h * 0.5 and cw < median_h * 0.5:
                tokens.append(".")
                continue
            crop = binary[y:y + ch, x:x + cw]
            norm = normalize(crop)
            best_char = None
            best_score = None
            early_stop = False
            for k, temps in self.ocr_templates.items():
                for temp in temps:
                    score = float(cv2.matchTemplate(norm, temp, cv2.TM_CCOEFF_NORMED)[0][0])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_char = k
                        if score >= SCALE_BAR_OCR_EARLY_STOP_SCORE:
                            early_stop = True
                            break
                if early_stop:
                    break
            if best_char is not None and best_score is not None and best_score >= SCALE_BAR_OCR_MATCH_THRESHOLD:
                tokens.append(best_char)
        
        return "".join(tokens)

    def _parse_scale_value(self, text: str) -> Optional[float]:
        """解析数值"""
        digits = []
        dot_used = False
        for ch in text:
            if ch.isdigit():
                digits.append(ch)
            elif ch == "." and not dot_used:
                digits.append(ch)
                dot_used = True

        try:
            if any(c.isdigit() for c in digits):
                value_text = "".join(digits)
                if 1 <= len(value_text.replace(".", "")) <= 4:
                    value = float(value_text)
                    if SCALE_BAR_VALUE_RANGE[0] <= value <= SCALE_BAR_VALUE_RANGE[1]:
                        return value
        except ValueError as exc:
            logger.debug("Failed to parse OCR scale value from %r: %s", text, exc)
        return None

    def _recognize_scale_value(self, text_roi: np.ndarray) -> Optional[float]:
        """通过OCR识别比例尺数值"""
        binary = self._preprocess_ocr_image(text_roi)
        boxes = self._segment_characters(binary)
        if not boxes:
            return None
            
        text = self._recognize_characters(binary, boxes)
        if not text:
            return None
            
        return self._parse_scale_value(text)

    def detect_scale_bar(self, recognize_text: bool = True) -> Optional[dict]:
        """检测图像中的比例尺"""
        image = self.original_image if self.original_image is not None else self._get_analysis_image()
        h, w = image.shape[:2]
        x0 = int(w * SCALE_BAR_ROI_X_RATIO)
        y0 = int(h * SCALE_BAR_ROI_Y_RATIO)
        roi = image[y0:h, x0:w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best = self._detect_scale_bar_blue(roi)
        if best is None:
            best = self._detect_scale_bar_mask(roi, gray_roi)
        if best is None:
            best = self._detect_scale_bar_gray(roi, gray_roi)
        if best is None:
            best = self._detect_scale_bar_hough(roi, gray_roi)
        if best is None:
            self.scale_bar_info = None
            self._set_scale_exclusion_rect(None)
            return None

        bar_x, bar_y, bar_w, bar_h = best
        bar_x += x0
        bar_y += y0

        text_rect = self._find_scale_text_rect(image, bar_x, bar_y, bar_w, bar_h)
        text_roi = None if text_rect is None else image[text_rect[1]:text_rect[3], text_rect[0]:text_rect[2]]
        micrometers = (
            self._recognize_scale_value(text_roi)
            if recognize_text and text_roi is not None else
            None
        )
        exclusion_rect = self._build_scale_exclusion_rect(
            image.shape,
            bar_x,
            bar_y,
            bar_w,
            bar_h,
            text_rect=text_rect,
        )

        self.scale_bar_info = {
            "pixels": float(bar_w),
            "micrometers": micrometers,
            "bar_rect": (int(bar_x), int(bar_y), int(bar_w), int(bar_h)),
            "text_rect": text_rect,
            "exclusion_rect": exclusion_rect,
        }
        self._set_scale_exclusion_rect(exclusion_rect)
        return dict(self.scale_bar_info)

    # ===== ROI管理方法 =====
    def add_roi(self, roi_or_name, x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> ROIRegion:
        """添加ROI区域，支持传入ROIRegion对象或分散参数"""
        if isinstance(roi_or_name, ROIRegion):
            roi = roi_or_name
        else:
            roi = ROIRegion(name=roi_or_name, x=x, y=y, width=width, height=height)
        self.rois.append(roi)
        return roi

    def remove_roi(self, identifier):
        """移除ROI区域，支持按索引(int)或名称(str)删除"""
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.rois):
                self.rois.pop(identifier)
        else:
            self.rois = [roi for roi in self.rois if roi.name != identifier]

    def clear_rois(self):
        """清空所有ROI"""
        self.rois = []

    def clear_measurements(self):
        """清空所有测量结果"""
        self.measurements = []

    def _get_analysis_region(self, roi: Optional[ROIRegion] = None) -> tuple:
        """获取分析区域"""
        analysis_image = self._get_analysis_image()
        if roi:
            x, y, w, h = roi.x, roi.y, roi.width, roi.height
            return (y, y + h, x, x + w)
        else:
            h, w = analysis_image.shape[:2]
            return (0, h, 0, w)

    def _get_valid_analysis_mask(self, roi: Optional[ROIRegion] = None) -> np.ndarray:
        """返回剔除比例尺区域后的有效分析掩码"""
        y1, y2, x1, x2 = self._get_analysis_region(roi)
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        mask = np.ones((height, width), dtype=bool)
        exclusion_mask = self.get_scale_exclusion_mask(roi)
        if exclusion_mask is not None and exclusion_mask.shape == mask.shape:
            mask &= exclusion_mask == 0
        return mask

    # ===== 自适应参数推荐 =====
    def _normalize_odd(self, value: int, min_value: int, max_value: int) -> int:
        """Clamp a kernel value into range and keep it odd."""
        value = int(max(min_value, min(max_value, value)))
        if value % 2 == 0:
            if value >= max_value:
                value -= 1
            else:
                value += 1
        return int(max(min_value, min(max_value, value)))

    def _suggest_adaptive_binary(self, gray: np.ndarray, blur_kernel: int,
                                 adaptive_block: int, adaptive_c: int) -> np.ndarray:
        """Build a temporary binary image for auto-parameter recommendation."""
        blur_kernel = self._normalize_odd(blur_kernel, 7, 15)
        adaptive_block = self._normalize_odd(adaptive_block, 9, 25)
        adaptive_c = int(max(1, min(7, adaptive_c)))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
        )

    def _masked_foreground_ratio(self, binary: np.ndarray, valid_mask: np.ndarray) -> float:
        """Measure foreground ratio only inside the valid analysis area."""
        valid_area = max(1, int(np.count_nonzero(valid_mask)))
        return float(np.count_nonzero((binary > 0) & valid_mask) / valid_area)

    def _get_fg_target_by_profile(self, profile: str) -> Tuple[float, float]:
        """Return target foreground ratio and tolerance for each detection profile."""
        if profile == "precision":
            return 0.060, 0.015
        if profile == "recall":
            return 0.105, 0.025
        return 0.080, 0.020

    def suggest_preprocess_params(self, roi: Optional[ROIRegion] = None, detection_profile: str = "balanced") -> dict:
        """根据校准基线、噪点、边缘密度和前景占比推荐预处理参数。"""
        y1, y2, x1, x2 = self._get_analysis_region(roi)
        gray = self._get_analysis_gray_image()[y1:y2, x1:x2]
        valid_mask = self._get_valid_analysis_mask(roi)

        baseline = {
            "blur_kernel": int(CALIBRATED_BLUR_KERNEL),
            "adaptive_block": int(CALIBRATED_ADAPTIVE_BLOCK),
            "adaptive_c": int(CALIBRATED_ADAPTIVE_C),
        }
        if gray.size == 0 or not np.any(valid_mask):
            return {
                **baseline,
                "reason_summary": (
                    f"使用基线参数 {baseline['blur_kernel']}/{baseline['adaptive_block']}/{baseline['adaptive_c']}；"
                    "当前没有可分析区域。"
                ),
                "reasons": ["当前没有有效分析区域，保持基线参数。"],
                "metrics": {},
            }

        profile = (detection_profile or "balanced").lower()
        blur_kernel = int(CALIBRATED_BLUR_KERNEL)
        adaptive_block = int(CALIBRATED_ADAPTIVE_BLOCK)
        adaptive_c = int(CALIBRATED_ADAPTIVE_C)
        reasons: List[str] = [f"基线参数为 {blur_kernel}/{adaptive_block}/{adaptive_c}。"]

        if profile == "precision":
            blur_kernel += 2
            adaptive_block += 2
            adaptive_c += 1
            reasons.append("当前风格偏向少误检，先采用更保守的平滑和阈值。")
        elif profile == "recall":
            blur_kernel -= 2
            adaptive_block -= 2
            adaptive_c -= 1
            reasons.append("当前风格偏向少漏检，先采用更敏感的平滑和阈值。")

        valid_pixels = gray[valid_mask]
        p10, p50, p90 = np.percentile(valid_pixels, [10, 50, 90])
        contrast_span = max(1.0, float(p90 - p10))

        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        noise_score = float(np.sqrt(max(float(np.var(laplacian[valid_mask])), 0.0)) / contrast_span)
        if noise_score >= PREPROCESS_NOISE_HIGH_THRESHOLD:
            blur_kernel += 2
            reasons.append(f"高频噪点偏多（噪点指标 {noise_score:.2f}），模糊核加大 2。")
        elif noise_score <= PREPROCESS_NOISE_LOW_THRESHOLD:
            blur_kernel -= 2
            reasons.append(f"高频噪点较低（噪点指标 {noise_score:.2f}），模糊核减小 2 以保留细节。")
        else:
            reasons.append(f"噪点指标 {noise_score:.2f} 处于中等区间，模糊核保持基线附近。")

        edge_input = cv2.GaussianBlur(gray, (3, 3), 0)
        sobel_x = cv2.Sobel(edge_input, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(edge_input, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(sobel_x, sobel_y)
        edge_density = float(np.mean(gradient_mag[valid_mask]) / contrast_span)
        if edge_density >= 1.10:
            adaptive_block -= 2
            reasons.append(f"边缘密度指标偏高（{edge_density:.3f}），块大小减小 2 以增强局部分辨率。")
        elif edge_density <= 0.92:
            adaptive_block += 2
            reasons.append(f"边缘密度指标偏低（{edge_density:.3f}），块大小增大 2 以提升稳定性。")
        else:
            reasons.append(f"边缘密度指标 {edge_density:.3f} 处于中等区间，块大小保持基线附近。")

        blur_kernel = self._normalize_odd(blur_kernel, 7, 15)
        adaptive_block = self._normalize_odd(adaptive_block, 9, 25)
        adaptive_c = int(max(1, min(7, adaptive_c)))

        target_fg_ratio, fg_tolerance = self._get_fg_target_by_profile(profile)
        best_c = adaptive_c
        best_fg_ratio = -1.0
        best_delta = float("inf")
        final_fg_ratio = -1.0
        c_iterations = 0

        for _ in range(5):
            c_iterations += 1
            binary = self._suggest_adaptive_binary(gray, blur_kernel, adaptive_block, adaptive_c)
            fg_ratio = self._masked_foreground_ratio(binary, valid_mask)
            delta = abs(fg_ratio - target_fg_ratio)

            if delta < best_delta:
                best_delta = delta
                best_c = adaptive_c
                best_fg_ratio = fg_ratio

            if delta <= fg_tolerance:
                final_fg_ratio = fg_ratio
                break

            if fg_ratio > target_fg_ratio:
                next_c = min(adaptive_c + 1, 7)
            else:
                next_c = max(adaptive_c - 1, 1)

            if next_c == adaptive_c:
                final_fg_ratio = fg_ratio
                break
            adaptive_c = next_c

        if final_fg_ratio < 0:
            adaptive_c = best_c
            final_fg_ratio = best_fg_ratio
            reasons.append(
                f"前景占比未完全进入目标区间（目标 {target_fg_ratio:.3f}±{fg_tolerance:.3f}），"
                f"已选取最接近目标的 C={adaptive_c}。"
            )
        elif c_iterations == 1:
            reasons.append(f"当前前景占比 {final_fg_ratio:.3f} 已接近目标区间，C 保持为 {adaptive_c}。")
        else:
            reasons.append(
                f"经过 {c_iterations} 次迭代后，前景占比调整到 {final_fg_ratio:.3f}，"
                f"C 收敛为 {adaptive_c}。"
            )

        metrics = {
            "noise_score": float(noise_score),
            "edge_density": float(edge_density),
            "target_fg_ratio": float(target_fg_ratio),
            "fg_tolerance": float(fg_tolerance),
            "foreground_ratio": float(final_fg_ratio),
            "contrast_span": float(contrast_span),
            "median_intensity": float(p50),
        }
        reason_summary = (
            f"推荐 {blur_kernel}/{adaptive_block}/{adaptive_c}；"
            f"噪点={noise_score:.2f}，边缘密度={edge_density:.3f}，前景占比={final_fg_ratio:.3f}。"
        )
        return {
            "blur_kernel": int(blur_kernel),
            "adaptive_block": int(adaptive_block),
            "adaptive_c": int(adaptive_c),
            "reason_summary": reason_summary,
            "reasons": reasons,
            "metrics": metrics,
        }

    # ===== 预处理方法 =====
    def preprocess(self, blur_kernel: int = 5,
                   adaptive_block: int = 11,
                   adaptive_c: int = 2,
                   bridge_strength: int = 0,
                   threshold_invert: bool = True,
                   roi: Optional[ROIRegion] = None,
                   generate_skeleton: bool = True) -> np.ndarray:
        """图像预处理"""
        y1, y2, x1, x2 = self._get_analysis_region(roi)
        gray = self._get_analysis_gray_image()[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        threshold_type = cv2.THRESH_BINARY_INV if threshold_invert else cv2.THRESH_BINARY
        self.binary_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type, adaptive_block, adaptive_c
        )

        kernel = np.ones((3, 3), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
        
        # 在形态学操作之前先清除比例尺区域，避免边缘扩散
        exclusion_mask = self.get_scale_exclusion_mask(roi)
        if exclusion_mask is not None and exclusion_mask.shape == self.binary_image.shape:
            self.binary_image[exclusion_mask > 0] = 0
        
        if bridge_strength > 0:
            bridge_size = max(3, min(2 * int(bridge_strength) + 1, 21))
            if bridge_size % 2 == 0:
                bridge_size += 1
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_size, bridge_size))
            self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, bridge_kernel)
            
            # 桥接操作后再次清除比例尺区域，防止桥接扩散到排除区
            if exclusion_mask is not None and exclusion_mask.shape == self.binary_image.shape:
                self.binary_image[exclusion_mask > 0] = 0

        if generate_skeleton:
            self.skeleton_image = self._skeletonize(self.binary_image.copy())
            self._generate_skeleton_overlay(y1, y2, x1, x2)
        else:
            self.skeleton_image = None
            self.skeleton_overlay = None

        self.processed_image = self.binary_image.copy()
        return self.binary_image

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """生成骨架"""
        work = np.where(binary > 0, 255, 0).astype(np.uint8)
        if not np.any(work):
            return work

        thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
        if thinning is not None:
            return thinning(work)

        skeleton = np.zeros_like(work)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(work, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(work, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            work = eroded
            if cv2.countNonZero(work) == 0:
                break

        return skeleton

    def _generate_skeleton_overlay(self, y1: int, y2: int, x1: int, x2: int):
        """生成骨架叠加到原图"""
        if self.image is None or self.skeleton_image is None:
            return
        self.skeleton_overlay = self.image.copy()
        skeleton_mask = self.skeleton_image > 0
        self.skeleton_overlay[y1:y2, x1:x2][skeleton_mask] = [0, 0, 255]
        self._draw_scale_exclusion_annotation(self.skeleton_overlay)

    def _draw_scale_exclusion_annotation(self, image: np.ndarray):
        """在可视化图上绘制比例尺排除区域"""
        if image is None or self.scale_exclusion_rect is None:
            return

        x1, y1, x2, y2 = self.scale_exclusion_rect
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (16, 140, 255), -1)
        cv2.addWeighted(overlay, 0.14, image, 0.86, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (16, 140, 255), 2)
        label_y = max(18, y1 - 6)
        cv2.putText(
            image,
            "Scale Exclusion",
            (x1 + 4, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (16, 140, 255),
            1,
            cv2.LINE_AA,
        )

    # ===== 骨架处理方法 =====
    def _build_skeleton_neighbors(self, skeleton: np.ndarray) -> dict:
        """构建骨架邻接表"""
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        if len(skeleton_points) < 2:
            return {}
        point_set = {(int(y), int(x)) for y, x in skeleton_points}
        neighbors = {}
        for y, x in point_set:
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    candidate = (y + dy, x + dx)
                    if candidate in point_set:
                        neigh.append(candidate)
            neighbors[(y, x)] = neigh
        return neighbors

    def _count_endpoints(self, skeleton: np.ndarray, neighbors: Optional[dict] = None) -> int:
        """计算骨架端点数量"""
        if neighbors is None:
            neighbors = self._build_skeleton_neighbors(skeleton)
        if not neighbors:
            return 0
        return sum(1 for neigh in neighbors.values() if len(neigh) == 1)

    @staticmethod
    def _calculate_path_length(path: List[Tuple[int, int]]) -> float:
        """计算骨架路径长度"""
        if len(path) < 2:
            return 0.0

        # 使用优化版本（如果可用）
        if NUMBA_AVAILABLE:
            path_array = np.array(path, dtype=np.float64)
            return _calculate_path_length_fast(path_array)
        
        # 回退到原始实现
        length = 0.0
        for index in range(1, len(path)):
            prev_y, prev_x = path[index - 1]
            curr_y, curr_x = path[index]
            length += float(np.hypot(curr_y - prev_y, curr_x - prev_x))
        return length

    @staticmethod
    def _build_path_neighbors(path: List[Tuple[int, int]]) -> dict:
        """Build a lightweight adjacency map for an already-pruned primary path."""
        if len(path) < 2:
            return {}

        neighbors = {}
        for index, point in enumerate(path):
            point_neighbors = []
            if index > 0:
                point_neighbors.append(path[index - 1])
            if index + 1 < len(path):
                point_neighbors.append(path[index + 1])
            neighbors[point] = point_neighbors
        return neighbors

    @staticmethod
    def _is_simple_skeleton_graph(neighbors: dict) -> bool:
        """Whether the skeleton graph is already a single unbranched chain."""
        return bool(neighbors) and all(len(point_neighbors) <= 2 for point_neighbors in neighbors.values())

    @staticmethod
    def _continuation_cosine_threshold(angle_deg: float) -> float:
        """将旧的直线阈值角度转换为前向延续所需的余弦阈值"""
        clamped = max(0.0, min(180.0, float(angle_deg)))
        max_turn_deg = max(0.0, 180.0 - clamped)
        return float(np.cos(np.deg2rad(max_turn_deg)))

    def _choose_skeleton_continuation(self,
                                      prev_pt: Optional[Tuple[int, int]],
                                      curr_pt: Tuple[int, int],
                                      candidates: List[Tuple[int, int]],
                                      degree: int,
                                      angle_deg: float) -> Optional[Tuple[int, int]]:
        """在分支点处选择更符合前向延续的骨架下一步"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if prev_pt is None or degree < 3:
            return candidates[0]

        in_vec = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]], dtype=float)
        in_norm = float(np.linalg.norm(in_vec))
        if in_norm <= 0:
            return candidates[0]

        best = None
        best_cos = -2.0
        min_cos = self._continuation_cosine_threshold(angle_deg)
        for cand in candidates:
            out_vec = np.array([cand[0] - curr_pt[0], cand[1] - curr_pt[1]], dtype=float)
            out_norm = float(np.linalg.norm(out_vec))
            if out_norm <= 0:
                continue
            cos_val = float(np.dot(in_vec, out_vec) / (in_norm * out_norm))
            if cos_val > best_cos:
                best_cos = cos_val
                best = cand

        if best is None or best_cos < min_cos:
            return None
        return best

    def _trace_skeleton_path(self,
                             start_pt: Tuple[int, int],
                             neighbors: dict,
                             angle_deg: float) -> List[Tuple[int, int]]:
        """沿骨架从端点追踪单条主路径"""
        path = [start_pt]
        prev_pt = None
        curr_pt = start_pt

        while True:
            neigh = neighbors.get(curr_pt, [])
            candidates = [point for point in neigh if point != prev_pt]
            if not candidates:
                break

            next_pt = self._choose_skeleton_continuation(
                prev_pt,
                curr_pt,
                candidates,
                len(neigh),
                angle_deg,
            )
            if next_pt is None:
                break

            path.append(next_pt)
            prev_pt, curr_pt = curr_pt, next_pt

        return path

    def _find_best_skeleton_path(self,
                                 neighbors: dict,
                                 angle_candidates: List[float]) -> List[Tuple[int, int]]:
        """在多个角度阈值下寻找最长的骨架主路径"""
        endpoints = [point for point, neigh in neighbors.items() if len(neigh) == 1]
        if len(endpoints) < 2:
            return []

        if self._is_simple_skeleton_graph(neighbors):
            return self._trace_skeleton_path(endpoints[0], neighbors, float(SKELETON_WALK_ANGLE_DEG))

        best_path: List[Tuple[int, int]] = []
        best_length = -1.0
        for angle_deg in angle_candidates:
            for endpoint in endpoints:
                path = self._trace_skeleton_path(endpoint, neighbors, angle_deg)
                path_length = self._calculate_path_length(path)
                if path_length > best_length:
                    best_length = path_length
                    best_path = path
        return best_path

    def _extract_primary_path(self, skeleton: np.ndarray, neighbors: Optional[dict] = None) -> np.ndarray:
        """提取骨架主路径"""
        if neighbors is None:
            neighbors = self._build_skeleton_neighbors(skeleton)
        if len(neighbors) < 2:
            return skeleton

        best_path = self._find_best_skeleton_path(neighbors, list(SKELETON_ANGLE_THRESHOLDS))
        if len(best_path) < 2:
            return skeleton

        pruned = np.zeros_like(skeleton, dtype=np.uint8)
        for y, x in best_path:
            pruned[y, x] = 255
        return pruned

    def _calculate_skeleton_length(self, skeleton: np.ndarray, neighbors: Optional[dict] = None) -> float:
        """计算骨架长度"""
        if neighbors is None:
            neighbors = self._build_skeleton_neighbors(skeleton)
        if len(neighbors) < 2:
            return 0.0

        endpoints = [point for point, neigh in neighbors.items() if len(neigh) == 1]
        if len(endpoints) >= 2 and self._is_simple_skeleton_graph(neighbors):
            return self._calculate_path_length(
                self._trace_skeleton_path(endpoints[0], neighbors, float(SKELETON_WALK_ANGLE_DEG))
            )

        best_path = self._find_best_skeleton_path(neighbors, [float(SKELETON_WALK_ANGLE_DEG)])
        if len(best_path) >= 2:
            return self._calculate_path_length(best_path)

        unique_edges = set()
        total_length = 0.0
        for point, point_neighbors in neighbors.items():
            for neigh in point_neighbors:
                edge = (point, neigh) if point <= neigh else (neigh, point)
                if edge in unique_edges:
                    continue
                unique_edges.add(edge)
                total_length += float(np.hypot(neigh[0] - point[0], neigh[1] - point[1]))
        return total_length

    # ===== 宽度测量方法 =====
    def _measure_width(self, skeleton: np.ndarray, cnt_binary: np.ndarray) -> dict:
        """通过骨架法测量CNT宽度（鲁棒统计）

        对骨架上的每个点，计算其到最近轮廓边界的距离（即半宽），
        返回均值、中位数和IQR，使用中位数作为主要指标以抵抗异常值。

        Args:
            skeleton: 骨架二值图
            cnt_binary: CNT区域二值图

        Returns:
            dict: 包含 mean, median, iqr 的宽度统计（像素），全部为直径
        """
        result = {'mean': 0.0, 'median': 0.0, 'iqr': 0.0}
        if skeleton is None or cnt_binary is None:
            return result

        dist_transform = cv2.distanceTransform(cnt_binary, cv2.DIST_L2, 5)

        skeleton_mask = skeleton > 0
        if not skeleton_mask.any():
            return result

        half_widths = dist_transform[skeleton_mask]
        if len(half_widths) == 0:
            return result

        result['mean'] = float(np.mean(half_widths)) * 2.0
        q25, median, q75 = np.percentile(half_widths, [25, 50, 75])
        result['median'] = float(median) * 2.0
        result['iqr'] = (q75 - q25) * 2.0
        return result

    def _split_stuck_cnt_region(self, cnt_binary: np.ndarray, split_mode: str = "conservative") -> List[np.ndarray]:
        """对疑似粘连CNT连通域进行分离（距离变换 + 分水岭）"""
        if cnt_binary is None or cnt_binary.size == 0:
            return []

        region = np.where(cnt_binary > 0, 255, 0).astype(np.uint8)
        area = int(cv2.countNonZero(region))
        if area <= 0:
            return []

        mode = (split_mode or "conservative").lower()
        if mode in ("off", "none", "close", "0"):
            return [region]

        # 保守策略：仅对“面积较大且骨架端点异常”的连通域尝试分离，降低过分割风险
        skeleton = self._skeletonize(region)
        neighbors = self._build_skeleton_neighbors(skeleton)
        endpoints = self._count_endpoints(skeleton, neighbors) if neighbors else 0

        if mode == "aggressive":
            area_threshold = 180
            peak_ratio = 0.35
            min_piece_ratio = 0.015
            bypass_endpoint_check = True
        else:
            area_threshold = 300
            peak_ratio = 0.42
            min_piece_ratio = 0.02
            bypass_endpoint_check = False

        if area < area_threshold:
            return [region]
        if not bypass_endpoint_check and endpoints == 2:
            return [region]

        dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist <= 1.0:
            return [region]

        sure_fg = cv2.threshold(dist, peak_ratio * max_dist, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)

        n_fg, markers = cv2.connectedComponents(sure_fg)
        if n_fg <= 2:
            return [region]

        sure_bg = cv2.dilate(region, kernel, iterations=1)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers = markers + 1
        markers[unknown > 0] = 0

        ws_input = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)
        ws_markers = cv2.watershed(ws_input, markers.astype(np.int32))

        min_piece_area = max(30, int(area * min_piece_ratio))
        pieces: List[np.ndarray] = []
        for label in np.unique(ws_markers):
            if label <= 1:
                continue
            piece = np.zeros_like(region, dtype=np.uint8)
            piece[ws_markers == label] = 255
            piece = cv2.bitwise_and(piece, region)
            if cv2.countNonZero(piece) >= min_piece_area:
                pieces.append(piece)

        return pieces if len(pieces) >= 2 else [region]

    def _get_endpoint_limit(self, profile: str) -> int:
        """根据识别风格返回骨架端点上限"""
        mode = (profile or "balanced").lower()
        if mode == "precision":
            return 2
        if mode == "recall":
            return 4
        return 3

    def _build_cnt_candidate(self,
                             cnt_binary: np.ndarray,
                             x_offset: int,
                             y_offset: int,
                             profile: str,
                             bypass_endpoint_filter: bool = False) -> Optional[dict]:
        """从局部二值区域提取候选 CNT"""
        sub_contours, _ = cv2.findContours(cnt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not sub_contours:
            return None

        sub_contour = max(sub_contours, key=cv2.contourArea)
        area = float(cv2.contourArea(sub_contour))
        if area < 1.0:
            return None

        skeleton_region = self._skeletonize(cnt_binary)
        neighbors = self._build_skeleton_neighbors(skeleton_region)
        if len(neighbors) < 2:
            return None

        best_path = self._find_best_skeleton_path(neighbors, list(SKELETON_ANGLE_THRESHOLDS))
        if len(best_path) < 2:
            return None
        length_px = float(self._calculate_path_length(best_path))
        if length_px < 1.0:
            return None

        skeleton_region = np.zeros_like(skeleton_region, dtype=np.uint8)
        for y, x in best_path:
            skeleton_region[y, x] = 255
        neighbors = self._build_path_neighbors(best_path)
        if len(neighbors) < 2:
            return None

        endpoint_count = self._count_endpoints(skeleton_region, neighbors)
        if not bypass_endpoint_filter:
            endpoint_limit = self._get_endpoint_limit(profile)
            if endpoint_count <= 0 or endpoint_count > endpoint_limit:
                return None

        width_stats = self._measure_width(skeleton_region, cnt_binary)
        width_mean_px = float(width_stats['mean'])
        width_median_px = float(width_stats['median'])
        width_iqr_px = float(width_stats['iqr'])
        slenderness_val = (length_px / width_median_px) if width_median_px > 0 else None

        contour_local = (sub_contour + np.array([x_offset, y_offset])).astype(np.int32)
        return {
            'contour_local': contour_local,
            'skeleton': skeleton_region,
            'skeleton_bbox_local': (int(x_offset), int(y_offset)),
            'length_pixels': length_px,
            'area': area,
            'width_mean_px': width_mean_px,
            'width_median_px': width_median_px,
            'width_iqr_px': width_iqr_px,
            'slenderness': slenderness_val,
            'endpoint_count': int(endpoint_count),
        }

    @staticmethod
    def _angle_difference_deg(angle_a: float, angle_b: float) -> float:
        """计算主方向下的最小夹角"""
        diff = abs(float(angle_a) - float(angle_b)) % 180.0
        return 180.0 - diff if diff > 90.0 else diff

    def _candidate_orientation_deg(self, candidate: dict) -> Optional[float]:
        """估计候选 CNT 的主方向"""
        contour = candidate.get('contour_local')
        if contour is None:
            return None

        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 2:
            return None

        vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx_scalar = float(np.asarray(vx).reshape(-1)[0])
        vy_scalar = float(np.asarray(vy).reshape(-1)[0])
        angle = float(np.degrees(np.arctan2(vy_scalar, vx_scalar)))
        if angle < 0:
            angle += 180.0
        return angle

    def _candidate_endpoints_xy(self, candidate: dict) -> List[np.ndarray]:
        """获取候选 CNT 端点在分析图坐标中的位置"""
        skeleton = candidate.get('skeleton')
        if skeleton is None or getattr(skeleton, 'size', 0) == 0:
            return []

        neighbors = self._build_skeleton_neighbors(skeleton)
        endpoints = [point for point, neigh in neighbors.items() if len(neigh) == 1]
        if not endpoints:
            return []

        x_offset, y_offset = candidate.get('skeleton_bbox_local', (0, 0))
        output = []
        for y, x in endpoints:
            output.append(np.array([float(x + x_offset), float(y + y_offset)], dtype=float))
        return output

    def _candidate_pair_metrics(self, first: dict, second: dict) -> dict:
        """计算两段候选 CNT 的空间关系"""
        first_endpoints = self._candidate_endpoints_xy(first)
        second_endpoints = self._candidate_endpoints_xy(second)
        if not first_endpoints or not second_endpoints:
            return {'distance': float('inf')}

        first_angle = self._candidate_orientation_deg(first)
        second_angle = self._candidate_orientation_deg(second)
        if first_angle is None or second_angle is None:
            return {'distance': float('inf')}

        best_distance = float('inf')
        best_pair = None
        for point_a in first_endpoints:
            for point_b in second_endpoints:
                distance = float(np.linalg.norm(point_a - point_b))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (point_a, point_b)

        if best_pair is None:
            return {'distance': float('inf')}

        point_a, point_b = best_pair
        gap_vector = point_b - point_a
        if np.linalg.norm(gap_vector) < 1e-6:
            connection_angle = first_angle
        else:
            connection_angle = float(np.degrees(np.arctan2(gap_vector[1], gap_vector[0])))
            if connection_angle < 0:
                connection_angle += 180.0

        return {
            'distance': best_distance,
            'angle_diff': self._angle_difference_deg(first_angle, second_angle),
            'alignment_a': self._angle_difference_deg(first_angle, connection_angle),
            'alignment_b': self._angle_difference_deg(second_angle, connection_angle),
        }

    def _merge_candidate_group(self,
                               members: List[dict],
                               merge_distance_px: float,
                               profile: str) -> Optional[dict]:
        """将一组近邻候选合并为一个 CNT"""
        if not members:
            return None
        if len(members) == 1:
            return members[0]

        all_points = np.vstack([member['contour_local'].reshape(-1, 2) for member in members]).astype(np.int32)
        x_min = max(0, int(all_points[:, 0].min()) - 1)
        y_min = max(0, int(all_points[:, 1].min()) - 1)
        x_max = int(all_points[:, 0].max()) + 2
        y_max = int(all_points[:, 1].max()) + 2
        if x_max <= x_min or y_max <= y_min:
            return None

        merge_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        for member in members:
            relative_contour = (member['contour_local'] - np.array([x_min, y_min])).astype(np.int32)
            cv2.drawContours(merge_mask, [relative_contour], 0, 255, -1)

        kernel_size = max(3, min(int(round(merge_distance_px)) * 2 + 1, 21))
        if kernel_size % 2 == 0:
            kernel_size += 1
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        merge_mask = cv2.morphologyEx(merge_mask, cv2.MORPH_CLOSE, merge_kernel)

        merged = self._build_cnt_candidate(merge_mask, x_min, y_min, profile)
        if merged is None:
            return None
        merged['merged_from'] = len(members)
        return merged

    def _merge_nearby_cnt_candidates(self,
                                     candidates: List[dict],
                                     merge_distance_px: float,
                                     profile: str) -> List[dict]:
        """合并距离近且主方向一致的候选 CNT"""
        if merge_distance_px <= 0 or len(candidates) < 2:
            return candidates

        parents = list(range(len(candidates)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            root_first = find(first)
            root_second = find(second)
            if root_first != root_second:
                parents[root_second] = root_first

        for index in range(len(candidates)):
            for next_index in range(index + 1, len(candidates)):
                metrics = self._candidate_pair_metrics(candidates[index], candidates[next_index])
                if metrics.get('distance', float('inf')) > merge_distance_px:
                    continue
                if metrics.get('angle_diff', 180.0) > CNT_MERGE_MAX_ANGLE_DIFF_DEG:
                    continue
                if metrics.get('alignment_a', 180.0) > CNT_MERGE_MAX_ALIGNMENT_DEG:
                    continue
                if metrics.get('alignment_b', 180.0) > CNT_MERGE_MAX_ALIGNMENT_DEG:
                    continue
                union(index, next_index)

        groups: Dict[int, List[dict]] = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(find(index), []).append(candidate)

        merged_candidates: List[dict] = []
        for root in sorted(groups.keys()):
            members = groups[root]
            merged = self._merge_candidate_group(members, merge_distance_px, profile)
            if merged is not None:
                merged_candidates.append(merged)
            else:
                merged_candidates.extend(members)

        return merged_candidates

    def _passes_candidate_filters(self,
                                  candidate: dict,
                                  min_length_um: float,
                                  max_length_um: float,
                                  min_slenderness: float,
                                  profile: str) -> bool:
        """对候选 CNT 应用长度和长宽比过滤"""
        mode = (profile or "balanced").lower()
        if mode == "precision":
            length_factor = 1.10
            slenderness_factor = 1.15
        elif mode == "recall":
            length_factor = 0.85
            slenderness_factor = 0.85
        else:
            length_factor = 1.0
            slenderness_factor = 1.0

        length_um = float(candidate['length_pixels']) * self.scale_um_per_pixel
        eff_min_length = min_length_um * length_factor if min_length_um > 0 else 0.0
        eff_min_slenderness = min_slenderness * slenderness_factor if min_slenderness > 0 else 0.0

        if eff_min_length > 0 and length_um < eff_min_length:
            return False
        if max_length_um > 0 and length_um > max_length_um:
            return False

        if eff_min_slenderness > 0:
            slenderness_val = candidate.get('slenderness')
            if slenderness_val is None:
                area = float(candidate.get('area', 0.0))
                slenderness_val = ((candidate['length_pixels'] ** 2) / area) if area > 0 else None
            if slenderness_val is None or slenderness_val < eff_min_slenderness:
                return False

        return True

    def _candidate_to_measurement(self,
                                  candidate: dict,
                                  cnt_id: int,
                                  x_origin: int,
                                  y_origin: int) -> CNTMeasurement:
        """将候选 CNT 转换为最终测量结果"""
        contour_global = (candidate['contour_local'] + np.array([x_origin, y_origin])).astype(np.int32)
        bbox_x, bbox_y = candidate['skeleton_bbox_local']
        width_mean_px = float(candidate['width_mean_px'])
        width_median_px = float(candidate['width_median_px'])
        width_iqr_px = float(candidate['width_iqr_px'])

        return CNTMeasurement(
            id=cnt_id,
            length_pixels=float(candidate['length_pixels']),
            length_um=float(candidate['length_pixels']) * self.scale_um_per_pixel,
            contour=contour_global,
            skeleton=candidate['skeleton'],
            skeleton_bbox=(int(bbox_x + x_origin), int(bbox_y + y_origin)),
            width_mean_um=width_mean_px * self.scale_um_per_pixel if width_mean_px > 0 else None,
            width_median_um=width_median_px * self.scale_um_per_pixel if width_median_px > 0 else None,
            width_iqr_um=width_iqr_px * self.scale_um_per_pixel if width_iqr_px > 0 else None,
            slenderness=candidate.get('slenderness'),
        )

    # ===== CNT检测方法 =====
    def detect_cnts_hybrid(self,
                           min_length_um: float = 0.0,
                           max_length_um: float = 0.0,
                           min_slenderness: float = 0.0,
                           detection_profile: str = "balanced",
                           split_mode: str = "conservative",
                           merge_distance_px: float = 0.0,
                           roi: Optional[ROIRegion] = None) -> List[CNTMeasurement]:
        """混合检测方法

        Args:
            min_length_um (float): 最小长度(微米)
            max_length_um (float): 最大长度(微米)
            min_slenderness (float): 最小长宽比
            roi (Optional[ROIRegion]): 指定ROI区域

        Returns:
            List[CNTMeasurement]: 测量结果列表
        """
        _ = self._get_analysis_image()
        profile = (detection_profile or "balanced").lower()
        if self.binary_image is None:
            raise ValueError("请先进行图像预处理")

        if roi:
            roi.measurements = []
        else:
            self.measurements = []

        y1, y2, x1, x2 = self._get_analysis_region(roi)
        candidates: List[dict] = []
        contours, _ = cv2.findContours(
            self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 1:
                continue

            x_min = int(contour[:, 0, 0].min())
            x_max = int(contour[:, 0, 0].max())
            y_min = int(contour[:, 0, 1].min())
            y_max = int(contour[:, 0, 1].max())

            x_min = max(0, x_min - 1)
            y_min = max(0, y_min - 1)
            x_max = min(self.binary_image.shape[1], x_max + 2)
            y_max = min(self.binary_image.shape[0], y_max + 2)

            cnt_region = self.binary_image[y_min:y_max, x_min:x_max]

            relative_contour = contour - np.array([x_min, y_min])
            mask = np.zeros(cnt_region.shape, dtype=np.uint8)
            cv2.drawContours(mask, [relative_contour], 0, 255, -1)

            cnt_binary = cnt_region & mask
            split_regions = self._split_stuck_cnt_region(cnt_binary, split_mode=split_mode)
            for sub_binary in split_regions:
                candidate = self._build_cnt_candidate(
                    sub_binary,
                    x_offset=x_min,
                    y_offset=y_min,
                    profile=profile,
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates = self._merge_nearby_cnt_candidates(
            candidates,
            merge_distance_px=merge_distance_px,
            profile=profile,
        )

        output = roi.measurements if roi else self.measurements
        cnt_id = 0
        for candidate in candidates:
            if not self._passes_candidate_filters(
                candidate,
                min_length_um=min_length_um,
                max_length_um=max_length_um,
                min_slenderness=min_slenderness,
                profile=profile,
            ):
                continue

            measurement = self._candidate_to_measurement(candidate, cnt_id, x_origin=x1, y_origin=y1)
            output.append(measurement)
            cnt_id += 1

        return output

    # ===== 可视化方法 =====
    def get_visualization(self, roi: Optional[ROIRegion] = None) -> np.ndarray:
        """获取可视化结果"""
        if self.image is None:
            raise ValueError("请先加载图像")

        vis_image = self.image.copy()
        self._draw_scale_exclusion_annotation(vis_image)

        for r in self.rois:
            cv2.rectangle(vis_image, (r.x, r.y), (r.x + r.width, r.y + r.height),
                          r.color, 2)
            cv2.putText(vis_image, r.name, (r.x + 5, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)

        measurements = roi.measurements if roi else self.measurements

        for m in measurements:
            cv2.drawContours(vis_image, [m.contour], -1, (0, 255, 0), 2)

            rect = cv2.minAreaRect(m.contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(vis_image, [box], 0, (0, 0, 255), 1)

            M = cv2.moments(m.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                text = f"#{m.id}: L{m.length_um:.1f}um"
                cv2.putText(vis_image, text, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        return vis_image

    def get_visualization_with_skeleton(self, roi: Optional[ROIRegion] = None) -> np.ndarray:
        """获取带骨架的可视化结果"""
        if self.image is None:
            raise ValueError("请先加载图像")

        vis_image = self.image.copy()
        self._draw_scale_exclusion_annotation(vis_image)

        for r in self.rois:
            cv2.rectangle(vis_image, (r.x, r.y), (r.x + r.width, r.y + r.height),
                          r.color, 2)
            cv2.putText(vis_image, r.name, (r.x + 5, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)

        measurements = roi.measurements if roi else self.measurements

        for m in measurements:
            cv2.drawContours(vis_image, [m.contour], -1, (0, 255, 0), 2)

            if m.skeleton is not None and m.skeleton.size > 0:
                try:
                    x_offset, y_offset = m.skeleton_bbox
                    skeleton_h, skeleton_w = m.skeleton.shape[:2]

                    x_start = max(0, x_offset)
                    y_start = max(0, y_offset)
                    x_end = min(vis_image.shape[1], x_offset + skeleton_w)
                    y_end = min(vis_image.shape[0], y_offset + skeleton_h)

                    valid_x_start = x_start - x_offset
                    valid_y_start = y_start - y_offset
                    valid_x_end = valid_x_start + (x_end - x_start)
                    valid_y_end = valid_y_start + (y_end - y_start)

                    skeleton_valid = m.skeleton[valid_y_start:valid_y_end,
                    valid_x_start:valid_x_end]

                    skeleton_mask = skeleton_valid > 0

                    if skeleton_mask.size > 0:
                        vis_image[y_start:y_end, x_start:x_end][skeleton_mask] = [0, 0, 255]

                except Exception as e:
                    logger.warning(f"骨架显示错误 (CNT #{m.id}): {e}")

            M = cv2.moments(m.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                text = f"#{m.id}"
                cv2.putText(vis_image, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return vis_image

    def get_skeleton_preview(self, roi: Optional[ROIRegion] = None) -> np.ndarray:
        """获取骨架预览图"""
        if self.skeleton_overlay is not None:
            preview = self.skeleton_overlay.copy()
        elif self.image is not None:
            preview = self.image.copy()
        else:
            raise ValueError("请先加载图像")
        self._draw_scale_exclusion_annotation(preview)

        # 绘制ROI边框
        for r in self.rois:
            cv2.rectangle(preview, (r.x, r.y), (r.x + r.width, r.y + r.height),
                          r.color, 2)
            cv2.putText(preview, r.name, (r.x + 5, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)

        return preview

    # ===== 统计方法 =====
    def _get_local_measurements(self, roi: Optional[ROIRegion] = None) -> Tuple[List[CNTMeasurement], int, int, int, int]:
        """获取用于空间统计的测量结果及局部坐标基准"""
        measurements = roi.measurements if roi else self.measurements
        if self.image is None:
            raise ValueError("请先加载图像")

        if roi:
            return measurements, roi.x, roi.y, roi.width, roi.height

        height, width = self.image.shape[:2]
        return measurements, 0, 0, width, height

    def _extract_spatial_distribution_inputs(self,
                                             measurements: List[CNTMeasurement],
                                             offset_x: int,
                                             offset_y: int) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
        """提取局部坐标系下的中心点和轮廓。"""
        centroids: List[Tuple[float, float]] = []
        local_contours: List[np.ndarray] = []
        offset = np.array([offset_x, offset_y], dtype=np.int32)

        for measurement in measurements:
            contour = np.asarray(measurement.contour, dtype=np.int32)
            if contour.size == 0:
                continue

            local_contour = contour - offset
            local_contours.append(local_contour)

            moments = cv2.moments(local_contour)
            if moments["m00"] == 0:
                continue

            centroids.append((
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            ))

        return centroids, local_contours

    def _calculate_nearest_neighbor_stats(self,
                                          centroid_array: np.ndarray,
                                          width: int,
                                          height: int) -> Dict[str, float]:
        """计算中心点最近邻统计量。"""
        result = {
            'nearest_neighbor_mean_px': 0.0,
            'nearest_neighbor_std_px': 0.0,
            'nearest_neighbor_cv': 0.0,
            'nearest_neighbor_expected_mean_px': 0.0,
            'nearest_neighbor_index': 0.0,
        }
        if len(centroid_array) < 2:
            return result

        diffs = centroid_array[:, None, :] - centroid_array[None, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        nn_mean = float(np.mean(nearest_distances))
        nn_std = float(np.std(nearest_distances))
        nn_cv = float(nn_std / nn_mean) if nn_mean > 0 else 0.0

        expected_mean = 0.0
        area = float(max(width, 1) * max(height, 1))
        density = float(len(centroid_array) / area) if area > 0 else 0.0
        if density > 0:
            expected_mean = float(0.5 / np.sqrt(density))
        nearest_neighbor_index = float(nn_mean / expected_mean) if expected_mean > 0 else 0.0

        result.update({
            'nearest_neighbor_mean_px': nn_mean,
            'nearest_neighbor_std_px': nn_std,
            'nearest_neighbor_cv': nn_cv,
            'nearest_neighbor_expected_mean_px': expected_mean,
            'nearest_neighbor_index': nearest_neighbor_index,
        })
        return result

    def _build_centroid_count_grid(self,
                                   centroids: List[Tuple[float, float]],
                                   width: int,
                                   height: int,
                                   grid_size: int) -> np.ndarray:
        """按中心点数量构建网格。"""
        if not centroids or width <= 0 or height <= 0:
            return np.zeros((grid_size, grid_size), dtype=float)

        # 使用优化版本（如果可用）
        if NUMBA_AVAILABLE:
            centroids_array = np.array(centroids, dtype=np.float64)
            return _build_centroid_count_grid_vectorized(centroids_array, width, height, grid_size)

        # 回退到原始实现
        grid = np.zeros((grid_size, grid_size), dtype=float)
        width_scale = float(max(width, 1))
        height_scale = float(max(height, 1))
        for cx, cy in centroids:
            col = min(grid_size - 1, max(0, int(cx / width_scale * grid_size)))
            row = min(grid_size - 1, max(0, int(cy / height_scale * grid_size)))
            grid[row, col] += 1.0
        return grid

    def _build_coverage_ratio_grid(self,
                                   local_contours: List[np.ndarray],
                                   width: int,
                                   height: int,
                                   grid_size: int) -> np.ndarray:
        """按像素覆盖率构建网格，作为团聚程度的辅助参考。"""
        grid = np.zeros((grid_size, grid_size), dtype=float)
        if not local_contours or width <= 0 or height <= 0:
            return grid

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, local_contours, -1, 255, -1)

        for row in range(grid_size):
            y1 = int(row * height / grid_size)
            y2 = int((row + 1) * height / grid_size)
            for col in range(grid_size):
                x1 = int(col * width / grid_size)
                x2 = int((col + 1) * width / grid_size)
                cell = mask[y1:y2, x1:x2]
                grid[row, col] = float(np.count_nonzero(cell) / max(1, cell.size))
        return grid

    def _build_shadow_density_grid(self,
                                   offset_x: int,
                                   offset_y: int,
                                   width: int,
                                   height: int,
                                   grid_size: int,
                                   roi: Optional[ROIRegion] = None) -> np.ndarray:
        """按原图阴影强度构建网格，捕捉大束 CNT 下方的暗影团聚区。"""
        grid = np.zeros((grid_size, grid_size), dtype=float)
        if width <= 0 or height <= 0:
            return grid

        if self.original_image is not None:
            gray_source = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_source = self._get_analysis_gray_image()

        y1 = max(0, int(offset_y))
        y2 = min(gray_source.shape[0], int(offset_y + height))
        x1 = max(0, int(offset_x))
        x2 = min(gray_source.shape[1], int(offset_x + width))
        if y2 <= y1 or x2 <= x1:
            return grid

        local_gray = gray_source[y1:y2, x1:x2].copy()
        if local_gray.size == 0:
            return grid

        exclusion_mask = self.get_scale_exclusion_mask(roi)
        if exclusion_mask is not None and exclusion_mask.shape == local_gray.shape and np.any(exclusion_mask):
            valid_pixels = local_gray[exclusion_mask == 0]
            fill_value = int(np.median(valid_pixels)) if valid_pixels.size else 255
            local_gray[exclusion_mask > 0] = fill_value

        kernel_size = max(9, int(round(min(local_gray.shape[:2]) / 18.0)))
        kernel_size = min(kernel_size, 61)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(local_gray, (kernel_size, kernel_size), 0)

        darkness = 255.0 - blurred.astype(np.float32)
        low = float(np.percentile(darkness, 55))
        high = float(np.percentile(darkness, 96))
        if high <= low:
            return grid

        shadow_response = np.clip((darkness - low) / (high - low), 0.0, 1.0)

        local_height, local_width = shadow_response.shape[:2]
        for row in range(grid_size):
            cell_y1 = int(row * local_height / grid_size)
            cell_y2 = int((row + 1) * local_height / grid_size)
            for col in range(grid_size):
                cell_x1 = int(col * local_width / grid_size)
                cell_x2 = int((col + 1) * local_width / grid_size)
                cell = shadow_response[cell_y1:cell_y2, cell_x1:cell_x2]
                if cell.size == 0:
                    continue
                grid[row, col] = float(np.mean(cell))
        return grid

    def _summarize_density_grid(self, grid: np.ndarray) -> Dict[str, float]:
        """汇总网格分布统计量。"""
        flat_grid = np.asarray(grid, dtype=float).ravel()
        if flat_grid.size == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'cv': 0.0,
                'occupancy_ratio': 0.0,
                'entropy': 0.0,
                'dispersion_index': 0.0,
            }

        grid_mean = float(np.mean(flat_grid))
        grid_std = float(np.std(flat_grid))
        grid_cv = float(grid_std / grid_mean) if grid_mean > 0 else 0.0
        occupancy_ratio = float(np.count_nonzero(flat_grid) / flat_grid.size)
        dispersion_index = float(np.var(flat_grid) / grid_mean) if grid_mean > 0 else 0.0

        total_density = float(np.sum(flat_grid))
        grid_entropy = 0.0
        if total_density > 0 and flat_grid.size > 1:
            probabilities = flat_grid / total_density
            probabilities = probabilities[probabilities > 0]
            grid_entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(len(flat_grid)))

        return {
            'mean': grid_mean,
            'std': grid_std,
            'cv': grid_cv,
            'occupancy_ratio': occupancy_ratio,
            'entropy': grid_entropy,
            'dispersion_index': dispersion_index,
        }

    def _calculate_grid_morans_i(self, grid: np.ndarray) -> float:
        """基于网格计数计算 Moran's I。"""
        flat_grid = np.asarray(grid, dtype=float).ravel()
        if flat_grid.size == 0:
            return 0.0

        grid_size = int(np.asarray(grid).shape[0])
        mean_value = float(np.mean(flat_grid))
        denominator = float(np.sum((flat_grid - mean_value) ** 2))
        if denominator <= 0 or grid_size <= 0:
            return 0.0

        weights = 0.0
        numerator = 0.0
        for row in range(grid_size):
            for col in range(grid_size):
                index = row * grid_size + col
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n_row = row + dy
                    n_col = col + dx
                    if 0 <= n_row < grid_size and 0 <= n_col < grid_size:
                        neighbor_index = n_row * grid_size + n_col
                        weights += 1.0
                        numerator += (flat_grid[index] - mean_value) * (flat_grid[neighbor_index] - mean_value)

        if weights <= 0:
            return 0.0
        return float((len(flat_grid) / weights) * (numerator / denominator))

    def _calculate_uniformity_scores(self,
                                     nearest_neighbor_cv: float,
                                     grid_density_cv: float,
                                     morans_i: float,
                                     centroid_count: int) -> Dict[str, float]:
        """将不同方向的原始指标转换为统一方向的均匀性得分。"""
        if centroid_count < 2:
            return {
                'nearest_neighbor': 0.0,
                'grid_density': 0.0,
                'moran': 0.0,
                'overall': 0.0,
            }

        nn_score = float(100.0 / (1.0 + max(nearest_neighbor_cv, 0.0)))
        grid_score = float(100.0 / (1.0 + max(grid_density_cv, 0.0)))
        moran_score = float(100.0 * np.clip((1.0 - float(morans_i)) / 2.0, 0.0, 1.0))
        overall_score = float(np.mean([nn_score, grid_score, moran_score]))
        return {
            'nearest_neighbor': nn_score,
            'grid_density': grid_score,
            'moran': moran_score,
            'overall': overall_score,
        }

    def _calculate_aggregation_scores(self, uniformity_scores: Dict[str, float]) -> Dict[str, float]:
        """基于均匀性得分派生团聚风险得分，数值越大表示越团聚。"""
        keys = ('nearest_neighbor', 'grid_density', 'moran', 'overall')
        aggregation_scores: Dict[str, float] = {}
        for key in keys:
            score = float(uniformity_scores.get(key, 0.0))
            aggregation_scores[key] = float(np.clip(100.0 - score, 0.0, 100.0))
        return aggregation_scores

    def analyze_spatial_distribution(self,
                                     roi: Optional[ROIRegion] = None,
                                     grid_size: int = 10) -> Dict[str, object]:
        """分析 CNT 空间分布均匀性。

        说明:
            - 主要网格指标默认基于“每格 CNT 中心点数量”，更适合跨参数、跨图比较。
            - 像素覆盖率网格作为辅助参考，用于观察局部团聚和大块覆盖。
            - 额外返回统一方向的均匀性得分，便于在对比图中稳定展示。
        """
        measurements, offset_x, offset_y, width, height = self._get_local_measurements(roi)

        if not measurements or width <= 0 or height <= 0:
            return {}

        centroids, local_contours = self._extract_spatial_distribution_inputs(
            measurements, offset_x, offset_y
        )
        centroid_array = np.array(centroids, dtype=float)

        grid_size = max(2, int(grid_size))
        nn_stats = self._calculate_nearest_neighbor_stats(centroid_array, width, height)
        point_density_grid = self._build_centroid_count_grid(centroids, width, height, grid_size)
        coverage_density_grid = self._build_coverage_ratio_grid(local_contours, width, height, grid_size)
        shadow_density_grid = self._build_shadow_density_grid(offset_x, offset_y, width, height, grid_size, roi=roi)
        point_grid_stats = self._summarize_density_grid(point_density_grid)
        coverage_grid_stats = self._summarize_density_grid(coverage_density_grid)
        shadow_grid_stats = self._summarize_density_grid(shadow_density_grid)
        morans_i = self._calculate_grid_morans_i(point_density_grid)
        uniformity_scores = self._calculate_uniformity_scores(
            nn_stats['nearest_neighbor_cv'],
            point_grid_stats['cv'],
            morans_i,
            len(centroids),
        )
        aggregation_scores = self._calculate_aggregation_scores(uniformity_scores)

        return {
            'grid_size': int(grid_size),
            'centroid_count': int(len(centroids)),
            'centroids': [tuple(point) for point in centroids],
            **nn_stats,
            'grid_density_mean': point_grid_stats['mean'],
            'grid_density_std': point_grid_stats['std'],
            'grid_density_cv': point_grid_stats['cv'],
            'grid_entropy': point_grid_stats['entropy'],
            'occupancy_ratio': point_grid_stats['occupancy_ratio'],
            'grid_dispersion_index': point_grid_stats['dispersion_index'],
            'point_density_mean': point_grid_stats['mean'],
            'point_density_std': point_grid_stats['std'],
            'point_density_cv': point_grid_stats['cv'],
            'point_grid_entropy': point_grid_stats['entropy'],
            'point_occupancy_ratio': point_grid_stats['occupancy_ratio'],
            'point_dispersion_index': point_grid_stats['dispersion_index'],
            'coverage_density_mean': coverage_grid_stats['mean'],
            'coverage_density_std': coverage_grid_stats['std'],
            'coverage_density_cv': coverage_grid_stats['cv'],
            'coverage_grid_entropy': coverage_grid_stats['entropy'],
            'coverage_occupancy_ratio': coverage_grid_stats['occupancy_ratio'],
            'coverage_dispersion_index': coverage_grid_stats['dispersion_index'],
            'shadow_density_mean': shadow_grid_stats['mean'],
            'shadow_density_std': shadow_grid_stats['std'],
            'shadow_density_cv': shadow_grid_stats['cv'],
            'shadow_grid_entropy': shadow_grid_stats['entropy'],
            'shadow_occupancy_ratio': shadow_grid_stats['occupancy_ratio'],
            'shadow_dispersion_index': shadow_grid_stats['dispersion_index'],
            'morans_i': morans_i,
            'density_grid': point_density_grid.tolist(),
            'point_density_grid': point_density_grid.tolist(),
            'coverage_density_grid': coverage_density_grid.tolist(),
            'shadow_density_grid': shadow_density_grid.tolist(),
            'uniformity_scores': uniformity_scores,
            'aggregation_scores': aggregation_scores,
        }

    def _is_cnt_in_hotspot(self, measurement: CNTMeasurement, hotspot_mask: np.ndarray, 
                           offset_x: int, offset_y: int, width: int, height: int, 
                           grid_size: int) -> bool:
        """判断CNT的中心点是否落在团聚热点网格内
        
        Args:
            measurement: CNT测量结果
            hotspot_mask: 热点网格掩码
            offset_x: ROI的X偏移
            offset_y: ROI的Y偏移
            width: ROI宽度
            height: ROI高度
            grid_size: 网格大小
            
        Returns:
            bool: True表示CNT在团聚区域内
        """
        if hotspot_mask is None or hotspot_mask.size == 0:
            return False
            
        # 计算CNT中心点
        contour = np.asarray(measurement.contour, dtype=np.int32)
        if contour.size == 0:
            return False
            
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return False
            
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        
        # 转换为局部坐标
        local_cx = cx - offset_x
        local_cy = cy - offset_y
        
        # 转换为网格坐标
        if width <= 0 or height <= 0:
            return False
            
        grid_col = int(local_cx / width * grid_size)
        grid_row = int(local_cy / height * grid_size)
        
        # 检查是否在网格范围内
        if grid_row < 0 or grid_row >= grid_size or grid_col < 0 or grid_col >= grid_size:
            return False
            
        return bool(hotspot_mask[grid_row, grid_col])

    @staticmethod
    def _get_measurement_local_contour(measurement: CNTMeasurement,
                                       offset_x: int,
                                       offset_y: int) -> Optional[np.ndarray]:
        """Convert a measurement contour into ROI-local coordinates."""
        contour = np.asarray(measurement.contour, dtype=np.float32)
        if contour.size == 0:
            return None

        local_contour = contour.copy()
        local_contour[:, 0, 0] -= float(offset_x)
        local_contour[:, 0, 1] -= float(offset_y)
        return local_contour

    @staticmethod
    def _upsample_hotspot_mask(mask: np.ndarray, factor: int) -> np.ndarray:
        """Upsample a coarse hotspot grid for contour-overlap estimation."""
        if mask is None or getattr(mask, 'size', 0) == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        upscale = max(1, int(factor))
        return np.kron(mask.astype(np.uint8), np.ones((upscale, upscale), dtype=np.uint8))

    def _get_measurement_hotspot_overlap(self,
                                         measurement: CNTMeasurement,
                                         hotspot_mask: np.ndarray,
                                         offset_x: int,
                                         offset_y: int,
                                         width: int,
                                         height: int,
                                         grid_size: int,
                                         upsample_factor: int = SPATIAL_HOTSPOT_MASK_UPSAMPLE) -> float:
        """Estimate the fraction of a CNT contour area that overlaps hotspot cells."""
        if hotspot_mask is None or getattr(hotspot_mask, 'size', 0) == 0 or width <= 0 or height <= 0:
            return 0.0

        local_contour = self._get_measurement_local_contour(measurement, offset_x, offset_y)
        if local_contour is None:
            return 0.0

        mask_size = max(1, int(grid_size) * max(1, int(upsample_factor)))
        scale_x = (mask_size - 1) / max(width - 1, 1)
        scale_y = (mask_size - 1) / max(height - 1, 1)
        scaled_contour = np.empty_like(local_contour, dtype=np.int32)
        scaled_contour[:, 0, 0] = np.clip(np.round(local_contour[:, 0, 0] * scale_x), 0, mask_size - 1).astype(np.int32)
        scaled_contour[:, 0, 1] = np.clip(np.round(local_contour[:, 0, 1] * scale_y), 0, mask_size - 1).astype(np.int32)

        contour_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [scaled_contour], 1)
        contour_area = int(np.count_nonzero(contour_mask))
        if contour_area <= 0:
            return 0.0

        hotspot_scaled = self._upsample_hotspot_mask(hotspot_mask, max(1, int(upsample_factor)))
        if hotspot_scaled.shape != contour_mask.shape:
            hotspot_scaled = cv2.resize(
                hotspot_scaled,
                (contour_mask.shape[1], contour_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlap_area = int(np.count_nonzero((contour_mask > 0) & (hotspot_scaled > 0)))
        return float(overlap_area / contour_area)

    def _is_measurement_agglomerated(self,
                                     measurement: CNTMeasurement,
                                     active_mask: np.ndarray,
                                     severe_mask: np.ndarray,
                                     offset_x: int,
                                     offset_y: int,
                                     width: int,
                                     height: int,
                                     grid_size: int) -> bool:
        """Classify agglomeration using severe-cell coverage first and centroid fallback last."""
        severe_overlap = self._get_measurement_hotspot_overlap(
            measurement,
            severe_mask,
            offset_x,
            offset_y,
            width,
            height,
            grid_size,
        )
        if severe_overlap > 0.0:
            return True

        hotspot_overlap = self._get_measurement_hotspot_overlap(
            measurement,
            active_mask,
            offset_x,
            offset_y,
            width,
            height,
            grid_size,
        )
        if hotspot_overlap >= float(SPATIAL_HOTSPOT_OVERLAP_RATIO_THRESHOLD):
            return True

        return self._is_cnt_in_hotspot(measurement, active_mask, offset_x, offset_y, width, height, grid_size)

    @staticmethod
    def _normalize_hotspot_grid(grid: np.ndarray, percentile: float = 85.0) -> np.ndarray:
        """Normalize a hotspot grid into 0-1 range using a high-percentile cap."""
        grid = np.asarray(grid, dtype=float)
        if grid.size == 0:
            return np.zeros((0, 0), dtype=float)

        positive = grid[grid > 0]
        if positive.size == 0:
            return np.zeros_like(grid, dtype=float)

        upper = float(np.percentile(positive, percentile))
        if upper <= 0:
            upper = max(float(np.max(positive)), 1e-6)
        return np.clip(grid / upper, 0.0, 1.0)

    def _build_spatial_hotspot_masks(self, spatial_distribution: Dict[str, object]) -> Dict[str, np.ndarray]:
        """Build hotspot masks from spatial-distribution grids."""
        point_grid = np.array(
            spatial_distribution.get('point_density_grid') or spatial_distribution.get('density_grid') or [],
            dtype=float,
        )
        coverage_grid = np.array(spatial_distribution.get('coverage_density_grid') or [], dtype=float)
        shadow_grid = np.array(spatial_distribution.get('shadow_density_grid') or [], dtype=float)

        if point_grid.size == 0 and coverage_grid.size == 0 and shadow_grid.size == 0:
            empty = np.zeros((0, 0), dtype=float)
            return {
                'point_grid': empty,
                'coverage_grid': empty,
                'shadow_grid': empty,
                'hotspot_grid': empty,
                'hotspot_mask': np.zeros((0, 0), dtype=bool),
                'severe_mask': np.zeros((0, 0), dtype=bool),
            }


        point_norm = self._normalize_hotspot_grid(point_grid, percentile=SPATIAL_HOTSPOT_POINT_PERCENTILE)
        coverage_norm = self._normalize_hotspot_grid(coverage_grid, percentile=SPATIAL_HOTSPOT_COVERAGE_PERCENTILE)
        shadow_norm = self._normalize_hotspot_grid(shadow_grid, percentile=SPATIAL_HOTSPOT_SHADOW_PERCENTILE)
        hotspot_grid = np.clip(
            point_norm * SPATIAL_HOTSPOT_POINT_WEIGHT +
            coverage_norm * SPATIAL_HOTSPOT_COVERAGE_WEIGHT +
            shadow_norm * SPATIAL_HOTSPOT_SHADOW_WEIGHT,
            0.0,
            1.0,
        )

        active_mask = (point_grid > 0) | (coverage_grid > 0) | (shadow_grid > 0)
        active_scores = hotspot_grid[active_mask]
        hotspot_mask = np.zeros_like(hotspot_grid, dtype=bool)
        severe_mask = np.zeros_like(hotspot_grid, dtype=bool)

        if active_scores.size > 0 and float(np.max(active_scores) - np.min(active_scores)) >= SPATIAL_HOTSPOT_ACTIVE_SCORE_RANGE_MIN:
            hotspot_percentile = (
                SPATIAL_HOTSPOT_MASK_PERCENTILE_LARGE
                if active_scores.size >= 6 else
                SPATIAL_HOTSPOT_MASK_PERCENTILE_SMALL
            )
            severe_percentile = (
                SPATIAL_HOTSPOT_SEVERE_PERCENTILE_LARGE
                if active_scores.size >= 8 else
                SPATIAL_HOTSPOT_SEVERE_PERCENTILE_SMALL
            )
            hotspot_threshold = float(np.percentile(active_scores, hotspot_percentile))
            severe_threshold = float(np.percentile(active_scores, severe_percentile))
            hotspot_mask = active_mask & (hotspot_grid >= hotspot_threshold) & (hotspot_grid > 0)
            severe_mask = active_mask & (hotspot_grid >= severe_threshold) & (hotspot_grid > 0)

        return {
            'point_grid': point_grid,
            'coverage_grid': coverage_grid,
            'shadow_grid': shadow_grid,
            'hotspot_grid': hotspot_grid,
            'hotspot_mask': hotspot_mask,
            'severe_mask': severe_mask,
        }

    @staticmethod
    def _summarize_length_statistics(measurements: List[CNTMeasurement]) -> Dict[str, object]:
        """Summarize CNT length statistics for a measurement subset."""
        lengths = [float(m.length_um) for m in measurements if m.length_um is not None]
        length_dist = {label: 0 for label in LENGTH_DISTRIBUTION_LABELS}
        if not lengths:
            return {
                'count': 0,
                'length_mean': 0.0,
                'length_std': 0.0,
                'length_min': 0.0,
                'length_max': 0.0,
                'lengths': [],
                'length_distribution': length_dist,
            }

        for i, label in enumerate(LENGTH_DISTRIBUTION_LABELS):
            length_dist[label] = int(sum(
                1 for length in lengths
                if LENGTH_DISTRIBUTION_BINS_UM[i] <= length < LENGTH_DISTRIBUTION_BINS_UM[i + 1]
            ))

        return {
            'count': int(len(lengths)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
            'length_min': float(np.min(lengths)),
            'length_max': float(np.max(lengths)),
            'lengths': lengths,
            'length_distribution': length_dist,
        }

    def get_dispersed_statistics(self,
                                 roi: Optional[ROIRegion] = None,
                                 strictness: str = "all_hotspots") -> Dict[str, object]:
        """获取排除团聚区域后的分散CNT统计信息
        
        Args:
            roi (Optional[ROIRegion]): 指定ROI区域，若为None则使用全局测量结果
            
        Returns:
            Dict: 包含分散CNT和团聚CNT的统计信息
        """
        measurements = roi.measurements if roi else self.measurements
        mode = str(strictness or "all_hotspots").lower()
        if mode not in {"all_hotspots", "hotspot_only", "severe_only"}:
            raise ValueError(f"Unsupported hotspot strictness: {strictness}")

        empty_length_stats = self._summarize_length_statistics([])
        if not measurements:
            return {
                'strictness': mode,
                'total_count': 0,
                'dispersed_count': 0,
                'agglomerated_count': 0,
                'dispersed_ratio': 0.0,
                'agglomerated_ratio': 0.0,
                'dispersed_measurements': [],
                'agglomerated_measurements': [],
                'dispersed_length_stats': empty_length_stats,
                'agglomerated_length_stats': empty_length_stats,
            }
        
        # 获取空间分布分析结果
        spatial_distribution = self.analyze_spatial_distribution(roi)
        if not spatial_distribution:
            dispersed_length_stats = self._summarize_length_statistics(measurements)
            # 如果没有空间分布数据，返回全部作为分散CNT
            return {
                'strictness': mode,
                'total_count': len(measurements),
                'dispersed_count': len(measurements),
                'agglomerated_count': 0,
                'dispersed_ratio': 1.0,
                'agglomerated_ratio': 0.0,
                'dispersed_measurements': list(measurements),
                'agglomerated_measurements': [],
                'dispersed_length_stats': dispersed_length_stats,
                'agglomerated_length_stats': empty_length_stats,
            }
        
        # 获取热点掩码
        hotspot_info = self._build_spatial_hotspot_masks(spatial_distribution)
        hotspot_mask = hotspot_info['hotspot_mask']
        severe_mask = hotspot_info['severe_mask']
        active_mask = severe_mask if mode == 'severe_only' else hotspot_mask
        
        if active_mask.size == 0:
            dispersed_length_stats = self._summarize_length_statistics(measurements)
            return {
                'strictness': mode,
                'total_count': len(measurements),
                'dispersed_count': len(measurements),
                'agglomerated_count': 0,
                'dispersed_ratio': 1.0,
                'agglomerated_ratio': 0.0,
                'dispersed_measurements': list(measurements),
                'agglomerated_measurements': [],
                'dispersed_length_stats': dispersed_length_stats,
                'agglomerated_length_stats': empty_length_stats,
                'hotspot_mask': hotspot_mask,
                'severe_mask': severe_mask,
            }
        
        # 构建热点网格
            
        # 归一化网格
        
        
        # 构建热点掩码
        
        # 获取ROI信息
        _, offset_x, offset_y, width, height = self._get_local_measurements(roi)
        grid_size = int(spatial_distribution.get('grid_size', 10))
        
        # 分类CNT
        dispersed_measurements = []
        agglomerated_measurements = []
        
        for measurement in measurements:
            if self._is_measurement_agglomerated(
                measurement,
                active_mask,
                severe_mask,
                offset_x,
                offset_y,
                width,
                height,
                grid_size,
            ):
                agglomerated_measurements.append(measurement)
            else:
                dispersed_measurements.append(measurement)
        
        dispersed_count = len(dispersed_measurements)
        agglomerated_count = len(agglomerated_measurements)
        total_count = len(measurements)
        dispersed_ratio = dispersed_count / total_count if total_count > 0 else 0.0
        agglomerated_ratio = agglomerated_count / total_count if total_count > 0 else 0.0

        return {
            'strictness': mode,
            'total_count': total_count,
            'dispersed_count': dispersed_count,
            'agglomerated_count': agglomerated_count,
            'dispersed_ratio': dispersed_ratio,
            'agglomerated_ratio': agglomerated_ratio,
            'dispersed_measurements': dispersed_measurements,
            'agglomerated_measurements': agglomerated_measurements,
            'dispersed_length_stats': self._summarize_length_statistics(dispersed_measurements),
            'agglomerated_length_stats': self._summarize_length_statistics(agglomerated_measurements),
            'hotspot_mask': hotspot_mask,
            'severe_mask': severe_mask,
        }

    def get_statistics(self, roi: Optional[ROIRegion] = None) -> Dict[str, object]:
        """获取测量结果的统计信息

        Args:
            roi (Optional[ROIRegion]): 指定ROI区域，若为None则使用全局测量结果

        Returns:
            Dict[str, float]: 包含 mean, median, std, min, max 等统计值
        """
        measurements = roi.measurements if roi else self.measurements

        if not measurements:
            return {}

        length_stats = self._summarize_length_statistics(measurements)
        spatial_distribution = self.analyze_spatial_distribution(roi)

        return {
            **length_stats,
            'spatial_distribution': spatial_distribution,
        }


# ===== Numba 优化函数 =====
@jit(nopython=True, cache=True)
def _calculate_path_length_fast(path_array: np.ndarray) -> float:
    """使用 Numba JIT 加速的路径长度计算
    
    Args:
        path_array: shape (N, 2) 的路径点数组，dtype=float64
        
    Returns:
        float: 路径总长度
    """
    if len(path_array) < 2:
        return 0.0
    
    length = 0.0
    for i in range(1, len(path_array)):
        dy = path_array[i, 0] - path_array[i-1, 0]
        dx = path_array[i, 1] - path_array[i-1, 1]
        length += np.sqrt(dy * dy + dx * dx)
    
    return length


@jit(nopython=True, cache=True)
def _build_centroid_count_grid_vectorized(centroids: np.ndarray,
                                          width: int,
                                          height: int,
                                          grid_size: int) -> np.ndarray:
    """使用 Numba JIT 加速的网格构建
    
    Args:
        centroids: shape (N, 2) 的中心点数组
        width: 图像宽度
        height: 图像高度
        grid_size: 网格大小
        
    Returns:
        np.ndarray: shape (grid_size, grid_size) 的网格计数
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    if len(centroids) == 0 or width <= 0 or height <= 0:
        return grid
    
    width_scale = float(max(width, 1))
    height_scale = float(max(height, 1))
    
    for i in range(len(centroids)):
        cx = centroids[i, 0]
        cy = centroids[i, 1]
        
        col = int(cx / width_scale * grid_size)
        row = int(cy / height_scale * grid_size)
        
        # 边界检查
        if col < 0:
            col = 0
        elif col >= grid_size:
            col = grid_size - 1
            
        if row < 0:
            row = 0
        elif row >= grid_size:
            row = grid_size - 1
        
        grid[row, col] += 1.0
    
    return grid
