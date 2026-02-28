"""
图像分析核心模块 - 包含CNTAnalyzer类及其所有图像处理方法
"""
import logging
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from skimage.morphology import skeletonize

from models import ROIRegion, CNTMeasurement
from utils import (
    SCALE_BAR_BLUE_THRESHOLD, SCALE_BAR_BLUE_SCORE_MIN, SCALE_BAR_MIN_SPAN_PX,
    SCALE_BAR_BGR_DIST_MAX, SCALE_BAR_ROI_X_RATIO, SCALE_BAR_ROI_Y_RATIO,
    SCALE_BAR_ASPECT_RATIO_MIN, SCALE_BAR_ASPECT_RATIO_STRICT,
    SCALE_BAR_OCR_MATCH_THRESHOLD, SCALE_BAR_VALUE_RANGE,
    SKELETON_ANGLE_THRESHOLDS
)

logger = logging.getLogger(__name__)


class CNTAnalyzer:
    """CNT图像分析核心类"""

    def __init__(self):
        self.image = None
        self.processed_image = None
        self.binary_image = None
        self.skeleton_image = None
        self.skeleton_overlay = None
        self.scale_um_per_pixel = 0.1
        self.measurements: List[CNTMeasurement] = []
        self.rois: List[ROIRegion] = []
        self.ocr_templates = None

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

    def load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"无法加载图像: {path}")
        return self.image

    def set_scale(self, pixels: float, micrometers: float):
        """设置比例尺"""
        if pixels <= 0 or micrometers <= 0:
            raise ValueError("像素数和微米数必须大于0")
        self.scale_um_per_pixel = micrometers / pixels

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
        h, w = image.shape[:2]
        search_x1 = max(0, bar_x - int(bar_w * 0.3))
        search_x2 = min(w, bar_x + bar_w + int(bar_w * 0.3))
        search_y1 = max(0, bar_y - int(bar_h * 10) - 20)
        search_y2 = min(h, bar_y + int(bar_h * 8) + 20)

        if search_y2 <= search_y1 or search_x2 <= search_x1:
            return None

        search_roi = image[search_y1:search_y2, search_x1:search_x2]
        hsv_t = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        white_mask = (hsv_t[:, :, 1] <= 80) & (hsv_t[:, :, 2] >= 160)
        white_mask = (white_mask.astype(np.uint8) * 255)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if white_contours:
            best_white = None
            best_score = None
            for c in white_contours:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch
                if area < 30:
                    continue
                if cw < bar_w * 0.2 or ch < bar_h * 1.0:
                    continue
                center_y = search_y1 + y + ch / 2
                dist = abs(center_y - bar_y)
                score = area / (1 + dist)
                if best_score is None or score > best_score:
                    best_score = score
                    best_white = (x, y, cw, ch)
            if best_white is not None:
                x, y, cw, ch = best_white
                return search_roi[y:y + ch, x:x + cw]

        # 备选方案
        text_x1 = max(0, bar_x - int(bar_w * 0.2))
        text_x2 = min(w, bar_x + bar_w + int(bar_w * 0.2))
        text_y2 = max(0, bar_y - 2)
        text_y1 = max(0, bar_y - int(bar_h * 10) - 20)
        if text_y2 > text_y1 and text_x2 > text_x1:
            return image[text_y1:text_y2, text_x1:text_x2]

        return None

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
            for k, temps in self.ocr_templates.items():
                for temp in temps:
                    score = float(cv2.matchTemplate(norm, temp, cv2.TM_CCOEFF_NORMED)[0][0])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_char = k
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
        except Exception:
            pass
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

    def detect_scale_bar(self) -> Optional[dict]:
        """检测图像中的比例尺"""
        if self.image is None:
            raise ValueError("请先加载图像")

        image = self.image
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
            return None

        bar_x, bar_y, bar_w, bar_h = best
        bar_x += x0
        bar_y += y0

        text_roi = self._extract_text_roi(image, bar_x, bar_y, bar_w, bar_h)
        micrometers = self._recognize_scale_value(text_roi) if text_roi is not None else None

        return {"pixels": float(bar_w), "micrometers": micrometers}

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
        if roi:
            x, y, w, h = roi.x, roi.y, roi.width, roi.height
            return (y, y + h, x, x + w)
        else:
            h, w = self.image.shape[:2]
            return (0, h, 0, w)

    # ===== 自适应参数推荐 =====
    def suggest_preprocess_params(self, roi: Optional[ROIRegion] = None) -> dict:
        """根据图像特征自动推荐预处理参数

        分析图像的灰度直方图和对比度，推荐合适的模糊核、自适应块大小和常数C。

        Returns:
            dict: 包含 blur_kernel, adaptive_block, adaptive_c 的推荐值
        """
        if self.image is None:
            raise ValueError("请先加载图像")

        y1, y2, x1, x2 = self._get_analysis_region(roi)
        analysis_image = self.image[y1:y2, x1:x2]
        gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)

        # 缩小图像加速分析
        h, w = gray.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # 用 Otsu 分割前景/背景
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_mask = gray > otsu_thresh
        bright_ratio = float(bright_mask.mean()) if bright_mask.size > 0 else 0.0

        bright_mean = float(gray[bright_mask].mean()) if bright_mask.any() else float(gray.mean())
        dark_mean = float(gray[~bright_mask].mean()) if (~bright_mask).any() else float(gray.mean())
        delta = abs(bright_mean - dark_mean)

        # 根据对比度选择参数
        if delta < 30:
            # 低对比度：需要更强的模糊和更大的块
            blur_kernel = 11
            adaptive_block = 21
            adaptive_c = 3
        elif delta < 60:
            blur_kernel = 11
            adaptive_block = 19
            adaptive_c = 3
        else:
            # 高对比度：较小的模糊和块即可
            blur_kernel = 9
            adaptive_block = 15
            adaptive_c = 2

        # 确保奇数
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        if adaptive_block % 2 == 0:
            adaptive_block += 1
        if adaptive_block < 3:
            adaptive_block = 3

        # 限制范围
        blur_kernel = max(7, min(15, blur_kernel))
        adaptive_block = max(11, min(31, adaptive_block))
        adaptive_c = max(1, min(5, adaptive_c))

        # 验证前景比例，微调C值
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, adaptive_block, adaptive_c
        )
        fg_ratio = float((binary > 0).mean())

        # 目标前景比例约 3%-15%
        if fg_ratio > 0.20:
            adaptive_c = min(adaptive_c + 2, 8)
        elif fg_ratio < 0.01:
            adaptive_c = max(adaptive_c - 1, 0)

        return {
            "blur_kernel": blur_kernel,
            "adaptive_block": adaptive_block,
            "adaptive_c": adaptive_c,
        }

    # ===== 预处理方法 =====
    def preprocess(self, blur_kernel: int = 5,
                   adaptive_block: int = 11,
                   adaptive_c: int = 2,
                   threshold_invert: bool = True,
                   roi: Optional[ROIRegion] = None) -> np.ndarray:
        """图像预处理"""
        if self.image is None:
            raise ValueError("请先加载图像")

        y1, y2, x1, x2 = self._get_analysis_region(roi)
        analysis_image = self.image[y1:y2, x1:x2]

        gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        threshold_type = cv2.THRESH_BINARY_INV if threshold_invert else cv2.THRESH_BINARY
        self.binary_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type, adaptive_block, adaptive_c
        )

        kernel = np.ones((3, 3), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)

        self.skeleton_image = self._skeletonize(self.binary_image.copy())
        self._generate_skeleton_overlay(y1, y2, x1, x2)

        self.processed_image = self.binary_image.copy()
        return self.binary_image

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """生成骨架"""
        binary_bool = binary > 0
        skeleton = skeletonize(binary_bool)
        return (skeleton.astype(np.uint8) * 255)

    def _generate_skeleton_overlay(self, y1: int, y2: int, x1: int, x2: int):
        """生成骨架叠加到原图"""
        if self.image is None or self.skeleton_image is None:
            return
        self.skeleton_overlay = self.image.copy()
        skeleton_mask = self.skeleton_image > 0
        self.skeleton_overlay[y1:y2, x1:x2][skeleton_mask] = [0, 0, 255]

    # ===== 骨架处理方法 =====
    def _build_skeleton_neighbors(self, skeleton: np.ndarray) -> dict:
        """构建骨架邻接表"""
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        if len(skeleton_points) < 2:
            return {}
        neighbors = {}
        for y, x in skeleton_points:
            y = int(y)
            x = int(x)
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] > 0:
                            neigh.append((int(ny), int(nx)))
            neighbors[(y, x)] = neigh
        return neighbors

    def _count_endpoints(self, skeleton: np.ndarray, neighbors: Optional[dict] = None) -> int:
        """计算骨架端点数量"""
        if neighbors is None:
            neighbors = self._build_skeleton_neighbors(skeleton)
        if not neighbors:
            return 0
        return sum(1 for neigh in neighbors.values() if len(neigh) == 1)

    def _extract_primary_path(self, skeleton: np.ndarray, neighbors: Optional[dict] = None) -> np.ndarray:
        """提取骨架主路径"""
        if neighbors is None:
            neighbors = self._build_skeleton_neighbors(skeleton)
        if len(neighbors) < 2:
            return skeleton

        endpoints = [p for p, neigh in neighbors.items() if len(neigh) == 1]
        if len(endpoints) < 2:
            return skeleton

        angle_thresholds = SKELETON_ANGLE_THRESHOLDS

        def choose_next(prev_pt, curr_pt, candidates, degree, threshold_cos):
            if len(candidates) == 1:
                return candidates[0]
            if prev_pt is None:
                return candidates[0]
            if degree < 3:
                return candidates[0]
            in_vec = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]], dtype=float)
            in_norm = np.linalg.norm(in_vec)
            if in_norm == 0:
                return candidates[0]
            best = None
            best_cos = 1.0
            for cand in candidates:
                out_vec = np.array([cand[0] - curr_pt[0], cand[1] - curr_pt[1]], dtype=float)
                out_norm = np.linalg.norm(out_vec)
                if out_norm == 0:
                    continue
                cos_val = float(np.dot(in_vec, out_vec) / (in_norm * out_norm))
                if cos_val < best_cos:
                    best_cos = cos_val
                    best = cand
            if best is None or best_cos > threshold_cos:
                return None
            return best

        def trace(start_pt, threshold_cos):
            path = [start_pt]
            prev_pt = None
            curr_pt = start_pt
            while True:
                neigh = neighbors.get(curr_pt, [])
                candidates = [n for n in neigh if n != prev_pt]
                if not candidates:
                    break
                next_pt = choose_next(prev_pt, curr_pt, candidates, len(neigh), threshold_cos)
                if next_pt is None:
                    break
                path.append(next_pt)
                prev_pt, curr_pt = curr_pt, next_pt
            return path

        best_path = None
        best_length = -1.0
        for angle in angle_thresholds:
            threshold_cos = float(np.cos(np.deg2rad(angle)))
            for ep in endpoints:
                path = trace(ep, threshold_cos)
                if len(path) < 2:
                    continue
                length = 0.0
                for i in range(1, len(path)):
                    length += np.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                if length > best_length:
                    best_length = length
                    best_path = path

        if not best_path or len(best_path) < 2:
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

        endpoints = [p for p, neigh in neighbors.items() if len(neigh) == 1]
        visited_edges = set()
        path_lengths = []
        angle_threshold_cos = float(np.cos(np.deg2rad(150)))

        def edge_key(a, b):
            return (a, b) if a <= b else (b, a)

        def choose_next(prev_pt, curr_pt, candidates, degree):
            if len(candidates) == 1:
                return candidates[0]
            if prev_pt is None:
                return candidates[0]
            if degree < 3:
                return candidates[0]
            in_vec = np.array([curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]], dtype=float)
            in_norm = np.linalg.norm(in_vec)
            if in_norm == 0:
                return candidates[0]
            best = None
            best_cos = 1.0
            for cand in candidates:
                out_vec = np.array([cand[0] - curr_pt[0], cand[1] - curr_pt[1]], dtype=float)
                out_norm = np.linalg.norm(out_vec)
                if out_norm == 0:
                    continue
                cos_val = float(np.dot(in_vec, out_vec) / (in_norm * out_norm))
                if cos_val < best_cos:
                    best_cos = cos_val
                    best = cand
            if best is None or best_cos > angle_threshold_cos:
                return None
            return best

        for ep in endpoints:
            curr_pt = ep
            prev_pt = None
            length = 0.0
            while True:
                neigh = neighbors.get(curr_pt, [])
                candidates = [n for n in neigh if n != prev_pt and edge_key(curr_pt, n) not in visited_edges]
                if not candidates:
                    break
                next_pt = choose_next(prev_pt, curr_pt, candidates, len(neigh))
                if next_pt is None:
                    break
                length += np.hypot(next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
                visited_edges.add(edge_key(curr_pt, next_pt))
                prev_pt, curr_pt = curr_pt, next_pt
            path_lengths.append(length)

        return max(path_lengths) if path_lengths else 0.0

    # ===== 宽度测量方法 =====
    def _measure_width(self, skeleton: np.ndarray, cnt_binary: np.ndarray) -> float:
        """通过骨架法测量CNT平均宽度

        对骨架上的每个点，计算其到最近轮廓边界的距离（即半宽），
        取所有点的平均值再乘以2得到平均宽度。

        使用距离变换实现，效率远高于逐点计算。

        Args:
            skeleton: 骨架二值图
            cnt_binary: CNT区域二值图

        Returns:
            float: 平均宽度（像素）
        """
        if skeleton is None or cnt_binary is None:
            return 0.0

        # 距离变换：计算每个前景像素到最近背景像素的距离
        dist_transform = cv2.distanceTransform(cnt_binary, cv2.DIST_L2, 5)

        # 获取骨架点位置
        skeleton_mask = skeleton > 0
        if not skeleton_mask.any():
            return 0.0

        # 骨架点处的距离值即为半宽
        half_widths = dist_transform[skeleton_mask]

        if len(half_widths) == 0:
            return 0.0

        # 平均宽度 = 2 × 平均半宽
        mean_width = float(np.mean(half_widths)) * 2.0
        return mean_width

    # ===== CNT检测方法 =====
    def detect_cnts_hybrid(self,
                           min_length_um: float = 0.0,
                           max_length_um: float = 0.0,
                           min_slenderness: float = 0.0,
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
        if self.image is None:
            raise ValueError("请先加载图像")
        if self.binary_image is None:
            raise ValueError("请先进行图像预处理")

        if roi:
            roi.measurements = []
        else:
            self.measurements = []

        y1, y2, x1, x2 = self._get_analysis_region(roi)

        contours, _ = cv2.findContours(
            self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cnt_id = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1:
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

            skeleton_region = self._skeletonize(cnt_binary)

            neighbors = self._build_skeleton_neighbors(skeleton_region)
            if len(neighbors) < 2:
                continue

            skeleton_region = self._extract_primary_path(skeleton_region, neighbors)

            neighbors = self._build_skeleton_neighbors(skeleton_region)
            if len(neighbors) < 2:
                continue

            if self._count_endpoints(skeleton_region, neighbors) != 2:
                continue

            length_px = self._calculate_skeleton_length(skeleton_region, neighbors)
            if length_px < 1:
                continue

            length_um = length_px * self.scale_um_per_pixel

            if min_length_um > 0 and length_um < min_length_um:
                continue
            if max_length_um > 0 and length_um > max_length_um:
                continue
            if min_slenderness > 0 and area > 0:
                slenderness = (length_px * length_px) / area
                if slenderness < min_slenderness:
                    continue

            # 测量宽度
            width_px = self._measure_width(skeleton_region, cnt_binary)
            width_um = width_px * self.scale_um_per_pixel if width_px > 0 else None

            # 计算长宽比（细长度）
            slenderness_val = (length_px / width_px) if width_px > 0 else None

            adjusted_contour = contour + np.array([x1, y1])

            measurement = CNTMeasurement(
                id=cnt_id,
                length_pixels=length_px,
                length_um=length_um,
                contour=adjusted_contour,
                skeleton=skeleton_region,
                skeleton_bbox=(x_min + x1, y_min + y1),
                width_mean_um=width_um,
                slenderness=slenderness_val
            )

            if roi:
                roi.measurements.append(measurement)
            else:
                self.measurements.append(measurement)
            cnt_id += 1

        return roi.measurements if roi else self.measurements

    # ===== 可视化方法 =====
    def get_visualization(self, roi: Optional[ROIRegion] = None) -> np.ndarray:
        """获取可视化结果"""
        if self.image is None:
            raise ValueError("请先加载图像")

        vis_image = self.image.copy()

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
                        vis_image[y_start:y_end, x_start:x_end][skeleton_mask] = [255, 0, 0]

                except Exception as e:
                    print(f"骨架显示错误 (CNT #{m.id}): {e}")

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

        # 绘制ROI边框
        for r in self.rois:
            cv2.rectangle(preview, (r.x, r.y), (r.x + r.width, r.y + r.height),
                          r.color, 2)
            cv2.putText(preview, r.name, (r.x + 5, r.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)

        return preview

    # ===== 统计方法 =====
    def get_statistics(self, roi: Optional[ROIRegion] = None) -> Dict[str, float]:
        """获取测量结果的统计信息

        Args:
            roi (Optional[ROIRegion]): 指定ROI区域，若为None则使用全局测量结果

        Returns:
            Dict[str, float]: 包含 mean, median, std, min, max 等统计值
        """
        measurements = roi.measurements if roi else self.measurements

        if not measurements:
            return {}

        lengths = [m.length_um for m in measurements]

        length_bins = [0, 5, 15, 30, float('inf')]
        length_labels = ['<5μm', '5-15μm', '15-30μm', '>30μm']
        length_dist = {}
        for i, label in enumerate(length_labels):
            count = sum(1 for l in lengths
                        if length_bins[i] <= l < length_bins[i + 1])
            length_dist[label] = count

        return {
            'count': int(len(measurements)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
            'length_min': float(np.min(lengths)),
            'length_max': float(np.max(lengths)),
            'lengths': [float(l) for l in lengths],
            'length_distribution': {k: int(v) for k, v in length_dist.items()}
        }
