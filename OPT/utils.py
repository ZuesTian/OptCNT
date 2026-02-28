"""
工具模块 - 常量定义和工具类
"""
import cv2
import numpy as np

# ==================== 常量定义 ====================
# 比例尺检测
SCALE_BAR_BLUE_THRESHOLD = 120          # 蓝色通道最低亮度
SCALE_BAR_BLUE_SCORE_MIN = 60           # 蓝色得分最小值
SCALE_BAR_MIN_SPAN_PX = 8              # 蓝色条最小像素跨度
SCALE_BAR_BGR_DIST_MAX = 16000         # BGR 距离阈值
SCALE_BAR_DEFAULT_UM = 10.0            # 默认比例尺微米数
SCALE_BAR_ROI_X_RATIO = 0.4           # 比例尺搜索区域 x 起始比例
SCALE_BAR_ROI_Y_RATIO = 0.6           # 比例尺搜索区域 y 起始比例
SCALE_BAR_ASPECT_RATIO_MIN = 5        # 比例尺最小宽高比
SCALE_BAR_ASPECT_RATIO_STRICT = 8     # 严格宽高比（灰度检测）
SCALE_BAR_OCR_MATCH_THRESHOLD = 0.4   # OCR 模板匹配最低分
SCALE_BAR_VALUE_RANGE = (0.1, 1000)   # 比例尺数值合法范围

# 预处理
# PREPROCESS_MAX_DIM, PREPROCESS_FOREGROUND_TARGET removed (unused)

# 检测
# DETECT_MIN_CONTOUR_AREA removed (unused)
SKELETON_ANGLE_THRESHOLDS = [120, 135, 150, 165]  # 骨架路径追踪角度阈值
SKELETON_WALK_ANGLE_DEG = 150         # 骨架长度计算角度阈值

# GUI
DEBOUNCE_DELAY_MS = 250               # 滑块防抖延迟(毫秒)
# ZOOM_FACTOR, ZOOM_MIN, ZOOM_MAX, OVERLAY_ALPHA removed (unused)
