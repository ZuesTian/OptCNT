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
ANALYSIS_BLACKHAT_KERNEL = 31         # 分析图黑帽背景校正核大小
CALIBRATED_BLUR_KERNEL = 9            # DATA 样本校准得到的标准模糊核
CALIBRATED_ADAPTIVE_BLOCK = 11        # DATA 样本校准得到的标准块大小
CALIBRATED_ADAPTIVE_C = 3             # DATA 样本校准得到的标准 C
CNT_BRIDGE_STRENGTH_DEFAULT = 0       # 默认关闭，避免改变既有结果基线
CNT_BRIDGE_STRENGTH_MAX = 10          # 桥接滑块上限

# 检测
# DETECT_MIN_CONTOUR_AREA removed (unused)
SKELETON_ANGLE_THRESHOLDS = [160, 150, 140, 130, 120]  # 骨架路径追踪角度阈值（放宽以支持弯曲CNT）
SKELETON_WALK_ANGLE_DEG = 150         # 骨架长度计算角度阈值
CNT_MERGE_DISTANCE_DEFAULT_PX = 0     # 近邻合并默认关闭，避免误合并
CNT_MERGE_DISTANCE_MAX_PX = 20        # 近邻合并滑块上限
CNT_MERGE_MAX_ANGLE_DIFF_DEG = 28.0   # 两段CNT方向差超过该值则不合并
CNT_MERGE_MAX_ALIGNMENT_DEG = 32.0    # 连接线与CNT主方向夹角上限

# GUI
DEBOUNCE_DELAY_MS = 380               # 滑块防抖延迟(毫秒)
# ZOOM_FACTOR, ZOOM_MIN, ZOOM_MAX, OVERLAY_ALPHA removed (unused)
