"""
GUI主控制器模块 - 负责协调各个面板和核心分析功能
"""
import json
import logging
import csv
import inspect
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import repeat
from pathlib import Path
from multiprocessing import cpu_count
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional, List, Tuple
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..core.models import ROIRegion, CNTMeasurement
from ..core.analyzer_core import CNTAnalyzer
from ..core.stats_compat import mannwhitneyu, ttest_ind
from ..core.utils import (
    DEBOUNCE_DELAY_MS,
    CHART_REBUILD_DRAW_LIMIT,
    SCALE_BAR_DEFAULT_UM,
    CNT_BRIDGE_STRENGTH_DEFAULT,
    CNT_MERGE_DISTANCE_DEFAULT_PX,
    CALIBRATED_BLUR_KERNEL,
    CALIBRATED_ADAPTIVE_BLOCK,
    CALIBRATED_ADAPTIVE_C,
    get_length_histogram_bins,
)
from .widgets import SortableTreeview
from .panels import ControlPanel, ImagePanel, ResultPanel, AdvancedAnalysisPanel, ComparisonAnalysisPanel
from .gui_styles import MODERN_COLORS, apply_modern_style

logger = logging.getLogger(__name__)

GUI_EXPECTED_ANALYSIS_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    tk.TclError,
    cv2.error,
)
GUI_EXPECTED_RENDER_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    RuntimeError,
    tk.TclError,
)
GUI_EXPECTED_DATA_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    cv2.error,
)


class CNTAnalyzerGUI:
    MODERN_COLORS = MODERN_COLORS
    """CNT分析器图形界面主控制器"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CNT图像分析系统 - 现代化骨架预览版")
        self._configure_opencv_runtime()

        # 应用Modern样式
        apply_modern_style(self.root, MODERN_COLORS)

        # 核心分析器
        self.analyzer = CNTAnalyzer()

        # 状态变量
        self.current_image = None
        self.photo = None
        self.current_roi: Optional[ROIRegion] = None
        self.roi_counter = 0
        self.zoom_level = 1.0
        self._preprocess_job = None
        self._preprocess_executor = self._create_preprocess_executor()
        self._preprocess_future = None
        self._preprocess_snapshot = None
        self._preprocess_token = 0
        self._preprocess_result_exact = False
        self._layout_job = None
        self._layout_retry_count = 0
        self._comparison_layout_job = None
        self.main_paned: Optional[tk.PanedWindow] = None
        self.current_image_path: Optional[str] = None
        self._analysis_cache = OrderedDict()
        self._analysis_cache_lock = threading.Lock()
        self._analysis_cache_limit = 24
        self._last_auto_suggest_result = None
        self._last_root_size: Optional[Tuple[int, int]] = None
        self._pending_scale_selection = None
        self._preprocess_preview_fast = False
        self._single_detect_executor = self._create_single_detect_executor()
        self._single_detect_future = None
        self._single_detect_snapshot = None
        self._single_detect_token = 0
        self._compare_executor = self._create_compare_executor()
        self._compare_future = None
        self._compare_snapshot = None
        self._compare_token = 0
        self.open_image_button: Optional[ttk.Button] = None
        self.paste_image_button: Optional[ttk.Button] = None
        self.save_results_button: Optional[ttk.Button] = None
        self.export_report_button: Optional[ttk.Button] = None
        self.compare_analysis_button: Optional[ttk.Button] = None
        
        # 图表缓存
        self._charts = {
            'score': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'histogram': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'pie': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'cluster': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'heatmap': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'comparison': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
        }

        # Tkinter变量
        self._init_variables()

        # 面板引用（在 _setup_ui 中初始化）
        self.control_panel: ControlPanel = None  # type: ignore[assignment]
        self.image_panel: ImagePanel = None  # type: ignore[assignment]
        self.result_panel: ResultPanel = None  # type: ignore[assignment]
        self.analysis_panel: AdvancedAnalysisPanel = None  # type: ignore[assignment]
        self.comparison_panel: ComparisonAnalysisPanel = None  # type: ignore[assignment]
        self._center_tabs = {}

        # 设置UI
        self._setup_ui()
        self._bind_detection_setting_traces()
        self._refresh_interaction_state()

        # 快捷键：从剪贴板粘贴图像
        self.root.bind_all("<Control-v>", self._paste_image_from_clipboard)
        self.root.bind_all("<Control-V>", self._paste_image_from_clipboard)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _configure_opencv_runtime() -> None:
        """Cap OpenCV's internal threading to avoid oversubscription during batch comparison."""
        try:
            cpu_count = os.cpu_count() or 1
            current_threads = int(cv2.getNumThreads())
            target_threads = 1 if cpu_count >= 8 else max(1, min(2, cpu_count))
            if current_threads <= 0 or current_threads > max(8, cpu_count):
                cv2.setNumThreads(target_threads)
            elif current_threads > target_threads:
                cv2.setNumThreads(target_threads)
            cv2.ocl.setUseOpenCL(False)
        except (AttributeError, TypeError, ValueError, cv2.error):
            logger.debug("Unable to adjust OpenCV runtime threading; using library defaults.")

    def _init_variables(self):
        """初始化Tkinter变量"""
        self.blur_kernel_var = tk.IntVar(value=9)
        self.adaptive_block_var = tk.IntVar(value=11)
        self.adaptive_c_var = tk.IntVar(value=3)
        self.bridge_strength_var = tk.IntVar(value=CNT_BRIDGE_STRENGTH_DEFAULT)
        self.min_length_um_var = tk.DoubleVar(value=5.0)
        self.max_length_um_var = tk.DoubleVar(value=1000.0)
        self.min_slenderness_var = tk.DoubleVar(value=4.0)
        self.merge_distance_px_var = tk.IntVar(value=CNT_MERGE_DISTANCE_DEFAULT_PX)
        self.detect_profile_var = tk.StringVar(value="标准（推荐）")
        self.split_mode_var = tk.StringVar(value="不拆分")
        self.scale_pixels_var = tk.DoubleVar(value=0)
        self.scale_um_var = tk.DoubleVar(value=SCALE_BAR_DEFAULT_UM)
        self.live_preview_var = tk.BooleanVar(value=True)
        self.display_var = tk.StringVar(value="original")
        self._last_preprocess_signature = None

    def _setup_ui(self):
        """设置用户界面"""
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.85)
        min_width = min(1360, max(900, screen_width - 120))
        min_height = min(820, max(560, screen_height - 120))
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)

        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min_width, min_height)
        self._last_root_size = (window_width, window_height)

        # 创建顶部工具栏
        self._create_toolbar()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10) # 增加边距

        # 创建水平PanedWindow (带把手样式)
        style = ttk.Style()
        style.configure('Sash', sashthickness=8, sashrelief='flat')
        
        main_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashwidth=6, bg=self.MODERN_COLORS['bg_primary'], bd=0)
        main_paned.pack(fill=tk.BOTH, expand=True)
        self.main_paned = main_paned

        # 左侧面板 - 控制面板
        left_frame = ttk.Frame(main_paned)
        left_width = int(window_width * 0.22)
        left_frame.configure(width=left_width)
        main_paned.add(left_frame, minsize=260, width=left_width)
        self._setup_control_panel(left_frame)

        # 中间面板 - 图像显示
        center_frame = ttk.Frame(main_paned)
        center_width = int(window_width * 0.54)
        center_frame.configure(width=center_width)
        main_paned.add(center_frame, minsize=520, width=center_width)
        self._setup_center_panel(center_frame)

        # 右侧面板 - 结果面板
        right_frame = ttk.Frame(main_paned)
        right_width = int(window_width * 0.24)
        right_frame.configure(width=right_width)
        main_paned.add(right_frame, minsize=220, width=right_width)
        self._setup_result_panel(right_frame)

        # 根据窗口尺寸自动优化三栏分配：左控制/中图像/右结果
        self._last_root_size = None
        self._schedule_window_distribution(delay_ms=160)
        self.root.bind("<Configure>", self._on_root_resize, add="+")

    def _on_root_resize(self, event):
        """窗口尺寸变化时防抖重排三栏布局"""
        if event.widget is not self.root or self.main_paned is None:
            return
        current_size = (int(event.width), int(event.height))
        if self._last_root_size == current_size:
            return
        self._last_root_size = current_size
        self._schedule_window_distribution(delay_ms=120)
        self._schedule_comparison_layout_refresh(delay_ms=180)

    def _schedule_window_distribution(self, delay_ms: int = 120) -> None:
        """防抖调度三栏布局优化。"""
        if self._layout_job is not None:
            self.root.after_cancel(self._layout_job)
        self._layout_job = self.root.after(delay_ms, self._optimize_window_distribution)

    def _get_active_center_tab_key(self) -> str:
        """返回当前选中的中间标签页键名。"""
        if not hasattr(self, 'center_notebook'):
            return 'image'

        selected_tab = str(self.center_notebook.select())
        for key, tab in self._center_tabs.items():
            if str(tab) == selected_tab:
                return key
        return 'image'

    def _select_center_tab(self, tab_key: str) -> bool:
        """切换到指定的中间标签页；标签不存在时安全返回。"""
        if not hasattr(self, 'center_notebook'):
            return False

        target_tab = self._center_tabs.get(tab_key)
        if target_tab is None:
            return False

        current_tab = str(self.center_notebook.select())
        if current_tab != str(target_tab):
            self.center_notebook.select(target_tab)
            self.center_notebook.update_idletasks()

        return True

    def _get_pane_layout_profile(self, tab_key: Optional[str] = None) -> dict:
        """返回统一的三栏布局配置，不再根据标签页变化。"""
        active_tab = tab_key or self._get_active_center_tab_key()
        if active_tab == 'comparison':
            return {
                'left_ratio': 0.18,
                'right_ratio': 0.18,
                'left_floor': 240,
                'right_floor': 200,
                'center_target': 720,
            }
        return {
            'left_ratio': 0.20,
            'right_ratio': 0.20,
            'left_floor': 260,
            'right_floor': 220,
            'center_target': 620,
        }

    def _calculate_pane_widths(self, total_w: int, tab_key: Optional[str] = None) -> Tuple[int, int, int]:
        """使用固定比例计算左/中/右三栏宽度。"""
        profile = self._get_pane_layout_profile(tab_key)
        total_w = max(1, int(total_w))
        left_floor = int(profile['left_floor'])
        right_floor = int(profile['right_floor'])
        left_ratio = float(profile['left_ratio'])
        right_ratio = float(profile['right_ratio'])
        
        # 使用固定比例分配
        left_w = max(left_floor, int(total_w * left_ratio))
        right_w = max(right_floor, int(total_w * right_ratio))
        center_w = max(420, total_w - left_w - right_w)
        
        # 确保总和正确
        if left_w + center_w + right_w != total_w:
            right_w = max(right_floor, total_w - left_w - center_w)

        return left_w, center_w, right_w

    def _optimize_window_distribution(self):
        """使用固定比例优化窗口分布"""
        self._layout_job = None
        paned = self.main_paned
        if paned is None or not paned.winfo_exists() or len(paned.panes()) < 3:
            return

        total_w = max(1, paned.winfo_width())
        root_w = max(1, self.root.winfo_width())
        expected_paned_w = max(1, root_w - 24)
        if total_w < max(640, int(expected_paned_w * 0.75)) and self._layout_retry_count < 6:
            self._layout_retry_count += 1
            self._schedule_window_distribution(delay_ms=80)
            return

        self._layout_retry_count = 0
        profile = self._get_pane_layout_profile()
        panes = paned.panes()
        if len(panes) >= 3:
            try:
                paned.paneconfigure(panes[0], minsize=int(profile['left_floor']))
                paned.paneconfigure(panes[1], minsize=420)
                paned.paneconfigure(panes[2], minsize=int(profile['right_floor']))
            except tk.TclError:
                pass

        left_w, _, right_w = self._calculate_pane_widths(total_w)

        left_sash = left_w
        right_sash = max(left_sash + 120, total_w - right_w)
        right_sash = min(right_sash, total_w - 1)

        try:
            paned.sash_place(0, left_sash, 0)
            paned.sash_place(1, right_sash, 0)
        except tk.TclError:
            return

        self._schedule_comparison_layout_refresh(delay_ms=120)

    def _create_toolbar(self):
        """创建顶部工具栏"""
        toolbar = tk.Frame(self.root, relief='flat', borderwidth=0, 
                          bg=self.MODERN_COLORS['bg_secondary'])
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
        
        separator = ttk.Frame(self.root, height=1, style='TFrame')
        separator.pack(side=tk.TOP, fill=tk.X)
        separator_line = tk.Frame(separator, height=2, bg=self.MODERN_COLORS['accent_primary'])
        separator_line.pack(fill=tk.X)

        button_frame = tk.Frame(toolbar, bg=self.MODERN_COLORS['bg_secondary'])
        button_frame.pack(side=tk.LEFT, padx=10, pady=8)

        self.open_image_button = ttk.Button(
            button_frame,
            text="📂 打开图像",
            style='Accent.TButton',
            command=self._open_image,
        )
        self.open_image_button.pack(side=tk.LEFT, padx=2)
        self.paste_image_button = ttk.Button(
            button_frame,
            text="📋 粘贴图像",
            style='Accent.TButton',
            command=self._paste_image_from_clipboard,
        )
        self.paste_image_button.pack(side=tk.LEFT, padx=2)
        self.save_results_button = ttk.Button(
            button_frame,
            text="💾 保存结果",
            style='Success.TButton',
            command=self._save_results,
        )
        self.save_results_button.pack(side=tk.LEFT, padx=2)
        self.export_report_button = ttk.Button(
            button_frame,
            text="📊 导出报告",
            style='Warning.TButton',
            command=self._export_report,
        )
        self.export_report_button.pack(side=tk.LEFT, padx=2)
        self.compare_analysis_button = ttk.Button(
            button_frame,
            text="🔬 对比分析",
            style='Accent.TButton',
            command=self._open_compare_mode_dialog,
        )
        self.compare_analysis_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        
        status_frame = tk.Frame(toolbar, bg=self.MODERN_COLORS['bg_secondary'])
        status_frame.pack(side=tk.RIGHT, padx=15)
        
        self.status_indicator = tk.Canvas(status_frame, width=12, height=12, 
                                          bg=self.MODERN_COLORS['bg_secondary'],
                                          highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_status_indicator('idle')
        
        title_label = tk.Label(status_frame, text="CNT图像分析系统", 
                                font=('Segoe UI', 12, 'bold'),
                                bg=self.MODERN_COLORS['bg_secondary'],
                                fg=self.MODERN_COLORS['accent_primary'])
        title_label.pack(side=tk.LEFT)

    def _draw_status_indicator(self, state: str):
        """绘制状态指示器"""
        colors = {
            'idle': self.MODERN_COLORS['text_muted'],
            'ready': self.MODERN_COLORS['success'],
            'processing': self.MODERN_COLORS['warning'],
            'error': self.MODERN_COLORS['error']
        }
        color = colors.get(state, self.MODERN_COLORS['text_muted'])
        self.status_indicator.delete('all')
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline='')

    @staticmethod
    def _set_ttk_widget_enabled(widget: Optional[ttk.Widget], enabled: bool) -> None:
        """统一设置 ttk 控件状态。"""
        if widget is None:
            return
        if enabled:
            widget.state(['!disabled'])
        else:
            widget.state(['disabled'])

    def _get_active_measurements(self) -> List[CNTMeasurement]:
        """返回当前上下文下的测量结果。"""
        if self.current_roi is not None:
            return list(self.current_roi.measurements)
        return list(self.analyzer.measurements)

    def _is_single_detection_running(self) -> bool:
        """Whether a background single-image detection job is still running."""
        future = getattr(self, '_single_detect_future', None)
        return future is not None and not future.done()

    def _is_compare_analysis_running(self) -> bool:
        """Whether a background comparison-analysis job is still running."""
        future = getattr(self, '_compare_future', None)
        return future is not None and not future.done()

    def _is_preprocessing_running(self) -> bool:
        """Whether a background preprocess preview job is still running."""
        future = getattr(self, '_preprocess_future', None)
        return future is not None and not future.done()

    @staticmethod
    def _create_preprocess_executor() -> ThreadPoolExecutor:
        """Create the dedicated executor used for background preprocess previews."""
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnt-preprocess")

    def _reset_preprocess_executor(self) -> None:
        """Swap in a fresh preprocess executor so stale jobs cannot block newer previews."""
        executor = getattr(self, '_preprocess_executor', None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.debug("Unable to reset the preprocess executor cleanly.")
        self._preprocess_executor = self._create_preprocess_executor()

    def _discard_preprocess_state(self,
                                  *,
                                  include_completed: bool = False,
                                  notify: bool = False,
                                  image_reason: Optional[str] = None) -> bool:
        """Invalidate stale preprocess preview state after context changes."""
        preprocess_job = getattr(self, '_preprocess_job', None)
        if preprocess_job is not None and getattr(self, 'root', None) is not None:
            try:
                self.root.after_cancel(preprocess_job)
            except Exception:
                logger.debug("Unable to cancel the pending preprocess debounce callback cleanly.")
            self._preprocess_job = None

        future = getattr(self, '_preprocess_future', None)
        snapshot = getattr(self, '_preprocess_snapshot', None)
        if future is None and snapshot is None:
            return preprocess_job is not None

        is_running = future is not None and not future.done()
        if not include_completed and not is_running:
            return preprocess_job is not None

        if is_running:
            try:
                future.cancel()
            except Exception:
                logger.debug("Unable to cancel the in-flight preprocess future; it will finish in the background.")
            self._reset_preprocess_executor()

        self._preprocess_future = None
        self._preprocess_snapshot = None
        self._preprocess_token += 1

        if notify and image_reason and getattr(self, 'image_panel', None) is not None:
            self.image_panel.show_status(image_reason)
        return True

    @staticmethod
    def _create_single_detect_executor() -> ThreadPoolExecutor:
        """Create the dedicated executor used for single-image CNT detection."""
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnt-single")

    @staticmethod
    def _create_compare_executor() -> ThreadPoolExecutor:
        """Create the dedicated executor used for compare-analysis requests."""
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnt-compare")

    def _reset_single_detect_executor(self) -> None:
        """Swap in a fresh executor so a stale running task cannot block new image analysis."""
        executor = getattr(self, '_single_detect_executor', None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.debug("Unable to reset the single-image detection executor cleanly.")
        self._single_detect_executor = self._create_single_detect_executor()

    def _reset_compare_executor(self) -> None:
        """Swap in a fresh executor so a stale compare task cannot block the next request."""
        executor = getattr(self, '_compare_executor', None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.debug("Unable to reset the compare-analysis executor cleanly.")
        self._compare_executor = self._create_compare_executor()

    def _discard_single_detection_state(self,
                                        *,
                                        reason: Optional[str] = None,
                                        image_reason: Optional[str] = None,
                                        include_completed: bool = False,
                                        notify: bool = True) -> bool:
        """Invalidate stale single-image detection state after context changes."""
        future = getattr(self, '_single_detect_future', None)
        snapshot = getattr(self, '_single_detect_snapshot', None)
        if future is None and snapshot is None:
            return False

        is_running = future is not None and not future.done()
        if not include_completed and not is_running:
            return False

        if is_running:
            try:
                future.cancel()
            except Exception:
                logger.debug("Unable to cancel the in-flight detection future; it will finish in the background.")
            self._reset_single_detect_executor()

        self._single_detect_future = None
        self._single_detect_snapshot = None
        self._single_detect_token += 1
        self._set_single_detection_busy_state(False)

        if notify and reason:
            if getattr(self, 'control_panel', None) is not None:
                self.control_panel.update_analysis_status(reason, color=self.MODERN_COLORS['warning'])
            if getattr(self, 'image_panel', None) is not None and image_reason:
                self.image_panel.show_status(image_reason)
        return True

    def _abandon_single_detection_if_running(self,
                                             reason: str = "检测参数已更新，当前后台结果将忽略，可重新开始分析。") -> bool:
        """Drop the current single-image detection result when settings change mid-run."""
        return self._discard_single_detection_state(
            reason=reason,
            image_reason="检测参数已更新，可重新开始CNT检测",
            include_completed=False,
            notify=True,
        )

    def _handle_detection_setting_update(self) -> None:
        """Refresh the analysis state after the user tweaks detection-related settings."""
        if self._abandon_single_detection_if_running():
            return
        self._refresh_analysis_status_ui()

    def _set_single_detection_busy_state(self, busy: bool) -> None:
        """Toggle the single-detection button and related entry points while keeping the UI responsive."""
        if getattr(self, 'control_panel', None) is not None:
            self._set_ttk_widget_enabled(getattr(self.control_panel, 'detect_button', None), not busy)
        self._set_ttk_widget_enabled(self.compare_analysis_button, not busy)
        if not busy:
            self._refresh_interaction_state()

    def _set_compare_analysis_busy_state(self, busy: bool) -> None:
        """Toggle compare-analysis entry points while a background compare task is running."""
        if getattr(self, 'control_panel', None) is not None:
            self._set_ttk_widget_enabled(getattr(self.control_panel, 'detect_button', None), not busy)
        self._set_ttk_widget_enabled(self.compare_analysis_button, not busy)
        if not busy:
            self._refresh_interaction_state()

    def _discard_compare_analysis_state(self,
                                        *,
                                        include_completed: bool = False,
                                        notify: bool = False,
                                        reason: Optional[str] = None) -> bool:
        """Invalidate stale compare-analysis state and optionally cancel an in-flight request."""
        future = getattr(self, '_compare_future', None)
        snapshot = getattr(self, '_compare_snapshot', None)
        if future is None and snapshot is None:
            return False

        is_running = future is not None and not future.done()
        if not include_completed and not is_running:
            return False

        if is_running:
            try:
                future.cancel()
            except Exception:
                logger.debug("Unable to cancel the in-flight compare future; it will finish in the background.")
            self._reset_compare_executor()

        self._compare_future = None
        self._compare_snapshot = None
        self._compare_token += 1
        if getattr(self, 'comparison_panel', None) is not None:
            self.comparison_panel.hide_progress()
        self._set_compare_analysis_busy_state(False)

        if notify and reason and getattr(self, 'image_panel', None) is not None:
            self.image_panel.show_status(reason)
        return True

    def _refresh_interaction_state(self) -> None:
        """根据当前上下文启用或禁用关键交互入口。"""
        has_image = self.analyzer.image is not None
        has_rois = bool(self.analyzer.rois)
        has_measurements = bool(self._get_active_measurements())

        if self.control_panel is not None:
            self.control_panel.set_interaction_state(has_image=has_image, has_rois=has_rois)
        if self.image_panel is not None:
            self.image_panel.set_image_actions_enabled(has_image)

        self._set_ttk_widget_enabled(self.save_results_button, has_measurements)
        self._set_ttk_widget_enabled(self.export_report_button, has_measurements)
        self._set_ttk_widget_enabled(self.compare_analysis_button, True)
        if self._is_compare_analysis_running():
            self._set_compare_analysis_busy_state(True)
        if self._is_single_detection_running():
            self._set_single_detection_busy_state(True)

    def _sync_views(self,
                    *,
                    refresh_display: bool = True,
                    refresh_results: bool = True,
                    refresh_analysis: bool = True,
                    refresh_controls: bool = True,
                    refresh_comparison: bool = False,
                    clear_comparison: bool = False) -> None:
        """统一刷新主界面各视图，避免局部更新导致状态不一致。"""
        if refresh_display:
            if self.analyzer.image is not None:
                self._update_display()
            elif self.image_panel is not None:
                self.current_image = None
                self.photo = None
                self.image_panel.clear_canvas()
                self.image_panel.hide_status()

        if refresh_results:
            self._update_results()

        if refresh_analysis:
            self._update_advanced_analysis()

        if refresh_comparison:
            if clear_comparison:
                self._clear_comparison_analysis()
            elif self.comparison_panel is not None:
                self.comparison_panel.refresh_layout()

        if refresh_controls:
            self._refresh_interaction_state()

    def _clear_advanced_analysis(self, reset_scroll: bool = False) -> None:
        """清空高级分析图表并恢复占位态。"""
        if self.analysis_panel is None:
            return

        for key in ('score', 'histogram', 'pie', 'cluster', 'heatmap'):
            self._dispose_chart(key)
            self.analysis_panel.clear_chart_content(key)

        self.analysis_panel.refresh_layout()
        if reset_scroll:
            self.analysis_panel.scroll_to_top()

    def _clear_comparison_analysis(self, reset_scroll: bool = False) -> None:
        """清空对比分析内容并恢复占位态。"""
        if self.comparison_panel is None:
            return

        self._dispose_chart('comparison')
        self.comparison_panel.set_section_height('comparison_summary', 160)
        self.comparison_panel.set_section_height('comparison', 1120)
        self.comparison_panel.set_text_content(
            'comparison_summary',
            "尚未执行对比分析。使用顶部“对比分析”按钮后，结果会显示在这里。",
        )
        self.comparison_panel.hide_progress()
        self.comparison_panel.clear_chart_content('comparison')
        self.comparison_panel.refresh_layout()
        if reset_scroll:
            self.comparison_panel.scroll_to_top()

    def _setup_control_panel(self, parent):
        """设置控制面板"""
        callbacks = {
            'open_image': self._open_image,
            'save_results': self._save_results,
            'export_report': self._export_report,
            'select_scale': self._select_scale_on_image,
            'apply_scale': self._apply_scale,
            'select_roi': self._select_roi,
            'on_select_roi': self._on_select_roi,
            'remove_roi': self._remove_selected_roi,
            'clear_rois': self._clear_all_rois,
            'on_live_preview_toggle': self._on_live_preview_toggle,
            'on_display_mode_change': self._on_display_mode_change,
            'on_blur_change': self._on_blur_change,
            'on_block_change': self._on_block_change,
            'on_c_change': self._on_c_change,
            'on_bridge_change': self._on_bridge_change,
            'auto_suggest_params': self._on_reapply_auto_suggest,
            'on_profile_change': self._on_profile_change,
            'on_split_mode_change': self._on_split_mode_change,
            'on_merge_distance_change': self._on_merge_distance_change,
            'detect_cnt': self._detect_cnt,
        }

        variables = {
            'scale_pixels': self.scale_pixels_var,
            'scale_um': self.scale_um_var,
            'live_preview': self.live_preview_var,
            'display_mode': self.display_var,
            'blur_kernel': self.blur_kernel_var,
            'adaptive_block': self.adaptive_block_var,
            'adaptive_c': self.adaptive_c_var,
            'bridge_strength': self.bridge_strength_var,
            'min_length': self.min_length_um_var,
            'max_length': self.max_length_um_var,
            'min_slenderness': self.min_slenderness_var,
            'merge_distance_px': self.merge_distance_px_var,
            'detect_profile': self.detect_profile_var,
            'split_mode': self.split_mode_var,
            'listbox_bg': self.MODERN_COLORS['input_bg'],
            'listbox_fg': self.MODERN_COLORS['text_primary'],
            'listbox_select_bg': self.MODERN_COLORS['selected_bg'],
            'listbox_select_fg': self.MODERN_COLORS['text_primary'],
        }

        self.control_panel = ControlPanel(parent, self.MODERN_COLORS, callbacks, variables)
        self.control_panel.pack(fill=tk.BOTH, expand=True)

    def _setup_center_panel(self, parent):
        """设置中间面板"""
        # 创建笔记本
        self.center_notebook = ttk.Notebook(parent)
        self.center_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=0) # 减少顶部边距
        self.center_notebook.bind("<<NotebookTabChanged>>", self._on_center_tab_changed, add="+")

        # 图像显示标签页
        image_tab = ttk.Frame(self.center_notebook, style='Card.TFrame')
        self.center_notebook.add(image_tab, text="图像显示")
        self._setup_image_panel(image_tab)
        self._center_tabs['image'] = image_tab

        # 高级分析标签页
        analysis_tab = ttk.Frame(self.center_notebook, style='Card.TFrame')
        self.center_notebook.add(analysis_tab, text="高级分析")
        self._setup_advanced_analysis_panel(analysis_tab)
        self._center_tabs['analysis'] = analysis_tab

        # 对比分析标签页
        comparison_tab = ttk.Frame(self.center_notebook, style='Card.TFrame')
        self.center_notebook.add(comparison_tab, text="对比分析")
        self._setup_comparison_panel(comparison_tab)
        self._center_tabs['comparison'] = comparison_tab

    def _setup_image_panel(self, parent):
        """设置图像显示面板"""
        callbacks = {
            'on_mousewheel': self._on_mousewheel,
            'fit_to_window': self._fit_image_to_window,
        }

        self.image_panel = ImagePanel(parent, self.MODERN_COLORS, callbacks)
        self.image_panel.pack(fill=tk.BOTH, expand=True)

    def _bind_detection_setting_traces(self) -> None:
        """Watch filter entries so parameter edits can immediately refresh detection state."""
        for variable in (
            self.min_length_um_var,
            self.max_length_um_var,
            self.min_slenderness_var,
        ):
            try:
                variable.trace_add('write', self._on_detection_filter_var_change)
            except (AttributeError, tk.TclError):
                logger.debug("Unable to bind detection filter trace for %s", variable)

    def _on_detection_filter_var_change(self, *_args) -> None:
        """React to direct edits in the analysis filter entry fields."""
        self._handle_detection_setting_update()

    def _setup_result_panel(self, parent):
        """设置结果面板"""
        callbacks = {
            'on_select_cnt': self._on_select_cnt,
        }

        variables = {
            'text_bg': self.MODERN_COLORS['input_bg'],
            'text_fg': self.MODERN_COLORS['text_primary'],
        }

        self.result_panel = ResultPanel(parent, self.MODERN_COLORS, callbacks, variables)
        self.result_panel.pack(fill=tk.BOTH, expand=True)

    def _setup_advanced_analysis_panel(self, parent):
        """设置高级分析面板"""
        self.analysis_panel = AdvancedAnalysisPanel(parent, self.MODERN_COLORS)
        self.analysis_panel.pack(fill=tk.BOTH, expand=True)

    def _setup_comparison_panel(self, parent):
        """设置对比分析面板"""
        self.comparison_panel = ComparisonAnalysisPanel(parent, self.MODERN_COLORS)
        self.comparison_panel.pack(fill=tk.BOTH, expand=True)

    def _on_center_tab_changed(self, event) -> None:
        """标签页切换时仅刷新对比分析图表，不再重新计算布局。"""
        if event.widget is self.center_notebook:
            self._schedule_comparison_layout_refresh(delay_ms=60)

    def _refresh_scale_status_ui(self):
        """刷新比例尺显示与状态文案"""
        status = self.analyzer.get_scale_status()
        pixels = status.get('pixels')
        micrometers = status.get('micrometers')
        um_per_pixel = status.get('um_per_pixel')
        ocr_um = status.get('ocr_micrometers')
        exclusion_state = "已排除比例尺区域" if status.get('exclusion_enabled') else "未排除比例尺区域"

        if pixels and micrometers and um_per_pixel:
            scale_text = f"当前比例尺: {pixels:.1f}px = {micrometers:.1f}μm ({um_per_pixel:.4f}μm/pixel)"
        else:
            scale_text = f"当前比例尺: 默认 {SCALE_BAR_DEFAULT_UM:g}μm（低置信度，待确认）"
        self.control_panel.update_scale_label(scale_text)

        source = status.get('source')
        if source == 'auto_detected':
            ocr_text = f"；OCR={ocr_um:g}μm（仅参考）" if ocr_um is not None else ""
            color = self.MODERN_COLORS['success']
            text = f"比例尺状态: 已自动应用 {SCALE_BAR_DEFAULT_UM:g}μm 标准比例尺，{exclusion_state}{ocr_text}"
        elif source == 'manual':
            color = self.MODERN_COLORS['info']
            text = f"比例尺状态: 已手动应用比例尺，{exclusion_state}"
        elif source == 'fallback_default':
            color = self.MODERN_COLORS['warning']
            text = f"比例尺状态: 未检测到比例尺，当前使用低置信度默认比例；建议手动确认。{exclusion_state}"
        else:
            color = self.MODERN_COLORS['text_secondary']
            text = "比例尺状态: 待检测"
        self.control_panel.update_scale_status(text, color=color)

    def _refresh_analysis_status_ui(self):
        """刷新识别输入状态文案"""
        if self.analyzer.image is None:
            self.control_panel.update_analysis_status("检测输入状态: 待加载图像", color=self.MODERN_COLORS['text_secondary'])
            return

        filter_settings = self._get_detection_filter_settings(strict=False)
        if not filter_settings.get('valid', False):
            self.control_panel.update_analysis_status(
                f"检测输入状态: 参数待修正，{filter_settings.get('message', '请检查检测参数')}",
                color=self.MODERN_COLORS['warning'],
            )
            return

        scale_status = self.analyzer.get_scale_status()
        exclusion_text = "已排除比例尺区域" if scale_status.get('exclusion_enabled') else "未排除比例尺区域"
        confidence_text = "低置信度比例尺" if scale_status.get('confidence') == 'low' else "比例尺已确认"
        text = (
            f"检测输入状态: 统一使用分析图；{exclusion_text}；"
            f"策略={self.detect_profile_var.get()}；拆分={self.split_mode_var.get()}；"
            f"桥接={self.bridge_strength_var.get()}；合并={self.merge_distance_px_var.get()}px；{confidence_text}"
        )
        auto_suggest_text = self._get_auto_suggest_status_text()
        if auto_suggest_text:
            text = f"{text}\n自动推荐: {auto_suggest_text}"
        color = self.MODERN_COLORS['warning'] if scale_status.get('confidence') == 'low' else self.MODERN_COLORS['text_secondary']
        self.control_panel.update_analysis_status(text, color=color)

    def _get_auto_suggest_status_text(self) -> str:
        """返回当前自动推荐参数的简要说明。"""
        info = self._last_auto_suggest_result
        if not info:
            return ""

        suggested = (
            int(info.get('blur_kernel', -1)),
            int(info.get('adaptive_block', -1)),
            int(info.get('adaptive_c', -1)),
        )
        current = (
            int(self.blur_kernel_var.get()),
            int(self.adaptive_block_var.get()),
            int(self.adaptive_c_var.get()),
        )
        if current != suggested:
            return f"上次推荐 {suggested[0]}/{suggested[1]}/{suggested[2]}；当前参数已手动调整。"
        return str(info.get('reason_summary', '')).strip()

    def _on_close(self) -> None:
        """Release background resources before closing the main window."""
        preprocess_executor = getattr(self, '_preprocess_executor', None)
        if preprocess_executor is not None:
            preprocess_executor.shutdown(wait=False, cancel_futures=True)
            self._preprocess_executor = None
        executor = getattr(self, '_single_detect_executor', None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._single_detect_executor = None
        compare_executor = getattr(self, '_compare_executor', None)
        if compare_executor is not None:
            compare_executor.shutdown(wait=False, cancel_futures=True)
            self._compare_executor = None
        self.root.destroy()

    # ===== 文件操作 =====
    def _load_image_common(self):
        """加载图像后的通用流程"""
        self._discard_preprocess_state(include_completed=True, notify=False)
        self._discard_single_detection_state(include_completed=True, notify=False)
        self._discard_compare_analysis_state(include_completed=True, notify=False)
        self._reset_display()

        self.scale_um_var.set(SCALE_BAR_DEFAULT_UM)
        scale_result = self.analyzer.apply_detected_scale(default_micrometers=SCALE_BAR_DEFAULT_UM)
        scale_info = scale_result.get('scale_info')
        if scale_result.get('applied') and scale_info:
            self.scale_pixels_var.set(float(scale_info['pixels']))
            self.image_panel.show_status(f"已自动应用比例尺: {scale_info['pixels']:.1f}px = {SCALE_BAR_DEFAULT_UM:g}μm")
        else:
            self.scale_pixels_var.set(0)
            self.image_panel.show_status("未检测到比例尺；当前使用低置信度默认比例，建议手动确认")

        self._refresh_scale_status_ui()

        # 自适应推荐预处理参数
        self._auto_suggest_params()
        self._refresh_analysis_status_ui()

        # 加载图像后，若实时预览开启则自动触发骨架预览
        if self.live_preview_var.get():
            self.display_var.set("skeleton_preview")
            self._schedule_preprocessing()

    def _open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            self._draw_status_indicator('processing')
            self.analyzer.load_image(file_path)
            self.current_image_path = file_path
            self._load_image_common()
            self._draw_status_indicator('ready')
        except (IOError, ValueError, cv2.error) as e:
            self._draw_status_indicator('error')
            messagebox.showerror("错误", f"无法加载图像: {e}")
        except Exception as e:
            self._draw_status_indicator('error')
            logger.exception("加载图像时发生未预期的错误")
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

    def _paste_image_from_clipboard(self, event=None):
        """从剪贴板粘贴图像（支持图像对象与文件路径）"""
        try:
            clip = ImageGrab.grabclipboard()
            if clip is None:
                messagebox.showwarning("提示", "剪贴板中没有可用的图像或图像文件路径")
                return "break"

            self._draw_status_indicator('processing')

            if isinstance(clip, Image.Image):
                pil_img = clip.convert("RGB")
                image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                self.analyzer.set_image(image_bgr)
                self.current_image_path = None
            elif isinstance(clip, list):
                image_file = None
                valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
                for p in clip:
                    if isinstance(p, str) and p.lower().endswith(valid_ext):
                        image_file = p
                        break

                if image_file is None:
                    self._draw_status_indicator('error')
                    messagebox.showwarning("提示", "剪贴板文件列表中未找到可识别的图像文件")
                    return "break"

                self.analyzer.load_image(image_file)
                self.current_image_path = image_file
            else:
                self._draw_status_indicator('error')
                messagebox.showwarning("提示", "剪贴板内容不是图像或图像文件")
                return "break"

            self._load_image_common()
            self._draw_status_indicator('ready')
        except (OSError, ValueError, cv2.error, tk.TclError) as e:
            self._draw_status_indicator('error')
            logger.exception("粘贴图像失败")
            messagebox.showerror("错误", f"粘贴图像失败: {e}")
        except Exception as e:
            self._draw_status_indicator('error')
            logger.exception("粘贴图像时发生未预期的错误")
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

        return "break"

    def _reset_display(self):
        """重置显示"""
        self.zoom_level = 1.0
        self._pending_scale_selection = None
        self._preprocess_preview_fast = False
        self._preprocess_result_exact = False
        if self.image_panel is not None:
            self.image_panel.set_zoom_level(self.zoom_level)
        self.current_roi = None
        self.roi_counter = 0
        self.analyzer.clear_rois()
        self.analyzer.clear_measurements()
        self.analyzer.binary_image = None
        self.analyzer.processed_image = None
        self.analyzer.skeleton_image = None
        self.analyzer.skeleton_overlay = None
        self._last_preprocess_signature = None
        self.control_panel.clear_roi_list()
        self._sync_views()
        self._refresh_scale_status_ui()
        self._refresh_analysis_status_ui()

    # ===== 比例尺操作 =====
    def _select_scale_on_image(self):
        """在图像上选择比例尺"""
        if self.analyzer.image is None:
            self.control_panel.update_scale_status(
                "比例尺状态: 请先加载图像后再选择比例尺",
                color=self.MODERN_COLORS['warning'],
            )
            self.image_panel.show_status("请先加载图像后再选择比例尺")
            return

        def on_scale_selected(length):
            selection = length if isinstance(length, dict) else None
            if selection is not None:
                zoom = self.zoom_level if self.zoom_level > 0 else 1.0
                real_length = float(selection.get('length', 0.0)) / zoom
                start = selection.get('start') or (0.0, 0.0)
                end = selection.get('end') or (0.0, 0.0)
                self._pending_scale_selection = {
                    'start': (float(start[0]) / zoom, float(start[1]) / zoom),
                    'end': (float(end[0]) / zoom, float(end[1]) / zoom),
                }
            else:
                # 兼容旧回调格式
                real_length = length / self.zoom_level
                self._pending_scale_selection = None
            self.scale_pixels_var.set(real_length)
            self.control_panel.update_scale_status(
                f"比例尺状态: 已选中 {real_length:.1f}px，请输入对应微米数后应用",
                color=self.MODERN_COLORS['info'],
            )
            self.image_panel.show_status(
                f"已选择比例尺长度 {real_length:.1f}px，请输入对应微米数并点击“应用比例尺”"
            )

        self.image_panel.start_scale_selection(on_scale_selected)
        self.image_panel.show_status("请在图像上拖拽绘制比例尺线段")

    def _apply_scale(self):
        """应用比例尺设置"""
        try:
            pixels = self.scale_pixels_var.get()
            micrometers = self.scale_um_var.get()

            if pixels <= 0 or micrometers <= 0:
                messagebox.showerror("错误", "像素数和微米数必须大于0！")
                return

            # 修复2: 比例尺变更后，重算所有已有测量结果的长度和宽度
            old_scale = self.analyzer.scale_um_per_pixel
            selection_line = None
            if isinstance(self._pending_scale_selection, dict):
                start = self._pending_scale_selection.get('start')
                end = self._pending_scale_selection.get('end')
                if start is not None and end is not None:
                    selection_line = (start, end)
            self.analyzer.record_manual_scale(
                pixels,
                micrometers,
                source='manual',
                confidence='high',
                selection_line=selection_line,
            )
            new_scale = self.analyzer.scale_um_per_pixel
            self._last_preprocess_signature = None
            
            # 重算全局测量结果
            for m in self.analyzer.measurements:
                m.length_um = m.length_pixels * new_scale
                if m.width_mean_um is not None:
                    width_px = m.width_mean_um / old_scale if old_scale > 0 else 0
                    m.width_mean_um = width_px * new_scale
                if m.width_median_um is not None:
                    width_px = m.width_median_um / old_scale if old_scale > 0 else 0
                    m.width_median_um = width_px * new_scale
                if m.width_iqr_um is not None:
                    width_px = m.width_iqr_um / old_scale if old_scale > 0 else 0
                    m.width_iqr_um = width_px * new_scale
            
            # 重算所有ROI的测量结果
            for roi in self.analyzer.rois:
                for m in roi.measurements:
                    m.length_um = m.length_pixels * new_scale
                    if m.width_mean_um is not None:
                        width_px = m.width_mean_um / old_scale if old_scale > 0 else 0
                        m.width_mean_um = width_px * new_scale
                    if m.width_median_um is not None:
                        width_px = m.width_median_um / old_scale if old_scale > 0 else 0
                        m.width_median_um = width_px * new_scale
                    if m.width_iqr_um is not None:
                        width_px = m.width_iqr_um / old_scale if old_scale > 0 else 0
                        m.width_iqr_um = width_px * new_scale

            self._refresh_scale_status_ui()
            self._refresh_analysis_status_ui()
            if self._is_preprocess_mode():
                self._apply_preprocessing(force=True)
            self._sync_views()
            
            messagebox.showinfo("成功", "比例尺已应用，测量结果已更新！")

        except (TypeError, ValueError, OverflowError, tk.TclError, cv2.error) as e:
            logger.exception("应用比例尺失败")
            messagebox.showerror("错误", f"应用比例尺失败: {e}")
        except Exception as e:
            logger.exception("应用比例尺时发生未预期的错误")
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

    # ===== ROI操作 =====
    def _select_roi(self):
        """选择ROI"""
        if self.analyzer.image is None:
            self.control_panel.update_analysis_status(
                "检测输入状态: 请先加载图像后再选择 ROI",
                color=self.MODERN_COLORS['warning'],
            )
            self.image_panel.show_status("请先加载图像后再选择 ROI")
            return

        def on_roi_selected(coords):
            cx, cy, cw, ch = coords
            # 画布坐标 → 原图坐标（消除缩放影响）
            x = int(cx / self.zoom_level)
            y = int(cy / self.zoom_level)
            w = int(cw / self.zoom_level)
            h = int(ch / self.zoom_level)
            self.roi_counter += 1
            roi_name = f"ROI_{self.roi_counter}"

            roi = ROIRegion(
                name=roi_name,
                x=x, y=y,
                width=w, height=h,
                color=(0, 255, 255)
            )

            self.analyzer.add_roi(roi)
            self.control_panel.add_roi_to_list(roi_name)
            self.current_roi = roi

            self._last_preprocess_signature = None
            if self._is_preprocess_mode():
                self._apply_preprocessing(force=True)
            elif self.live_preview_var.get():
                self._schedule_preprocessing()
            self._sync_views()

        self.image_panel.start_roi_selection(on_roi_selected)
        self.image_panel.show_status("请在图像上拖拽绘制ROI矩形")

    def _on_select_roi(self, event):
        """选择ROI事件"""
        index = self.control_panel.get_selected_roi_index()
        if 0 <= index < len(self.analyzer.rois):
            self.current_roi = self.analyzer.rois[index]
            self._last_preprocess_signature = None
            if self._is_preprocess_mode():
                self._apply_preprocessing(force=True)
            elif self.live_preview_var.get():
                self._schedule_preprocessing()
            self._sync_views()

    def _remove_selected_roi(self):
        """删除选中的ROI"""
        index = self.control_panel.get_selected_roi_index()
        if index >= 0:
            self.analyzer.remove_roi(index)
            self.control_panel.clear_roi_list()
            for roi in self.analyzer.rois:
                self.control_panel.add_roi_to_list(roi.name)
            self.current_roi = None
            self._last_preprocess_signature = None
            if self._is_preprocess_mode():
                self._apply_preprocessing(force=True)
            self._sync_views()

    def _clear_all_rois(self):
        """清空所有ROI"""
        self.analyzer.clear_rois()
        self.control_panel.clear_roi_list()
        self.current_roi = None
        self._last_preprocess_signature = None
        if self._is_preprocess_mode():
            self._apply_preprocessing(force=True)
        self._sync_views()

    # ===== 自适应参数推荐 =====
    def _get_detection_profile_key(self) -> str:
        """将中文检测风格映射为核心算法配置键"""
        return {
            "严格（少误检）": "precision",
            "标准（推荐）": "balanced",
            "敏感（少漏检）": "recall",
        }.get(self.detect_profile_var.get(), "balanced")

    @staticmethod
    def _get_detection_profile_label(profile_key: str) -> str:
        """Map the analyzer profile key back to the UI label."""
        return {
            "precision": "严格（少误检）",
            "balanced": "标准（推荐）",
            "recall": "敏感（少漏检）",
        }.get(str(profile_key or "").lower(), "标准（推荐）")

    @staticmethod
    def _get_split_mode_label(split_mode: str) -> str:
        """Map the analyzer split mode key back to the UI label."""
        return {
            "off": "不拆分",
            "conservative": "标准拆分",
            "aggressive": "强力拆分",
        }.get(str(split_mode or "").lower(), str(split_mode or ""))

    def _auto_suggest_params(self):
        """根据图像特征自动推荐预处理参数"""
        try:
            roi = self._get_active_preprocess_roi()
            params = self.analyzer.suggest_preprocess_params(
                roi=roi,
                detection_profile=self._get_detection_profile_key(),
            )

            self.blur_kernel_var.set(params['blur_kernel'])
            self.adaptive_block_var.set(params['adaptive_block'])
            self.adaptive_c_var.set(params['adaptive_c'])

            self.control_panel.update_blur_label(str(params['blur_kernel']))
            self.control_panel.update_block_label(str(params['adaptive_block']))
            self.control_panel.update_c_label(str(params['adaptive_c']))

            self._last_auto_suggest_result = params
            self._last_preprocess_signature = None
            self._refresh_analysis_status_ui()
            return params
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as e:
            self._last_auto_suggest_result = None
            logger.debug(f"自适应参数推荐失败，使用默认值: {e}")
            return None
        except Exception as e:
            self._last_auto_suggest_result = None
            logger.exception("自动推荐参数时发生未预期的错误")
            return None

    def _on_reapply_auto_suggest(self):
        """手动触发一次自动参数推荐"""
        if self.analyzer.image is None:
            self.control_panel.update_analysis_status(
                "检测输入状态: 请先加载图像后再自动推荐参数",
                color=self.MODERN_COLORS['warning'],
            )
            self.image_panel.show_status("请先加载图像后再自动推荐参数")
            return
        params = self._auto_suggest_params()
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        elif not self.live_preview_var.get():
            self._discard_preprocess_state(include_completed=True, notify=False)
        elif not self.live_preview_var.get():
            self._discard_preprocess_state(include_completed=True, notify=False)
        elif self._is_preprocess_mode():
            self._apply_preprocessing(force=True)
        if params:
            self.image_panel.show_status(
                f"已重新推荐参数: {params['blur_kernel']}/{params['adaptive_block']}/{params['adaptive_c']}"
            )
        else:
            self.image_panel.show_status("自动推荐失败，已保留当前参数")
        self._handle_detection_setting_update()

    def _on_profile_change(self, event=None):
        """识别策略变化时刷新推荐参数和状态"""
        if self.analyzer.image is None:
            return
        self._auto_suggest_params()
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._handle_detection_setting_update()

    def _on_split_mode_change(self, event=None):
        """粘连拆分模式变化时刷新状态"""
        self._handle_detection_setting_update()

    # ===== 预处理参数 =====
    def _is_preprocess_mode(self) -> bool:
        """当前显示模式是否需要预处理结果"""
        return self.display_var.get() in ("binary", "skeleton_preview")

    def _should_use_fast_preprocess_preview(self, force: bool = False) -> bool:
        """Use a lightweight preview path during live binary tuning."""
        return (
            not force
            and self.live_preview_var.get()
            and self.display_var.get() == "binary"
        )

    def _should_limit_bridge_for_preview(self, force: bool = False) -> bool:
        """Use a smaller bridge kernel while the UI is in live preview mode."""
        return (
            not force
            and self.live_preview_var.get()
            and self._is_preprocess_mode()
        )

    def _get_effective_bridge_strength(self, force: bool = False) -> int:
        """Clamp bridge strength during live previews to keep interaction responsive."""
        bridge_strength = int(self.bridge_strength_var.get())
        if self._should_limit_bridge_for_preview(force=force):
            return min(bridge_strength, 5)
        return bridge_strength

    def _get_active_preprocess_roi(self) -> Optional[ROIRegion]:
        """获取当前预处理使用的ROI"""
        roi_to_use = self.current_roi
        if roi_to_use is None and self.analyzer.rois:
            roi_to_use = self.analyzer.rois[0]
        return roi_to_use

    def _get_preprocess_signature(self) -> tuple:
        """构建用于判断缓存有效性的预处理签名"""
        roi = self._get_active_preprocess_roi()
        roi_signature = None if roi is None else (roi.name, roi.x, roi.y, roi.width, roi.height)
        scale_exclusion_signature = tuple(self.analyzer.scale_exclusion_rect) if self.analyzer.scale_exclusion_rect is not None else None
        return (
            int(self.blur_kernel_var.get()),
            int(self.adaptive_block_var.get()),
            int(self.adaptive_c_var.get()),
            int(self.bridge_strength_var.get()),
            True,  # threshold_invert
            roi_signature,
            scale_exclusion_signature,
        )

    def _needs_preprocessing(self) -> bool:
        """判断当前参数/ROI是否需要重新预处理"""
        require_skeleton = self.display_var.get() == "skeleton_preview"
        require_exact = require_skeleton
        return not self._has_usable_preprocess_result(
            require_exact=require_exact,
            require_skeleton=require_skeleton,
        )

    def _has_usable_preprocess_result(self,
                                      *,
                                      require_exact: bool = False,
                                      require_skeleton: bool = False) -> bool:
        """Whether the live analyzer already has a preprocess result we can trust."""
        if self.analyzer.binary_image is None:
            return False
        if self._get_preprocess_signature() != self._last_preprocess_signature:
            return False
        if require_exact and not getattr(self, '_preprocess_result_exact', False):
            return False
        if require_skeleton and getattr(self.analyzer, "skeleton_image", None) is None:
            return False
        return True

    def _on_live_preview_toggle(self):
        """实时预览开关切换 - 控制滑块拖动时是否自动刷新"""
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()

    def _on_display_mode_change(self):
        """显示模式切换"""
        if self.analyzer.image is None:
            return
        mode = self.display_var.get()
        if mode in ("binary", "skeleton_preview"):
            if self._needs_preprocessing():
                self._apply_preprocessing(force=True)
            else:
                self._update_display()
        else:
            self._update_display()

    def _schedule_preprocessing(self):
        """调度预处理（带防抖）"""
        if self.analyzer.image is None:
            return
        if not self.live_preview_var.get():
            return
        if not self._is_preprocess_mode():
            return
        if self._preprocess_job is not None:
            self.root.after_cancel(self._preprocess_job)
        self._preprocess_job = self.root.after(DEBOUNCE_DELAY_MS, self._apply_preprocessing)

    def _apply_preprocessing(self, force: bool = False):
        """应用预处理 - threshold_invert 统一为 True"""
        try:
            self._preprocess_job = None
            blur_kernel = self.blur_kernel_var.get()
            adaptive_block = self.adaptive_block_var.get()
            adaptive_c = self.adaptive_c_var.get()
            roi_to_use = self._get_active_preprocess_roi()
            signature = self._get_preprocess_signature()
            fast_preview = self._should_use_fast_preprocess_preview(force=force)
            effective_bridge_strength = self._get_effective_bridge_strength(force=force)
            result_exact = (not fast_preview) and effective_bridge_strength == int(self.bridge_strength_var.get())

            require_skeleton = self.display_var.get() == "skeleton_preview" and not fast_preview
            if (
                not force
                and self._has_usable_preprocess_result(
                    require_exact=False,
                    require_skeleton=require_skeleton,
                )
            ):
                self._update_display()
                return

            pending_snapshot = getattr(self, '_preprocess_snapshot', None)
            if (
                not force
                and self._is_preprocessing_running()
                and pending_snapshot is not None
                and pending_snapshot.get('preprocess_signature') == signature
                and pending_snapshot.get('active_roi_signature') == self._roi_signature(roi_to_use)
                and pending_snapshot.get('fast_preview') == fast_preview
                and pending_snapshot.get('effective_bridge_strength') == effective_bridge_strength
            ):
                return

            preprocess_settings = {
                'blur_kernel': int(blur_kernel),
                'adaptive_block': int(adaptive_block),
                'adaptive_c': int(adaptive_c),
                'bridge_strength': int(effective_bridge_strength),
                'threshold_invert': True,
                'generate_skeleton': not fast_preview,
            }
            snapshot = {
                'image_id': id(self.analyzer.image),
                'preprocess_signature': signature,
                'active_roi_signature': self._roi_signature(roi_to_use),
                'fast_preview': fast_preview,
                'result_exact': result_exact,
                'effective_bridge_strength': int(effective_bridge_strength),
            }

            if force or getattr(self, 'root', None) is None:
                self._discard_preprocess_state(include_completed=True, notify=False)
                self.analyzer.preprocess(roi=roi_to_use, **preprocess_settings)
                self._preprocess_preview_fast = fast_preview
                self._preprocess_result_exact = result_exact
                self._last_preprocess_signature = signature
                self._update_display()
                return

            self._discard_preprocess_state(include_completed=True, notify=False)
            analyzer_snapshot, roi_snapshot = self._clone_analyzer_for_preprocess()
            self._preprocess_token += 1
            task_token = self._preprocess_token
            self._preprocess_snapshot = snapshot
            self._preprocess_future = self._preprocess_executor.submit(
                self._run_preprocess_task,
                analyzer_snapshot,
                preprocess_settings,
                roi_snapshot,
            )
            if getattr(self, 'image_panel', None) is not None:
                self.image_panel.show_status("预处理预览更新中...")
            self.root.after(40, self._poll_preprocess_result, task_token, snapshot)
        except (TypeError, ValueError, tk.TclError, cv2.error) as e:
            logger.exception(f"预处理错误: {e}")
        except Exception as e:
            logger.exception(f"预处理时发生未预期的错误: {e}")

    def _on_blur_change(self, value):
        """高斯模糊核大小变化"""
        val = int(float(value))
        if val % 2 == 0:
            val += 1
        if self.blur_kernel_var.get() == val:
            return
        self.blur_kernel_var.set(val)
        self.control_panel.update_blur_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._handle_detection_setting_update()

    def _on_block_change(self, value):
        """自适应块大小变化"""
        val = int(float(value))
        if val % 2 == 0:
            val += 1
        if val < 3:
            val = 3
        if self.adaptive_block_var.get() == val:
            return
        self.adaptive_block_var.set(val)
        self.control_panel.update_block_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._handle_detection_setting_update()

    def _on_c_change(self, value):
        """自适应常数C变化"""
        val = int(float(value))
        self.adaptive_c_var.set(val)
        self.control_panel.update_c_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._handle_detection_setting_update()

    def _on_bridge_change(self, value):
        """桥接强度变化"""
        val = int(float(value))
        self.bridge_strength_var.set(val)
        self.control_panel.update_bridge_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._handle_detection_setting_update()

    def _on_merge_distance_change(self, value):
        """近邻合并距离变化"""
        val = int(float(value))
        self.merge_distance_px_var.set(val)
        self.control_panel.update_merge_distance_label(str(val))
        self._handle_detection_setting_update()

    # ===== CNT检测 =====
    @staticmethod
    def _roi_signature(roi: Optional[ROIRegion]) -> Optional[tuple]:
        """Build a stable identity tuple for the active ROI."""
        if roi is None:
            return None
        return (str(roi.name), int(roi.x), int(roi.y), int(roi.width), int(roi.height))

    @staticmethod
    def _clone_roi_region(roi: ROIRegion) -> ROIRegion:
        """Create a measurement-free ROI copy for background analysis."""
        return ROIRegion(
            name=str(roi.name),
            x=int(roi.x),
            y=int(roi.y),
            width=int(roi.width),
            height=int(roi.height),
            color=tuple(roi.color),
            measurements=[],
        )

    def _clone_analyzer_snapshot(self, *, include_preprocess_outputs: bool) -> CNTAnalyzer:
        """Snapshot the live analyzer state for background work."""
        snapshot = CNTAnalyzer()
        attrs = [
            'original_image',
            'analysis_image',
            'analysis_gray_image',
            'image',
            'scale_exclusion_mask',
        ]
        if include_preprocess_outputs:
            attrs.extend([
                'processed_image',
                'binary_image',
                'skeleton_image',
                'skeleton_overlay',
            ])

        for attr in attrs:
            value = getattr(self.analyzer, attr, None)
            setattr(snapshot, attr, value.copy() if isinstance(value, np.ndarray) else value)

        snapshot.scale_um_per_pixel = float(self.analyzer.scale_um_per_pixel)
        snapshot.scale_bar_info = dict(self.analyzer.scale_bar_info) if self.analyzer.scale_bar_info else None
        snapshot.scale_exclusion_rect = tuple(self.analyzer.scale_exclusion_rect) if self.analyzer.scale_exclusion_rect else None
        snapshot.scale_status = dict(self.analyzer.scale_status)
        snapshot.auto_enhance_enabled = bool(self.analyzer.auto_enhance_enabled)
        snapshot.rois = [self._clone_roi_region(roi) for roi in self.analyzer.rois]
        return snapshot

    def _clone_analyzer_for_preprocess(self) -> Tuple[CNTAnalyzer, Optional[ROIRegion]]:
        """Snapshot the current analyzer state for background preprocess previews."""
        roi_to_use = self._get_active_preprocess_roi()
        snapshot = self._clone_analyzer_snapshot(include_preprocess_outputs=False)
        roi_snapshot = self._clone_roi_region(roi_to_use) if roi_to_use is not None else None
        return snapshot, roi_snapshot

    def _clone_analyzer_for_detection(self) -> Tuple[CNTAnalyzer, Optional[ROIRegion]]:
        """Snapshot the current analyzer state for background single-image detection."""
        snapshot = self._clone_analyzer_snapshot(include_preprocess_outputs=True)
        roi_snapshot = self._clone_roi_region(self.current_roi) if self.current_roi is not None else None
        return snapshot, roi_snapshot

    @staticmethod
    def _run_preprocess_task(analyzer_snapshot: CNTAnalyzer,
                             preprocess_settings: dict,
                             roi_snapshot: Optional[ROIRegion]) -> dict:
        """Run preprocess work off the UI thread using a frozen analyzer snapshot."""
        analyzer_snapshot.preprocess(roi=roi_snapshot, **preprocess_settings)
        return {
            'binary_image': None if analyzer_snapshot.binary_image is None else analyzer_snapshot.binary_image.copy(),
            'processed_image': None if analyzer_snapshot.processed_image is None else analyzer_snapshot.processed_image.copy(),
            'skeleton_image': None if analyzer_snapshot.skeleton_image is None else analyzer_snapshot.skeleton_image.copy(),
            'skeleton_overlay': None if analyzer_snapshot.skeleton_overlay is None else analyzer_snapshot.skeleton_overlay.copy(),
        }

    @staticmethod
    def _run_single_detection_task(analyzer_snapshot: CNTAnalyzer,
                                   detect_settings: dict,
                                   roi_snapshot: Optional[ROIRegion]) -> List[CNTMeasurement]:
        """Run CNT detection off the UI thread using a frozen analyzer snapshot."""
        return list(analyzer_snapshot.detect_cnts_hybrid(roi=roi_snapshot, **detect_settings))

    def _handle_preprocess_result(self,
                                  task_token: int,
                                  snapshot: dict,
                                  future) -> None:
        """Apply a finished preprocess preview back onto the live analyzer."""
        if getattr(self, '_preprocess_future', None) is future:
            self._preprocess_future = None
        if getattr(self, '_preprocess_snapshot', None) is snapshot:
            self._preprocess_snapshot = None

        try:
            result = future.result()
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
            logger.exception("Preprocess preview failed")
            if getattr(self, 'image_panel', None) is not None:
                self.image_panel.show_status(f"预处理预览失败: {exc}")
            return
        except Exception as exc:
            logger.exception("Unexpected preprocess preview failure")
            if getattr(self, 'image_panel', None) is not None:
                self.image_panel.show_status(f"预处理预览失败: {exc}")
            return

        current_roi_signature = self._roi_signature(self._get_active_preprocess_roi())
        if (
            task_token != getattr(self, '_preprocess_token', -1) or
            self.analyzer.image is None or
            id(self.analyzer.image) != snapshot.get('image_id') or
            self._get_preprocess_signature() != snapshot.get('preprocess_signature') or
            current_roi_signature != snapshot.get('active_roi_signature')
        ):
            logger.debug("Discarding stale preprocess preview result.")
            return

        self.analyzer.binary_image = result.get('binary_image')
        self.analyzer.processed_image = result.get('processed_image')
        self.analyzer.skeleton_image = result.get('skeleton_image')
        self.analyzer.skeleton_overlay = result.get('skeleton_overlay')
        self._preprocess_preview_fast = bool(snapshot.get('fast_preview', False))
        self._preprocess_result_exact = bool(snapshot.get('result_exact', False))
        self._last_preprocess_signature = snapshot.get('preprocess_signature')
        self._update_display()

    def _poll_preprocess_result(self, task_token: int, snapshot: dict) -> None:
        """Poll a background preprocess future from the Tk main thread."""
        future = getattr(self, '_preprocess_future', None)
        if future is None or task_token != getattr(self, '_preprocess_token', -1):
            return
        if not future.done():
            try:
                self.root.after(40, self._poll_preprocess_result, task_token, snapshot)
            except tk.TclError:
                logger.debug("Preprocess preview polling was cancelled after the window closed.")
            return
        self._handle_preprocess_result(task_token, snapshot, future)

    def _handle_single_detection_result(self,
                                        task_token: int,
                                        snapshot: dict,
                                        future) -> None:
        """Apply a finished single-image detection back onto the live analyzer."""
        if getattr(self, '_single_detect_future', None) is future:
            self._single_detect_future = None
        if getattr(self, '_single_detect_snapshot', None) is snapshot:
            self._single_detect_snapshot = None

        try:
            measurements = future.result()
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
            logger.exception("CNT检测失败")
            self._set_single_detection_busy_state(False)
            self.control_panel.update_analysis_status(
                f"检测进度: 失败，{exc}",
                color=self.MODERN_COLORS['error'],
            )
            self.image_panel.show_status("CNT检测失败")
            messagebox.showerror("错误", f"CNT检测失败: {exc}")
            return
        except Exception as exc:
            logger.exception("CNT检测时发生未预期的错误")
            self._set_single_detection_busy_state(False)
            self.control_panel.update_analysis_status(
                f"检测进度: 失败，{exc}",
                color=self.MODERN_COLORS['error'],
            )
            self.image_panel.show_status("CNT检测失败")
            messagebox.showerror("错误", f"发生未预期的错误: {exc}")
            return

        if (
            task_token != getattr(self, '_single_detect_token', -1) or
            self.analyzer.image is None or
            id(self.analyzer.image) != snapshot.get('image_id') or
            self._last_preprocess_signature != snapshot.get('preprocess_signature') or
            self._roi_signature(self.current_roi) != snapshot.get('roi_signature')
        ):
            self._set_single_detection_busy_state(False)
            self.image_panel.show_status("检测结果已过期，未覆盖当前图像")
            self._refresh_analysis_status_ui()
            return

        self.control_panel.update_analysis_status(
            "检测进度: 统计中，正在回填结果...",
            color=self.MODERN_COLORS['info'],
        )

        if self.current_roi is not None:
            self.current_roi.measurements = list(measurements)
        else:
            self.analyzer.measurements = list(measurements)

        self._sync_views(refresh_analysis=False)
        self._set_single_detection_busy_state(False)
        self._refresh_analysis_status_ui()
        if getattr(self, 'root', None) is not None:
            try:
                self.root.after(10, self._update_advanced_analysis)
            except tk.TclError:
                logger.debug("Advanced analysis refresh was skipped after the window closed.")

        target_text = f"ROI {self.current_roi.name}" if self.current_roi is not None else "全图"
        count = len(measurements)
        self.image_panel.show_status(f"检测完成: {target_text} 共 {count} 个CNT")
        messagebox.showinfo("检测完成", f"在{target_text}中检测到 {count} 个CNT")

    def _poll_single_detection_result(self, task_token: int, snapshot: dict) -> None:
        """Poll a background detection future from the Tk main thread."""
        future = getattr(self, '_single_detect_future', None)
        if future is None or task_token != getattr(self, '_single_detect_token', -1):
            return
        if not future.done():
            try:
                self.root.after(60, self._poll_single_detection_result, task_token, snapshot)
            except tk.TclError:
                logger.debug("检测轮询在窗口关闭后被取消")
            return
        self._handle_single_detection_result(task_token, snapshot, future)

    def _detect_cnt(self):
        """异步检测 CNT，避免主界面在识别期间卡住。"""
        if self.analyzer.image is None:
            self.control_panel.update_analysis_status(
                "检测输入状态: 请先加载图像后再开始检测",
                color=self.MODERN_COLORS['warning'],
            )
            self.image_panel.show_status("请先加载图像后再开始检测")
            return

        if self._is_single_detection_running():
            self.control_panel.update_analysis_status(
                "检测进度: 已有任务在运行，请等待当前检测完成",
                color=self.MODERN_COLORS['info'],
            )
            self.image_panel.show_status("已有CNT检测正在运行")
            return

        try:
            filter_settings = self._get_detection_filter_settings(strict=True)

            current_signature = self._get_preprocess_signature()
            if not self._has_usable_preprocess_result(require_exact=True, require_skeleton=False):
                self._discard_preprocess_state(include_completed=True, notify=False)
                self._apply_preprocessing(force=True)
            current_signature = self._last_preprocess_signature

            detect_settings = {
                'min_length_um': filter_settings['min_length_um'],
                'max_length_um': filter_settings['max_length_um'],
                'min_slenderness': filter_settings['min_slenderness'],
                'detection_profile': self._get_detection_profile_key(),
                'split_mode': {
                    "不拆分": "off",
                    "标准拆分": "conservative",
                    "强力拆分": "aggressive",
                }.get(self.split_mode_var.get(), self.split_mode_var.get()),
                'merge_distance_px': filter_settings['merge_distance_px'],
            }

            analyzer_snapshot, roi_snapshot = self._clone_analyzer_for_detection()
            self._single_detect_token += 1
            task_token = self._single_detect_token
            snapshot = {
                'image_id': id(self.analyzer.image),
                'preprocess_signature': current_signature,
                'roi_signature': self._roi_signature(self.current_roi),
            }
            self._single_detect_snapshot = snapshot

            self.control_panel.update_analysis_status(
                "检测进度: 开始分析，已完成预处理，后台正在检测...",
                color=self.MODERN_COLORS['info'],
            )
            self.image_panel.show_status("CNT检测中...")
            self._set_single_detection_busy_state(True)

            future = self._single_detect_executor.submit(
                self._run_single_detection_task,
                analyzer_snapshot,
                detect_settings,
                roi_snapshot,
            )
            self._single_detect_future = future
            self.control_panel.update_analysis_status(
                "检测进度: 检测中，界面可继续响应",
                color=self.MODERN_COLORS['info'],
            )
            self.root.after(60, self._poll_single_detection_result, task_token, snapshot)

        except (ValueError, cv2.error, tk.TclError) as e:
            logger.exception("CNT检测失败")
            self._set_single_detection_busy_state(False)
            messagebox.showerror("错误", f"CNT检测失败: {e}")
        except Exception as e:
            logger.exception("CNT检测时发生未预期的错误")
            self._set_single_detection_busy_state(False)
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

    # ===== 显示更新 =====
    def _build_binary_preview_image(self) -> Optional[np.ndarray]:
        """Build the highlighted binary overlay preview in RGB."""
        if self.analyzer.image is None or self.analyzer.binary_image is None:
            return None

        overlay = self.analyzer.image.copy()
        self.analyzer._draw_scale_exclusion_annotation(overlay)
        green_overlay = np.zeros_like(overlay)
        green_overlay[:] = [0, 200, 100]
        roi = self._get_active_preprocess_roi()
        if roi:
            y1, y2, x1, x2 = roi.y, roi.y + roi.height, roi.x, roi.x + roi.width
            binary_mask = np.zeros((overlay.shape[0], overlay.shape[1]), dtype=np.uint8)
            binary_mask[y1:y2, x1:x2] = self.analyzer.binary_image
            mask = binary_mask > 0
        else:
            mask = self.analyzer.binary_image > 0
        alpha = 0.5
        overlay[mask] = cv2.addWeighted(
            overlay, 1 - alpha, green_overlay, alpha, 0
        )[mask]
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    def _update_display(self):
        """更新显示"""
        if self.analyzer.image is None:
            return

        # 同步缩放级别到 ImagePanel（用于比例尺显示）
        self.image_panel.set_zoom_level(self.zoom_level)

        try:
            mode = self.display_var.get()

            if mode == "original":
                image = self.analyzer.image.copy()
                self.analyzer._draw_scale_exclusion_annotation(image)
                for r in self.analyzer.rois:
                    cv2.rectangle(image, (r.x, r.y), (r.x + r.width, r.y + r.height),
                                  r.color, 2)
                    cv2.putText(image, r.name, (r.x + 5, r.y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif mode == "binary":
                image = self._build_binary_preview_image()
                if image is None:
                    return

            elif mode == "skeleton_preview":
                if getattr(self, "_preprocess_preview_fast", False):
                    image = self._build_binary_preview_image()
                    if image is None:
                        return
                else:
                    image = cv2.cvtColor(self.analyzer.get_skeleton_preview(self.current_roi),
                                         cv2.COLOR_BGR2RGB)
            elif mode == "result":
                image = cv2.cvtColor(self.analyzer.get_visualization(self.current_roi),
                                     cv2.COLOR_BGR2RGB)
            elif mode == "skeleton":
                image = cv2.cvtColor(self.analyzer.get_visualization_with_skeleton(self.current_roi),
                                     cv2.COLOR_BGR2RGB)
            else:
                return

            h, w = image.shape[:2]
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            image = cv2.resize(image, (new_w, new_h))

            self.current_image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(self.current_image)

            self.image_panel.clear_canvas()
            self.image_panel.create_image(self.photo, center=True)
            self.image_panel.set_scroll_region(new_w, new_h)

        except (ValueError, cv2.error, tk.TclError) as e:
            logger.exception(f"显示更新错误: {e}")
        except Exception as e:
            logger.exception(f"显示更新时发生未预期的错误: {e}")

    def _on_mousewheel(self, event):
        """鼠标滚轮缩放（严格以鼠标位置为中心）"""
        if self.analyzer.image is None or self.image_panel.canvas is None:
            return

        canvas = self.image_panel.canvas
        old_zoom = self.zoom_level

        # 原图尺寸
        orig_h, orig_w = self.analyzer.image.shape[:2]
        old_img_w = int(orig_w * old_zoom)
        old_img_h = int(orig_h * old_zoom)

        view_w = max(1, canvas.winfo_width())
        view_h = max(1, canvas.winfo_height())

        # 旧图像在画布中的偏移（居中时的 padding）
        old_offset_x = max(0, (view_w - old_img_w) // 2) if old_img_w < view_w else 0
        old_offset_y = max(0, (view_h - old_img_h) // 2) if old_img_h < view_h else 0

        # 鼠标在画布坐标系中的位置
        mouse_canvas_x = canvas.canvasx(event.x)
        mouse_canvas_y = canvas.canvasy(event.y)

        # 鼠标指向的原图像素坐标（浮点）
        img_x = (mouse_canvas_x - old_offset_x) / old_zoom
        img_y = (mouse_canvas_y - old_offset_y) / old_zoom

        # 计算新缩放级别
        if event.num == 4 or event.delta > 0:
            self.zoom_level *= 1.1
        elif event.num == 5 or event.delta < 0:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        if abs(self.zoom_level - old_zoom) < 1e-9:
            return

        self.image_panel.show_status(f"缩放: {self.zoom_level:.0%}")
        self._update_display()

        if self.current_image is None:
            return

        new_img_w, new_img_h = self.current_image.size

        # 新图像在画布中的偏移（居中时的 padding）
        new_offset_x = max(0, (view_w - new_img_w) // 2) if new_img_w < view_w else 0
        new_offset_y = max(0, (view_h - new_img_h) // 2) if new_img_h < view_h else 0

        # 鼠标指向的原图像素在新缩放下的画布坐标
        new_target_x = img_x * self.zoom_level + new_offset_x
        new_target_y = img_y * self.zoom_level + new_offset_y

        # 需要滚动到的位置：让 new_target 出现在鼠标的窗口位置 event.x/y
        scroll_region_w = max(new_img_w, view_w)
        scroll_region_h = max(new_img_h, view_h)

        desired_left = new_target_x - event.x
        desired_top = new_target_y - event.y

        if scroll_region_w > view_w:
            max_left = float(scroll_region_w - view_w)
            desired_left = max(0.0, min(max_left, float(desired_left)))
            # Canvas.xview_moveto 使用“总滚动区域宽度”比例
            x_frac = desired_left / float(scroll_region_w)
            canvas.xview_moveto(x_frac)
        else:
            canvas.xview_moveto(0.0)

        if scroll_region_h > view_h:
            max_top = float(scroll_region_h - view_h)
            desired_top = max(0.0, min(max_top, float(desired_top)))
            y_frac = desired_top / float(scroll_region_h)
            canvas.yview_moveto(y_frac)
        else:
            canvas.yview_moveto(0.0)

    def _fit_image_to_window(self):
        """将当前图像缩放到适应窗口"""
        if self.analyzer.image is None or self.image_panel.canvas is None:
            return

        canvas = self.image_panel.canvas
        canvas.update_idletasks()

        view_w = max(1, canvas.winfo_width())
        view_h = max(1, canvas.winfo_height())
        orig_h, orig_w = self.analyzer.image.shape[:2]
        if orig_w <= 0 or orig_h <= 0:
            return

        fit_zoom = min(view_w / orig_w, view_h / orig_h)
        new_zoom = max(0.1, min(5.0, fit_zoom))
        if abs(new_zoom - self.zoom_level) < 1e-9:
            self.image_panel.show_status("图像已适应当前窗口")
            return

        self.zoom_level = new_zoom
        self._update_display()
        canvas.xview_moveto(0.0)
        canvas.yview_moveto(0.0)
        self.image_panel.show_status("图像已适应当前窗口")

    # ===== 结果更新 =====
    def _update_results(self):
        """更新结果显示"""
        self.result_panel.clear_stats()
        self.result_panel.clear_tree()

        measurements = self._get_active_measurements()
        if not measurements:
            return

        stats = self.analyzer.get_statistics(self.current_roi)
        text_widget = self.result_panel.stats_text

        dispersed_stats = None
        primary_count = int(stats.get('count', len(measurements)))
        length_stats = stats
        display_measurements = measurements
        dispersed_ids = set()
        agglomerated_ids = set()
        try:
            dispersed_stats = self.analyzer.get_dispersed_statistics(self.current_roi)
            if dispersed_stats:
                primary_count = int(dispersed_stats.get('dispersed_count', primary_count))
                length_stats = dispersed_stats.get('dispersed_length_stats') or stats
                display_measurements = dispersed_stats.get('dispersed_measurements') or []
                dispersed_ids = {int(measurement.id) for measurement in (dispersed_stats.get('dispersed_measurements') or [])}
                agglomerated_ids = {int(measurement.id) for measurement in (dispersed_stats.get('agglomerated_measurements') or [])}
        except GUI_EXPECTED_DATA_EXCEPTIONS as e:
            logger.debug(f"获取分散统计失败: {e}")
        except Exception as e:
            logger.exception("获取分散统计时发生未预期的错误")

        if dispersed_stats:
            text_widget.insert(tk.END, "分散CNT数量: ", 'header')
            text_widget.insert(tk.END, f"{primary_count}\n", 'value')
            text_widget.insert(tk.END, "总CNT数量: ", 'header')
            text_widget.insert(tk.END, f"{dispersed_stats['total_count']}\n", 'value')
            text_widget.insert(tk.END, "团聚区域CNT数量: ", 'header')
            text_widget.insert(tk.END, f"{dispersed_stats['agglomerated_count']}\n", 'value')
            text_widget.insert(tk.END, "分散比例: ", 'header')
            text_widget.insert(tk.END, f"{dispersed_stats['dispersed_ratio']:.1%}\n\n", 'value')
        else:
            text_widget.insert(tk.END, "检测到的CNT数量: ", 'header')
            text_widget.insert(tk.END, f"{primary_count}\n\n", 'value')

        text_widget.insert(tk.END, "===== 分散CNT长度统计 (μm) =====\n", 'header')
        text_widget.insert(tk.END, "平均值: ", 'header')
        text_widget.insert(tk.END, f"{length_stats['length_mean']:.2f}\n", 'value')
        text_widget.insert(tk.END, "标准差: ", 'header')
        text_widget.insert(tk.END, f"{length_stats['length_std']:.2f}\n", 'value')
        text_widget.insert(tk.END, "最小值: ", 'header')
        text_widget.insert(tk.END, f"{length_stats['length_min']:.2f}\n", 'value')
        text_widget.insert(tk.END, "最大值: ", 'header')
        text_widget.insert(tk.END, f"{length_stats['length_max']:.2f}\n\n", 'value')

        text_widget.insert(tk.END, "===== 分散CNT长度分布 =====\n", 'header')
        for label, count in length_stats['length_distribution'].items():
            text_widget.insert(tk.END, f"{label}: ", 'header')
            text_widget.insert(tk.END, f"{count}根\n", 'value')

        widths_median = [m.width_median_um for m in display_measurements if m.width_median_um]
        if widths_median:
            text_widget.insert(tk.END, "\n===== 分散CNT宽度统计 (μm) =====\n", 'header')
            text_widget.insert(tk.END, "中位数均值: ", 'header')
            text_widget.insert(tk.END, f"{np.mean(widths_median):.3f}\n", 'value')
            widths_iqr = [m.width_iqr_um for m in display_measurements if m.width_iqr_um]
            if widths_iqr:
                text_widget.insert(tk.END, "IQR均值: ", 'header')
                text_widget.insert(tk.END, f"{np.mean(widths_iqr):.3f}\n", 'value')

        spatial = stats.get('spatial_distribution') or {}
        if spatial:
            text_widget.insert(tk.END, "\n===== 空间分布均匀性 =====\n", 'header')
            uniformity_scores = spatial.get('uniformity_scores') or {}
            text_widget.insert(tk.END, "综合均匀性得分: ", 'header')
            text_widget.insert(tk.END, f"{uniformity_scores.get('overall', 0.0):.1f} / 100（越大越均匀）\n", 'value')
            text_widget.insert(tk.END, "中心点最近邻距离CV: ", 'header')
            text_widget.insert(tk.END, f"{spatial['nearest_neighbor_cv']:.3f}（越小越均匀）\n", 'value')
            text_widget.insert(tk.END, "最近邻指数NNI: ", 'header')
            text_widget.insert(tk.END, f"{spatial.get('nearest_neighbor_index', 0.0):.3f}（大于1更均匀）\n", 'value')
            text_widget.insert(tk.END, f"{spatial['grid_size']}×{spatial['grid_size']}网格CNT数CV: ", 'header')
            text_widget.insert(tk.END, f"{spatial['grid_density_cv']:.3f}（越小越均匀）\n", 'value')
            text_widget.insert(tk.END, "空间熵: ", 'header')
            text_widget.insert(tk.END, f"{spatial['grid_entropy']:.3f}（越大越均匀）\n", 'value')
            text_widget.insert(tk.END, "Moran's I: ", 'header')
            text_widget.insert(tk.END, f"{spatial['morans_i']:.3f}（越大越聚集）\n", 'value')
            text_widget.insert(tk.END, "网格占用率: ", 'header')
            text_widget.insert(tk.END, f"{spatial['occupancy_ratio']:.1%}\n", 'value')

        for m in measurements:
            measurement_id = int(m.id)
            self.result_panel.add_measurement((
                measurement_id,
                f"{m.length_um:.2f}",
                "是" if measurement_id in dispersed_ids else "",
                "是" if measurement_id in agglomerated_ids else "",
            ))

    def _on_select_cnt(self, event):
        """选择CNT时高亮显示"""
        selection = self.result_panel.tree.selection()
        if selection:
            item = self.result_panel.tree.item(selection[0])
            cnt_id = int(item['values'][0])
            self._highlight_cnt(cnt_id)

    def _highlight_cnt(self, cnt_id: int):
        """高亮显示指定的CNT"""
        if self.analyzer.image is None:
            return

        vis_image = self.analyzer.image.copy()
        measurements = self._get_active_measurements()

        for m in measurements:
            if m.id == cnt_id:
                cv2.drawContours(vis_image, [m.contour], -1, (0, 255, 255), 3)
                rect = cv2.minAreaRect(m.contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(vis_image, [box], 0, (255, 0, 255), 2)
            else:
                cv2.drawContours(vis_image, [m.contour], -1, (100, 100, 100), 1)

        image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)
        image = cv2.resize(image, (new_w, new_h))

        self.current_image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.image_panel.clear_canvas()
        self.image_panel.create_image(self.photo)

    # ===== 高级分析 =====
    def _update_advanced_analysis(self):
        """更新高级分析内容"""
        measurements = self._get_active_measurements()
        if not measurements:
            self._clear_advanced_analysis()
            return

        stats = self.analyzer.get_statistics(self.current_roi)
        spatial = stats.get('spatial_distribution') or {}
        cnt_count = int(stats.get('count', len(measurements)))
        dispersed_stats = None
        analysis_measurements = measurements
        try:
            dispersed_stats = self.analyzer.get_dispersed_statistics(self.current_roi)
            if dispersed_stats:
                analysis_measurements = dispersed_stats.get('dispersed_measurements') or measurements
        except GUI_EXPECTED_DATA_EXCEPTIONS as e:
            logger.debug(f"获取高级分析分散统计失败: {e}")
        except Exception:
            logger.exception("获取高级分析分散统计时发生未预期的错误")

        pie_distribution = (
            {
                '分散CNT': int(dispersed_stats.get('dispersed_count', 0)),
                '团聚CNT': int(dispersed_stats.get('agglomerated_count', 0)),
            }
            if dispersed_stats
            else {'CNT': int(cnt_count)}
        )

        self._draw_spatial_score_chart(spatial, cnt_count=cnt_count)
        self._draw_distribution_chart(analysis_measurements)
        self._draw_pie_chart(pie_distribution)
        self._draw_cluster_analysis(analysis_measurements)
        self._draw_spatial_heatmap(spatial, cnt_count=cnt_count)

        # 强制刷新布局
        self.analysis_panel.refresh_layout()

    def _dispose_chart(self, key: str):
        """销毁图表对象，避免长时间运行时累积旧 Figure。"""
        chart = self._charts[key]
        colorbar = chart.get('colorbar')
        if colorbar is not None:
            try:
                colorbar.remove()
            except Exception:
                logger.debug("Unable to remove cached colorbar for chart %s during disposal.", key)
        canvas = chart.get('canvas')
        if canvas is not None:
            try:
                canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass

        figure = chart.get('fig')
        if figure is not None:
            figure.clear()

        chart['fig'] = None
        chart['ax'] = None
        chart['canvas'] = None
        chart['colorbar'] = None
        chart['draw_count'] = 0

    def _init_chart(self, key: str, figsize=(6, 4)):
        """初始化或获取图表对象"""
        chart = self._charts[key]
        frame = self.analysis_panel.get_chart_frame(key)
        if not frame:
            return chart

        should_rebuild = (
            chart['fig'] is None
            or chart.get('draw_count', 0) >= CHART_REBUILD_DRAW_LIMIT
        )
        if should_rebuild:
            self._dispose_chart(key)
            for child in frame.winfo_children():
                child.destroy()
            chart['fig'] = Figure(figsize=figsize, dpi=100)
            chart['fig'].patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            chart['ax'] = chart['fig'].add_subplot(111)
            chart['canvas'] = FigureCanvasTkAgg(chart['fig'], master=frame)
            chart['canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        elif chart['ax'] is not None:
            colorbar = chart.get('colorbar')
            if colorbar is not None:
                try:
                    colorbar.remove()
                except Exception:
                    logger.debug("Unable to remove cached colorbar for chart %s before redraw.", key)
                chart['colorbar'] = None
            chart['ax'].clear()

        chart['draw_count'] = chart.get('draw_count', 0) + 1
        return chart

    def _draw_spatial_score_chart(self, spatial: dict, cnt_count: Optional[int] = None):
        """绘制阴影团聚、均匀度与数量概览。"""
        try:
            chart = self._init_chart('score', figsize=(6.2, 4.3))
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas:
                return

            metrics = self._get_shadow_aggregation_metrics(spatial)
            values = [metrics['score'], metrics['uniformity_score']]
            labels = ['阴影团聚↓', '均匀度↑']
            colors = [
                self.MODERN_COLORS['accent_rose'],
                self.MODERN_COLORS['accent_teal'],
            ]

            y_positions = np.arange(len(labels))
            bars = ax.barh(y_positions, values, color=colors, alpha=0.9, height=0.54)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel('得分 (0-100)', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_title('当前图像 / ROI 双指标概览', fontsize=10, color=self.MODERN_COLORS['text_primary'])
            ax.grid(True, axis='x', alpha=0.25, linestyle='--', color=self.MODERN_COLORS['border'])
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.MODERN_COLORS['border'])
            ax.spines['bottom'].set_color(self.MODERN_COLORS['border'])

            for bar, value in zip(bars, values):
                ax.text(
                    min(value + 2.0, 98.0),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    va='center',
                    fontsize=9,
                    color=self.MODERN_COLORS['text_primary'],
                )

            ax.text(
                0.02,
                0.98,
                (
                    f"CNT数量 {int(cnt_count)}\n" if cnt_count is not None else ""
                ) + "阴影团聚越低越好；均匀度越高越好",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8.4,
                color=self.MODERN_COLORS['text_secondary'],
                bbox={
                    'boxstyle': 'round,pad=0.22',
                    'facecolor': self.MODERN_COLORS['bg_tertiary'],
                    'edgecolor': self.MODERN_COLORS['border'],
                    'alpha': 0.95,
                },
            )

            chart['fig'].tight_layout()
            canvas.draw()

        except GUI_EXPECTED_RENDER_EXCEPTIONS as e:
            logger.exception(f"绘制双指标概览错误: {e}")
        except Exception as e:
            logger.exception(f"绘制双指标概览错误: {e}")

    @staticmethod
    def _build_detailed_length_bins(lengths: List[float], max_bins: int = 22) -> np.ndarray:
        """Build finer adaptive histogram bins for the on-screen length distribution chart."""
        values = np.array([float(v) for v in lengths if np.isfinite(v)], dtype=float)
        base_bins = get_length_histogram_bins(values.tolist())
        left_edge = float(base_bins[0])
        right_edge = float(base_bins[-1])

        if values.size == 0:
            return base_bins

        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if values.size == 1 or np.isclose(value_min, value_max):
            spread = max(0.5, value_max * 0.15 if value_max > 0 else 1.0)
            left = max(0.0, value_min - spread)
            right = max(left + 1.0, value_max + spread)
            return np.linspace(left, right, 7, dtype=float)

        value_range = max(value_max - value_min, 1e-6)
        q25, q75 = np.percentile(values, [25, 75])
        iqr = max(float(q75 - q25), 0.0)
        if iqr > 1e-6:
            bin_width = 2.0 * iqr / np.cbrt(values.size)
        else:
            bin_width = value_range / max(6, min(max_bins, int(np.sqrt(values.size)) + 4))

        if not np.isfinite(bin_width) or bin_width <= 1e-6:
            bin_width = value_range / max(6, min(max_bins, int(np.sqrt(values.size)) + 4))

        bin_count = int(np.clip(np.ceil((right_edge - left_edge) / max(bin_width, 1e-6)), 6, max_bins))
        bins = np.linspace(left_edge, right_edge, bin_count + 1, dtype=float)
        if np.any(np.diff(bins) <= 0):
            return base_bins
        return bins

    @staticmethod
    def _build_smoothed_length_curve(lengths: List[float],
                                     bins: np.ndarray,
                                     sample_points: int = 240) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Approximate a Gaussian-like smooth curve for the histogram using a lightweight KDE."""
        values = np.array([float(v) for v in lengths if np.isfinite(v)], dtype=float)
        original_count = int(values.size)
        if original_count < 2 or bins.size < 2:
            return None, None

        if original_count > 4000:
            sample_idx = np.linspace(0, original_count - 1, 4000, dtype=int)
            values = values[np.unique(sample_idx)]

        value_range = max(float(np.max(values) - np.min(values)), 1e-6)
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if std <= 1e-6:
            std = max(value_range / 6.0, 0.25)

        bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
        if not np.isfinite(bandwidth) or bandwidth <= 1e-6:
            bandwidth = max(value_range / 8.0, 0.25)

        x_curve = np.linspace(float(bins[0]), float(bins[-1]), int(sample_points), dtype=float)
        scaled = (x_curve[:, None] - values[None, :]) / bandwidth
        kernel = np.exp(-0.5 * np.square(scaled)) / (np.sqrt(2.0 * np.pi) * bandwidth)
        density = np.mean(kernel, axis=1)
        mean_bin_width = float(np.mean(np.diff(bins)))
        y_curve = density * original_count * mean_bin_width
        return x_curve, y_curve

    def _draw_distribution_chart(self, measurements: List[CNTMeasurement]):
        """绘制长度分布图 (直方图)"""
        try:
            chart = self._init_chart('histogram')
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas:
                return

            lengths = [m.length_um for m in measurements if m.length_um is not None]
            if not lengths:
                ax.text(0.5, 0.5, "暂无有效长度数据", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return

            bins = self._build_detailed_length_bins(lengths)

            counts, _, _ = ax.hist(
                lengths,
                bins=bins,
                edgecolor='white',
                alpha=0.8,
                color=self.MODERN_COLORS['accent_primary']
            )

            x_curve, y_curve = self._build_smoothed_length_curve(lengths, bins)
            if x_curve is not None and y_curve is not None:
                ax.plot(
                    x_curve,
                    y_curve,
                    color=self.MODERN_COLORS['accent_teal'],
                    linewidth=2.2,
                    label='平滑趋势',
                    zorder=3,
                )
                ax.fill_between(
                    x_curve,
                    y_curve,
                    color=self.MODERN_COLORS['accent_teal'],
                    alpha=0.08,
                    zorder=2,
                )

            # 若数据全部未落入分箱（极端边界情况下），给出明确提示
            if np.sum(counts) == 0:
                ax.text(0.5, 0.5, "当前分箱下无可视柱形，请检查比例尺或过滤参数",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['warning'])

            ax.set_xlabel('长度 (μm)', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_ylabel('数量', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_title('CNT 长度分布', fontsize=10, color=self.MODERN_COLORS['text_primary'])
            if x_curve is not None and y_curve is not None:
                ax.legend(frameon=False, fontsize=8, loc='upper right')

            ax.grid(True, axis='y', alpha=0.3, linestyle='--', color=self.MODERN_COLORS['border'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.MODERN_COLORS['border'])
            ax.spines['bottom'].set_color(self.MODERN_COLORS['border'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])

            chart['fig'].tight_layout()
            canvas.draw()

        except GUI_EXPECTED_RENDER_EXCEPTIONS as e:
            logger.exception(f"绘制直方图错误: {e}")
        except Exception as e:
            logger.exception(f"绘制直方图错误: {e}")

    def _draw_pie_chart(self, distribution: dict):
        """绘制长度占比饼状图"""
        try:
            chart = self._init_chart('pie', figsize=(6, 5))
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas:
                return

            # 确保 distribution 是 dict
            if not isinstance(distribution, dict):
                ax.text(0.5, 0.5, "分布数据无效",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return

            # 过滤掉数量为0的部分
            filtered_data = [(k, v) for k, v in distribution.items() if v > 0]
            if not filtered_data:
                ax.text(0.5, 0.5, "所有分组数量为0",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return

            labels = [item[0] for item in filtered_data]
            sizes = [item[1] for item in filtered_data]

            pie_colors = [
                self.MODERN_COLORS['accent_primary'],
                self.MODERN_COLORS['accent_secondary'],
                self.MODERN_COLORS['accent_tertiary'],
                self.MODERN_COLORS['accent_teal'],
                self.MODERN_COLORS['accent_amber'],
                self.MODERN_COLORS['accent_rose'],
                self.MODERN_COLORS['success'],
                self.MODERN_COLORS['info']
            ]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                startangle=90,
                pctdistance=0.78,
                colors=pie_colors[:len(sizes)],
                textprops={'color': self.MODERN_COLORS['text_secondary'], 'fontsize': 9},
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
            )

            # 环形图效果
            from matplotlib.patches import Circle as MplCircle
            centre_circle = MplCircle((0, 0), 0.65, fc=self.MODERN_COLORS['bg_secondary'])
            ax.add_artist(centre_circle)

            # 中心显示总数
            total = sum(sizes)
            ax.text(0, 0, f'{total}\n根',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14, fontweight='bold',
                    color=self.MODERN_COLORS['accent_primary'])

            ax.set_aspect('equal')

            plt.setp(autotexts, size=8, weight="bold", color="white")
            plt.setp(texts, size=9)

            chart['fig'].tight_layout()
            canvas.draw()

        except GUI_EXPECTED_RENDER_EXCEPTIONS as e:
            logger.exception(f"绘制饼状图错误: {e}")
        except Exception as e:
            logger.exception(f"绘制饼状图错误: {e}")

    def _draw_cluster_analysis(self, measurements: List[CNTMeasurement]):
        """绘制聚类分析图 (散点图)"""
        try:
            chart = self._init_chart('cluster')
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas: return

            # 准备数据: 长度 vs 宽度
            data = []
            for m in measurements:
                # 如果宽度无效，用随机扰动或者设为0，或者跳过
                width = m.width_mean_um if m.width_mean_um and m.width_mean_um > 0 else 0
                if width > 0:
                     data.append([m.length_um, width])
            
            if not data:
                ax.text(0.5, 0.5, "缺乏宽度数据，无法进行聚类分析", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return
            
            X = np.array(data)
            
            # 尝试聚类
            try:
                from sklearn.cluster import KMeans
                # 简单的逻辑：如果数据点少于3个，就分1类；否则分3类
                n_clusters = 3 if len(data) >= 10 else (len(data) if len(data) > 0 else 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                centers = kmeans.cluster_centers_
            except ImportError:
                # 如果没有sklearn，使用简单的基于长度的分组
                labels = []
                for x in X:
                    if x[0] < 5: labels.append(0)
                    elif x[0] < 20: labels.append(1)
                    else: labels.append(2)
                labels = np.array(labels)
                n_clusters = 3
                centers = None

            # 绘制散点
            scatter_colors = [
                self.MODERN_COLORS['accent_primary'],
                self.MODERN_COLORS['accent_secondary'],
                self.MODERN_COLORS['accent_tertiary'],
                self.MODERN_COLORS['accent_teal'],
                self.MODERN_COLORS['accent_amber']
            ]
            
            for i in range(n_clusters):
                cluster_data = X[labels == i]
                if len(cluster_data) > 0:
                    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                               s=30, alpha=0.7, 
                               c=scatter_colors[i % len(scatter_colors)], 
                               label=f'Cluster {i+1}', edgecolors='white', linewidth=0.5)
            
            # 绘制中心点
            if centers is not None:
                ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x', alpha=0.5)

            ax.set_xlabel('长度 (μm)', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_ylabel('平均宽度 (μm)', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_title('长度-宽度散点 / 聚类', fontsize=10, color=self.MODERN_COLORS['text_primary'])
            if n_clusters > 1:
                ax.legend(frameon=False, fontsize=8)
            
            # 样式
            ax.grid(True, alpha=0.3, linestyle='--', color=self.MODERN_COLORS['border'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.MODERN_COLORS['border'])
            ax.spines['bottom'].set_color(self.MODERN_COLORS['border'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            
            chart['fig'].tight_layout()
            canvas.draw()

        except GUI_EXPECTED_RENDER_EXCEPTIONS as e:
            logger.exception(f"绘制聚类图错误: {e}")
        except Exception as e:
            logger.exception(f"绘制聚类图错误: {e}")

    @staticmethod
    def _normalize_hotspot_grid(grid: np.ndarray, percentile: float = 85.0) -> np.ndarray:
        """将网格值归一化到 0-1，并放大高分位热点。"""
        grid = np.asarray(grid, dtype=float)
        if grid.size == 0:
            return np.zeros((0, 0), dtype=float)

        positive = grid[grid > 0]
        if positive.size == 0:
            return np.zeros_like(grid, dtype=float)

        upper = float(np.percentile(positive, percentile))
        upper = max(upper, float(np.max(positive)), 1e-6) if upper <= 0 else upper
        return np.clip(grid / upper, 0.0, 1.0)

    @staticmethod
    def _count_grid_regions(mask: np.ndarray) -> int:
        """统计热点网格中的连通区域数。"""
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0:
            return 0

        visited = np.zeros_like(mask, dtype=bool)
        region_count = 0
        rows, cols = mask.shape

        for row in range(rows):
            for col in range(cols):
                if not mask[row, col] or visited[row, col]:
                    continue
                region_count += 1
                stack = [(row, col)]
                visited[row, col] = True
                while stack:
                    cur_row, cur_col = stack.pop()
                    for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        next_row = cur_row + d_row
                        next_col = cur_col + d_col
                        if not (0 <= next_row < rows and 0 <= next_col < cols):
                            continue
                        if visited[next_row, next_col] or not mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        return region_count

    def _build_spatial_hotspot_grid(self, spatial: dict) -> Tuple[np.ndarray, dict]:
        """基于点密度与覆盖率构建更强调局部团聚的热点强度图。"""
        point_grid = np.array(
            spatial.get('point_density_grid') or spatial.get('density_grid') or [],
            dtype=float,
        )
        coverage_grid = np.array(spatial.get('coverage_density_grid') or [], dtype=float)
        shadow_grid = np.array(spatial.get('shadow_density_grid') or [], dtype=float)

        if point_grid.size == 0 and coverage_grid.size == 0 and shadow_grid.size == 0:
            empty = np.zeros((0, 0), dtype=float)
            return empty, {
                'hotspot_mask': np.zeros((0, 0), dtype=bool),
                'severe_mask': np.zeros((0, 0), dtype=bool),
                'hotspot_regions': 0,
                'severe_regions': 0,
                'hotspot_cell_ratio': 0.0,
                'hotspot_mass_ratio': 0.0,
                'peak_share': 0.0,
                'peak_count': 0.0,
            }

        if point_grid.size == 0:
            reference = coverage_grid if coverage_grid.size else shadow_grid
            point_grid = np.zeros_like(reference, dtype=float)
        if coverage_grid.size == 0:
            coverage_grid = np.zeros_like(point_grid, dtype=float)
        if shadow_grid.size == 0:
            shadow_grid = np.zeros_like(point_grid, dtype=float)

        point_norm = self._normalize_hotspot_grid(point_grid, percentile=86.0)
        coverage_norm = self._normalize_hotspot_grid(coverage_grid, percentile=82.0)
        shadow_norm = self._normalize_hotspot_grid(shadow_grid, percentile=84.0)
        hotspot_grid = np.clip(point_norm * 0.46 + coverage_norm * 0.20 + shadow_norm * 0.34, 0.0, 1.0)

        active_mask = (point_grid > 0) | (coverage_grid > 0) | (shadow_grid > 0)
        active_scores = hotspot_grid[active_mask]
        if active_scores.size == 0:
            hotspot_mask = np.zeros_like(hotspot_grid, dtype=bool)
            severe_mask = np.zeros_like(hotspot_grid, dtype=bool)
        elif float(np.max(active_scores) - np.min(active_scores)) < 0.08:
            hotspot_mask = np.zeros_like(hotspot_grid, dtype=bool)
            severe_mask = np.zeros_like(hotspot_grid, dtype=bool)
        else:
            hotspot_percentile = 78.0 if active_scores.size >= 6 else 65.0
            severe_percentile = 91.0 if active_scores.size >= 8 else 82.0
            hotspot_threshold = float(np.percentile(active_scores, hotspot_percentile))
            severe_threshold = float(np.percentile(active_scores, severe_percentile))
            hotspot_mask = active_mask & (hotspot_grid >= hotspot_threshold) & (hotspot_grid > 0)
            severe_mask = active_mask & (hotspot_grid >= severe_threshold) & (hotspot_grid > 0)

        total_points = float(np.sum(point_grid))
        hotspot_points = float(np.sum(point_grid[hotspot_mask])) if hotspot_mask.size else 0.0
        peak_count = float(np.max(point_grid)) if point_grid.size else 0.0
        peak_share = float(peak_count / total_points) if total_points > 0 else 0.0
        hotspot_mass_ratio = float(hotspot_points / total_points) if total_points > 0 else 0.0
        active_cells = int(np.count_nonzero(active_mask))
        hotspot_cells = int(np.count_nonzero(hotspot_mask))
        hotspot_cell_ratio = float(hotspot_cells / active_cells) if active_cells > 0 else 0.0

        return hotspot_grid, {
            'hotspot_mask': hotspot_mask,
            'severe_mask': severe_mask,
            'hotspot_regions': self._count_grid_regions(hotspot_mask),
            'severe_regions': self._count_grid_regions(severe_mask),
            'hotspot_cell_ratio': hotspot_cell_ratio,
            'hotspot_mass_ratio': hotspot_mass_ratio,
            'peak_share': peak_share,
            'peak_count': peak_count,
            'shadow_support_ratio': float(np.mean(shadow_norm[hotspot_mask])) if np.any(hotspot_mask) else 0.0,
        }

    def _get_shadow_aggregation_metrics(self, spatial: Optional[dict]) -> dict:
        """提取用于界面展示的阴影团聚核心指标。"""
        spatial = spatial or {}
        uniformity_scores = spatial.get('uniformity_scores') or {}
        aggregation_scores = spatial.get('aggregation_scores') or {}
        overall_uniformity = float(uniformity_scores.get('overall', 0.0))
        fallback_score = float(aggregation_scores.get('overall', 100.0 - overall_uniformity))
        shadow_grid = np.array(spatial.get('shadow_density_grid') or [], dtype=float)
        shadow_mean = float(spatial.get('shadow_density_mean', 0.0))

        _, hotspot_info = self._build_spatial_hotspot_grid(spatial)
        shadow_support = float(hotspot_info.get('shadow_support_ratio', 0.0))
        hotspot_mass = float(hotspot_info.get('hotspot_mass_ratio', 0.0))
        peak_share = float(hotspot_info.get('peak_share', 0.0))
        shadow_available = shadow_grid.size > 0 or shadow_mean > 0.0

        if shadow_available:
            score = np.clip(
                fallback_score * 0.25 +
                shadow_support * 38.0 +
                hotspot_mass * 23.0 +
                peak_share * 8.0 +
                shadow_mean * 18.0,
                0.0,
                100.0,
            )
        else:
            score = fallback_score

        return {
            **hotspot_info,
            'score': float(score),
            'uniformity_score': overall_uniformity,
            'fallback_score': fallback_score,
        }

    def _format_hotspot_summary(self, spatial: dict) -> str:
        """为典型图叠加说明生成双指标摘要。"""
        metrics = self._get_shadow_aggregation_metrics(spatial)
        return f"阴影团聚{metrics['score']:.1f} | 均匀度{metrics['uniformity_score']:.1f}"

    def _overlay_spatial_hotspots_on_image(self,
                                           image: Optional[np.ndarray],
                                           spatial: Optional[dict]) -> Optional[np.ndarray]:
        """在代表图上叠加热点框，让局部团聚区域一眼可见。"""
        if image is None or getattr(image, 'size', 0) == 0 or not spatial:
            return image

        hotspot_grid, hotspot_info = self._build_spatial_hotspot_grid(spatial)
        hotspot_mask = hotspot_info.get('hotspot_mask')
        severe_mask = hotspot_info.get('severe_mask')
        if hotspot_grid.size == 0 or hotspot_mask is None or not np.any(hotspot_mask):
            return image

        output = image.copy()
        rows, cols = hotspot_grid.shape
        height, width = output.shape[:2]
        hotspot_fill = np.array([245, 158, 11], dtype=np.uint8)
        severe_fill = np.array([244, 63, 94], dtype=np.uint8)

        for row in range(rows):
            for col in range(cols):
                if not hotspot_mask[row, col]:
                    continue

                x1 = int(col * width / cols)
                x2 = int((col + 1) * width / cols)
                y1 = int(row * height / rows)
                y2 = int((row + 1) * height / rows)
                if x2 <= x1 or y2 <= y1:
                    continue

                intensity = float(np.clip(hotspot_grid[row, col], 0.0, 1.0))
                is_severe = bool(severe_mask[row, col]) if severe_mask is not None else False
                fill_color = severe_fill if is_severe else hotspot_fill
                alpha = min(0.48, 0.18 + intensity * 0.24 + (0.08 if is_severe else 0.0))
                cell = output[y1:y2, x1:x2]
                overlay = np.empty_like(cell)
                overlay[:] = fill_color
                output[y1:y2, x1:x2] = cv2.addWeighted(cell, 1.0 - alpha, overlay, alpha, 0)

                border_color = (190, 24, 93) if is_severe else (217, 119, 6)
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (max(x1, x2 - 1), max(y1, y2 - 1)),
                    border_color,
                    2 if is_severe else 1,
                )

        return output

    def _draw_spatial_heatmap(self, spatial: dict, cnt_count: Optional[int] = None):
        """绘制阴影团聚空间热图。"""
        try:
            chart = self._init_chart('heatmap')
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas:
                return

            metrics = self._get_shadow_aggregation_metrics(spatial)
            hotspot_grid, hotspot_info = self._build_spatial_hotspot_grid(spatial)
            if hotspot_grid.size == 0:
                ax.text(0.5, 0.5, "暂无阴影团聚数据",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return

            heatmap = ax.imshow(hotspot_grid * 100.0, cmap='magma', interpolation='nearest', vmin=0, vmax=100)
            ax.set_title('阴影团聚热图', fontsize=10, color=self.MODERN_COLORS['text_primary'])
            ax.set_xlabel('X网格', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_ylabel('Y网格', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            ax.grid(False)

            hotspot_mask = hotspot_info.get('hotspot_mask')
            severe_mask = hotspot_info.get('severe_mask')
            if hotspot_mask is not None:
                rows, cols = hotspot_mask.shape
                for row in range(rows):
                    for col in range(cols):
                        if not hotspot_mask[row, col]:
                            continue
                        edge_color = self.MODERN_COLORS['accent_rose'] if severe_mask is not None and severe_mask[row, col] else self.MODERN_COLORS['accent_amber']
                        ax.add_patch(
                            plt.Rectangle(
                                (col - 0.5, row - 0.5),
                                1.0,
                                1.0,
                                fill=False,
                                linewidth=2.2 if severe_mask is not None and severe_mask[row, col] else 1.3,
                                edgecolor=edge_color,
                            )
                        )

            ax.text(
                0.02,
                0.98,
                (
                    f"CNT数量{int(cnt_count)} | " if cnt_count is not None else ""
                ) + f"阴影团聚{metrics['score']:.1f} | 均匀度{metrics['uniformity_score']:.1f}",
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=8.4,
                color='white',
                bbox={
                    'boxstyle': 'round,pad=0.24',
                    'facecolor': '#111827',
                    'edgecolor': '#111827',
                    'alpha': 0.76,
                },
            )

            chart['colorbar'] = chart['fig'].colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            chart['colorbar'].ax.tick_params(colors=self.MODERN_COLORS['text_secondary'])
            chart['colorbar'].set_label('阴影团聚强度 (0-100)', color=self.MODERN_COLORS['text_secondary'])

            chart['fig'].tight_layout()
            canvas.draw()

        except GUI_EXPECTED_RENDER_EXCEPTIONS as e:
            logger.exception(f"绘制空间热图错误: {e}")
        except Exception as e:
            logger.exception(f"绘制空间热图错误: {e}")

    def _get_current_analysis_settings(self) -> Tuple[dict, dict]:
        """读取当前界面的识别条件，确保多张图在同一条件下比较"""
        filter_settings = self._get_detection_filter_settings(strict=True)
        preprocess_settings = {
            'blur_kernel': int(self.blur_kernel_var.get()),
            'adaptive_block': int(self.adaptive_block_var.get()),
            'adaptive_c': int(self.adaptive_c_var.get()),
            'bridge_strength': int(self.bridge_strength_var.get()),
            'threshold_invert': True,
        }
        detect_settings = {
            'min_length_um': filter_settings['min_length_um'],
            'max_length_um': filter_settings['max_length_um'],
            'min_slenderness': filter_settings['min_slenderness'],
            'detection_profile': self._get_detection_profile_key(),
            'merge_distance_px': filter_settings['merge_distance_px'],
            'split_mode': {
                "不拆分": "off",
                "标准拆分": "conservative",
                "强力拆分": "aggressive",
            }.get(self.split_mode_var.get(), self.split_mode_var.get()),
        }
        return preprocess_settings, detect_settings

    def _get_detection_filter_settings(self, *, strict: bool = False) -> dict:
        """Read and validate user-editable detection filters from the control panel."""
        field_specs = (
            ('min_length_um', '最小长度', self.min_length_um_var, float),
            ('max_length_um', '最大长度', self.max_length_um_var, float),
            ('min_slenderness', '最小长宽比', self.min_slenderness_var, float),
            ('merge_distance_px', '近邻合并距离', self.merge_distance_px_var, float),
        )
        values = {}
        for key, label, variable, cast in field_specs:
            try:
                values[key] = cast(variable.get())
            except (TypeError, ValueError, tk.TclError) as exc:
                if strict:
                    raise ValueError(f"{label}必须是有效数字") from exc
                return {'valid': False, 'message': f"{label}输入无效"}

        if values['min_length_um'] < 0:
            message = "最小长度不能小于 0"
        elif values['max_length_um'] <= 0:
            message = "最大长度必须大于 0"
        elif values['min_slenderness'] < 0:
            message = "最小长宽比不能小于 0"
        elif values['merge_distance_px'] < 0:
            message = "近邻合并距离不能小于 0"
        elif values['max_length_um'] < values['min_length_um']:
            message = "最大长度不能小于最小长度"
        else:
            message = ""

        if message:
            if strict:
                raise ValueError(message)
            return {'valid': False, 'message': message}

        values['valid'] = True
        values['message'] = ""
        return values

    def _build_analysis_context(self) -> dict:
        """快照当前分析参数，避免重复读取界面变量"""
        preprocess_settings, detect_settings = self._get_current_analysis_settings()
        return self._compose_analysis_context(preprocess_settings, detect_settings)

    def _build_compare_analysis_context(self) -> dict:
        """Build compare context from the current UI settings plus compare-specific ROI/scale rules."""
        preprocess_settings, detect_settings = self._get_current_analysis_settings()
        return self._compose_analysis_context(
            preprocess_settings,
            detect_settings,
            analysis_roi={
                'mode': 'center_fraction',
                'fraction': 0.75,
                'label': '中部75%区域',
            },
            scale_detection={
                'recognize_text': False,
            },
        )

    def _get_compare_display_context(self, *results: dict) -> dict:
        """Prefer the actual comparison context stored on results over current live widget values."""
        for result in results:
            if isinstance(result, dict):
                context = result.get('analysis_context')
                if isinstance(context, dict):
                    return context
        return self._build_compare_analysis_context()

    def _compose_analysis_context(self,
                                  preprocess_settings: dict,
                                  detect_settings: dict,
                                  analysis_roi: Optional[dict] = None,
                                  scale_detection: Optional[dict] = None) -> dict:
        """Compose a cacheable analysis context from prepared preprocess/detect settings."""
        scale_um = float(self.scale_um_var.get()) if self.scale_um_var.get() > 0 else float(SCALE_BAR_DEFAULT_UM)
        manual_scale_pixels = float(self.scale_pixels_var.get()) if self.scale_pixels_var.get() > 0 else 0.0
        return {
            'preprocess_settings': preprocess_settings,
            'detect_settings': detect_settings,
            'scale_um': scale_um,
            'manual_scale_pixels': manual_scale_pixels,
            'analysis_roi': analysis_roi,
            'scale_detection': scale_detection,
        }

    @staticmethod
    def _freeze_cache_value(value):
        """将分析上下文转换为可哈希结构"""
        if isinstance(value, dict):
            return tuple((key, CNTAnalyzerGUI._freeze_cache_value(val)) for key, val in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(CNTAnalyzerGUI._freeze_cache_value(item) for item in value)
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _make_analysis_cache_key(self, image_path: str, context: dict, include_visualization: bool) -> tuple:
        """构建带参数签名的缓存键"""
        return (
            str(Path(image_path).resolve()),
            include_visualization,
            self._freeze_cache_value(context),
        )

    @staticmethod
    def _build_center_fraction_roi(width: int,
                                   height: int,
                                   fraction: float,
                                   name: str = "compare_center_focus") -> Optional[ROIRegion]:
        """Build a centered ROI that keeps only the middle fraction of the image."""
        if width <= 0 or height <= 0:
            return None

        clamped_fraction = max(0.1, min(1.0, float(fraction)))
        roi_width = max(1, int(round(width * clamped_fraction)))
        roi_height = max(1, int(round(height * clamped_fraction)))
        x = max(0, (int(width) - roi_width) // 2)
        y = max(0, (int(height) - roi_height) // 2)
        return ROIRegion(name=name, x=x, y=y, width=roi_width, height=roi_height)

    def _resolve_batch_analysis_roi(self, analyzer: CNTAnalyzer, context: dict) -> Optional[ROIRegion]:
        """Resolve a batch-analysis ROI from the frozen context."""
        analysis_roi = context.get('analysis_roi') or {}
        if not analysis_roi:
            return None

        image = getattr(analyzer, 'image', None)
        if image is None or getattr(image, 'size', 0) == 0:
            return None

        mode = str(analysis_roi.get('mode', '')).lower()
        if mode != 'center_fraction':
            return None

        fraction = float(analysis_roi.get('fraction', 1.0))
        label = str(analysis_roi.get('label', 'compare_center_focus'))
        height, width = image.shape[:2]
        return self._build_center_fraction_roi(width, height, fraction, name=label)

    @staticmethod
    def _crop_visualization_to_roi(image: np.ndarray, roi: Optional[ROIRegion]) -> np.ndarray:
        """Crop a rendered visualization down to the analysis ROI when needed."""
        if image is None or getattr(image, 'size', 0) == 0 or roi is None:
            return image

        y1 = max(0, int(roi.y))
        y2 = min(image.shape[0], int(roi.y + roi.height))
        x1 = max(0, int(roi.x))
        x2 = min(image.shape[1], int(roi.x + roi.width))
        if y2 <= y1 or x2 <= x1:
            return image
        return image[y1:y2, x1:x2].copy()

    def _get_cached_analysis_result(self, cache_key: tuple) -> Optional[dict]:
        """读取缓存结果，并刷新其 LRU 顺序"""
        cache_lock = getattr(self, '_analysis_cache_lock', None)
        if cache_lock is None:
            cached = self._analysis_cache.get(cache_key)
            if cached is None:
                return None
            self._analysis_cache.move_to_end(cache_key)
            return cached

        with cache_lock:
            cached = self._analysis_cache.get(cache_key)
            if cached is None:
                return None
            self._analysis_cache.move_to_end(cache_key)
            return cached

    def _store_analysis_result(self, cache_key: tuple, result: dict) -> dict:
        """写入分析缓存，并限制缓存大小"""
        cache_lock = getattr(self, '_analysis_cache_lock', None)
        if cache_lock is None:
            self._analysis_cache[cache_key] = result
            self._analysis_cache.move_to_end(cache_key)
            while len(self._analysis_cache) > self._analysis_cache_limit:
                self._analysis_cache.popitem(last=False)
            return result

        with cache_lock:
            self._analysis_cache[cache_key] = result
            self._analysis_cache.move_to_end(cache_key)
            while len(self._analysis_cache) > self._analysis_cache_limit:
                self._analysis_cache.popitem(last=False)
        return result

    def _run_image_analysis(self,
                            image_path: str,
                            context: dict,
                            include_visualization: bool = False,
                            preview_visualization: bool = False) -> dict:
        """执行单张图片分析，支持缓存复用"""
        analyzer = CNTAnalyzer()
        analyzer.load_image(image_path)

        scale_um = float(context['scale_um'])
        manual_scale_pixels = float(context['manual_scale_pixels'])
        scale_detection = context.get('scale_detection') or {}
        scale_result = analyzer.apply_detected_scale(
            default_micrometers=scale_um,
            recognize_text=bool(scale_detection.get('recognize_text', True)),
        )
        scale_info = scale_result.get('scale_info')
        if not scale_result.get('applied') and manual_scale_pixels > 0 and scale_um > 0:
            analyzer.record_manual_scale(manual_scale_pixels, scale_um, source='batch_manual', confidence='low')

        analysis_roi = self._resolve_batch_analysis_roi(analyzer, context)
        detect_settings = dict(context['detect_settings'])
        detect_settings.pop('roi', None)

        analyzer.preprocess(**context['preprocess_settings'], roi=analysis_roi)
        analyzer.detect_cnts_hybrid(**detect_settings, roi=analysis_roi)
        stats = analyzer.get_statistics(analysis_roi)
        dispersed_stats = analyzer.get_dispersed_statistics(analysis_roi)

        visualization = None
        if include_visualization:
            vis_bgr = analyzer.get_visualization(analysis_roi)
            vis_bgr = self._crop_visualization_to_roi(vis_bgr, analysis_roi)
            visualization = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            if preview_visualization:
                visualization = self._prepare_comparison_display_image(
                    visualization,
                    max_width=1000,
                    max_height=560,
                )

        return {
            'path': image_path,
            'name': Path(image_path).name,
            'stats': stats,
            'dispersed_stats': dispersed_stats,
            'scale_info': scale_info,
            'scale_status': analyzer.get_scale_status(),
            'visualization': visualization,
            'analysis_context': context,
            'analysis_roi': None if analysis_roi is None else {
                'name': analysis_roi.name,
                'x': int(analysis_roi.x),
                'y': int(analysis_roi.y),
                'width': int(analysis_roi.width),
                'height': int(analysis_roi.height),
            },
        }

    def _get_group_analysis_worker_count(self, image_count: int) -> int:
        """根据待分析图片数量和 CPU 数量决定组图并行线程数。"""
        if image_count <= 1:
            return 1

        cpu_count = os.cpu_count() or 1
        return max(1, min(image_count, max(1, cpu_count // 4), 8))

    @staticmethod
    def _should_use_process_pool_for_group_analysis(platform: Optional[str] = None,
                                                    frozen: Optional[bool] = None) -> bool:
        """Process pools are disabled for the Tk GUI path because GUI-bound workers are not process-safe."""
        _ = platform, frozen
        return False

    def _run_group_analysis_task(self,
                                 task: Tuple[int, str],
                                 context: dict,
                                 include_visualization: bool = False,
                                 preview_visualization: bool = False) -> Tuple[int, Optional[dict], Optional[str]]:
        """执行单张组图分析任务，并将异常转换为可汇总的失败信息。"""
        index, image_path = task
        try:
            return index, self._run_image_analysis(
                image_path,
                context,
                include_visualization=include_visualization,
                preview_visualization=preview_visualization,
            ), None
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
            return index, None, f"{Path(image_path).name}: {exc}"
        except Exception as exc:
            logger.exception("单图分析任务发生未预期的错误")
            return index, None, f"{Path(image_path).name}: {exc}"

    def _run_group_analysis_tasks(self,
                                  tasks: List[Tuple[int, str]],
                                  context: dict,
                                  include_visualization: bool = False,
                                  preview_visualization: bool = False,
                                  progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[Tuple[int, Optional[dict], Optional[str]]]:
        """并行执行一批组图分析任务；优先使用进程池，失败时回退到线程池，最后回退到串行。"""
        if not tasks:
            return []

        worker_count = self._get_group_analysis_worker_count(len(tasks))
        if worker_count <= 1:
            results = []
            for idx, task in enumerate(tasks):
                result = self._run_group_analysis_task(
                    task,
                    context,
                    include_visualization=include_visualization,
                    preview_visualization=preview_visualization,
                )
                results.append(result)
                if progress_callback:
                    progress_callback(idx + 1, len(tasks), Path(task[1]).name)
            return results

        if self._should_use_process_pool_for_group_analysis():
            # 尝试使用 ProcessPoolExecutor 进行 CPU 密集型并行处理
            try:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(
                            self._run_group_analysis_task,
                            task,
                            context,
                            include_visualization,
                            preview_visualization,
                        )
                        for task in tasks
                    ]
                    results = []
                    for idx, future in enumerate(futures):
                        result = future.result()
                        results.append(result)
                        if progress_callback:
                            task_name = Path(tasks[idx][1]).name if idx < len(tasks) else "未知"
                            progress_callback(idx + 1, len(tasks), task_name)
                    return results
            except (BrokenProcessPool, RuntimeError) as exc:
                logger.warning("进程池并行分析失败，回退到线程池: %s", exc)
            except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
                logger.warning("进程池并行分析失败，回退到线程池: %s", exc)
            except Exception:
                logger.exception("进程池并行分析时发生未预期的错误，回退到线程池")
        else:
            logger.info("Frozen Windows build detected; skipping process pool for group comparison.")

        # 回退到 ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cnt-group") as executor:
                futures = [
                    executor.submit(
                        self._run_group_analysis_task,
                        task,
                        context,
                        include_visualization,
                        preview_visualization,
                    )
                    for task in tasks
                ]
                results = []
                for idx, future in enumerate(futures):
                    result = future.result()
                    results.append(result)
                    if progress_callback:
                        task_name = Path(tasks[idx][1]).name if idx < len(tasks) else "未知"
                        progress_callback(idx + 1, len(tasks), task_name)
                return results
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
            logger.warning("线程池并行分析失败，已回退到串行模式: %s", exc)
        except Exception as exc:
            logger.exception("线程池并行分析时发生未预期的错误，已回退到串行模式")

        # 最终回退到串行处理
        results = []
        for idx, task in enumerate(tasks):
            result = self._run_group_analysis_task(
                task,
                context,
                include_visualization=include_visualization,
                preview_visualization=preview_visualization,
            )
            results.append(result)
            if progress_callback:
                progress_callback(idx + 1, len(tasks), Path(task[1]).name)
        return results

    def _analyze_image_with_context(self,
                                    image_path: str,
                                    context: dict,
                                    include_visualization: bool = False,
                                    preview_visualization: Optional[bool] = None) -> dict:
        """Analyze a single image under an explicit frozen context with cache reuse."""
        if preview_visualization is None:
            preview_visualization = include_visualization

        base_cache_key = self._make_analysis_cache_key(image_path, context, False)
        visual_cache_key = self._make_analysis_cache_key(image_path, context, True)
        cache_key = visual_cache_key if include_visualization else base_cache_key

        cached = self._get_cached_analysis_result(cache_key)
        if cached is not None:
            return cached

        if not include_visualization:
            cached_visual = self._get_cached_analysis_result(visual_cache_key)
            if cached_visual is not None:
                return cached_visual

        if include_visualization:
            cached_base = self._get_cached_analysis_result(base_cache_key)
            if cached_base is not None and cached_base.get('visualization') is not None:
                return cached_base

        result = self._run_image_analysis(
            image_path,
            context,
            include_visualization=include_visualization,
            preview_visualization=preview_visualization,
        )
        stored = self._store_analysis_result(cache_key, result)
        if include_visualization:
            self._store_analysis_result(base_cache_key, stored)
        return stored

    def _analyze_image_file(self, image_path: str, include_visualization: bool = False) -> dict:
        """在不影响当前主界面的前提下分析单张图像"""
        context = self._build_analysis_context()
        return self._analyze_image_with_context(image_path, context, include_visualization=include_visualization)

    def _analyze_image_files(self,
                             image_paths: List[str],
                             group_label: str,
                             context: Optional[dict] = None,
                             include_visualization: bool = True,
                             preview_visualization: bool = True,
                             progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[dict], List[str]]:
        """批量分析一组图像，优先复用缓存并并行处理未命中的图片。"""
        context = context or self._build_analysis_context()
        ordered_results: List[Optional[dict]] = [None] * len(image_paths)
        failures: List[str] = []
        pending_tasks: List[Tuple[int, str]] = []
        pending_cache_keys = {}

        for index, image_path in enumerate(image_paths):
            base_cache_key = self._make_analysis_cache_key(image_path, context, False)
            visual_cache_key = self._make_analysis_cache_key(image_path, context, True)
            cached = self._get_cached_analysis_result(base_cache_key)
            if cached is None:
                cached = self._get_cached_analysis_result(visual_cache_key)

            if cached is not None:
                ordered_results[index] = cached
                continue

            pending_tasks.append((index, image_path))
            pending_cache_keys[index] = base_cache_key

        if progress_callback and pending_tasks:
            progress_callback(0, len(pending_tasks), "准备开始...")

        run_group_analysis_tasks = self._run_group_analysis_tasks
        task_kwargs = {
            'include_visualization': include_visualization,
            'preview_visualization': preview_visualization,
        }
        if pending_tasks and 'progress_callback' in inspect.signature(run_group_analysis_tasks).parameters:
            task_kwargs['progress_callback'] = progress_callback

        for index, result, error in run_group_analysis_tasks(
            pending_tasks,
            context,
            **task_kwargs,
        ):
            if result is not None:
                stored = self._store_analysis_result(pending_cache_keys[index], result)
                ordered_results[index] = stored
                if stored.get('visualization') is not None:
                    visual_cache_key = self._make_analysis_cache_key(stored['path'], context, True)
                    self._store_analysis_result(visual_cache_key, stored)
            elif error:
                failures.append(error)

        results = [result for result in ordered_results if result is not None]

        if not results:
            failure_summary = "；".join(failures[:3]) if failures else "未返回具体失败原因"
            raise ValueError(f"{group_label}未成功分析任何图像: {failure_summary}")

        return results, failures

    def _summarize_numeric_series(self, values: List[float]) -> dict:
        """计算均值、标准差、方差等聚合统计"""
        array = np.array(values, dtype=float)
        if array.size == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'var': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0,
            }

        return {
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'var': float(np.var(array)),
            'min': float(np.min(array)),
            'max': float(np.max(array)),
            'total': float(np.sum(array)),
        }

    def _summarize_group_results(self, group_label: str, results: List[dict]) -> dict:
        """汇总一组图像的CNT统计结果"""
        count_values: List[float] = []
        dispersed_count_values: List[float] = []
        agglomerated_count_values: List[float] = []
        dispersed_ratio_values: List[float] = []
        length_mean_values: List[float] = []
        dispersed_length_mean_values: List[float] = []
        nn_values: List[float] = []
        nn_index_values: List[float] = []
        grid_values: List[float] = []
        moran_values: List[float] = []
        entropy_values: List[float] = []
        occupancy_values: List[float] = []
        uniformity_nn_values: List[float] = []
        uniformity_grid_values: List[float] = []
        uniformity_moran_values: List[float] = []
        uniformity_values: List[float] = []
        aggregation_nn_values: List[float] = []
        aggregation_grid_values: List[float] = []
        aggregation_moran_values: List[float] = []
        aggregation_values: List[float] = []
        shadow_aggregation_values: List[float] = []
        density_grids: List[np.ndarray] = []
        file_details: List[dict] = []

        for result in results:
            stats = result.get('stats', {})
            dispersed_stats = result.get('dispersed_stats', {})
            spatial = stats.get('spatial_distribution') or {}
            uniformity_scores = spatial.get('uniformity_scores') or {}
            aggregation_scores = spatial.get('aggregation_scores') or {}

            count = float(stats.get('count', 0))
            dispersed_count = float(dispersed_stats.get('dispersed_count', count))
            agglomerated_count = float(dispersed_stats.get('agglomerated_count', 0.0))
            dispersed_ratio = float(dispersed_stats.get('dispersed_ratio', 1.0 if count > 0 else 0.0))
            length_mean = float(stats.get('length_mean', 0.0))
            dispersed_length_mean = float((dispersed_stats.get('dispersed_length_stats') or {}).get('length_mean', length_mean))
            nn_cv = float(spatial.get('nearest_neighbor_cv', 0.0))
            nn_index = float(spatial.get('nearest_neighbor_index', 0.0))
            grid_cv = float(spatial.get('grid_density_cv', 0.0))
            morans_i = float(spatial.get('morans_i', 0.0))
            grid_entropy = float(spatial.get('grid_entropy', 0.0))
            occupancy_ratio = float(spatial.get('occupancy_ratio', 0.0))
            uniformity_nn = float(uniformity_scores.get('nearest_neighbor', 0.0))
            uniformity_grid = float(uniformity_scores.get('grid_density', 0.0))
            uniformity_moran = float(uniformity_scores.get('moran', 0.0))
            uniformity_overall = float(uniformity_scores.get('overall', 0.0))
            aggregation_nn = float(aggregation_scores.get('nearest_neighbor', 100.0 - uniformity_nn))
            aggregation_grid = float(aggregation_scores.get('grid_density', 100.0 - uniformity_grid))
            aggregation_moran = float(aggregation_scores.get('moran', 100.0 - uniformity_moran))
            aggregation_overall = float(aggregation_scores.get('overall', 100.0 - uniformity_overall))
            shadow_aggregation = float(self._get_shadow_aggregation_metrics(spatial).get('score', aggregation_overall))

            count_values.append(count)
            dispersed_count_values.append(dispersed_count)
            agglomerated_count_values.append(agglomerated_count)
            dispersed_ratio_values.append(dispersed_ratio)
            length_mean_values.append(length_mean)
            dispersed_length_mean_values.append(dispersed_length_mean)
            nn_values.append(nn_cv)
            nn_index_values.append(nn_index)
            grid_values.append(grid_cv)
            moran_values.append(morans_i)
            entropy_values.append(grid_entropy)
            occupancy_values.append(occupancy_ratio)
            uniformity_nn_values.append(uniformity_nn)
            uniformity_grid_values.append(uniformity_grid)
            uniformity_moran_values.append(uniformity_moran)
            uniformity_values.append(uniformity_overall)
            aggregation_nn_values.append(aggregation_nn)
            aggregation_grid_values.append(aggregation_grid)
            aggregation_moran_values.append(aggregation_moran)
            aggregation_values.append(aggregation_overall)
            shadow_aggregation_values.append(shadow_aggregation)

            density_grid = np.array(spatial.get('density_grid') or np.zeros((10, 10)), dtype=float)
            density_grids.append(density_grid)

            file_details.append({
                'name': result['name'],
                'count': count,
                'dispersed_count': dispersed_count,
                'agglomerated_count': agglomerated_count,
                'dispersed_ratio': dispersed_ratio,
                'length_mean': length_mean,
                'dispersed_length_mean': dispersed_length_mean,
                'nearest_neighbor_cv': nn_cv,
                'nearest_neighbor_index': nn_index,
                'grid_density_cv': grid_cv,
                'morans_i': morans_i,
                'aggregation_nn_score': aggregation_nn,
                'aggregation_grid_score': aggregation_grid,
                'aggregation_moran_score': aggregation_moran,
                'aggregation_score': aggregation_overall,
                'shadow_aggregation_score': shadow_aggregation,
                'uniformity_nn_score': uniformity_nn,
                'uniformity_grid_score': uniformity_grid,
                'uniformity_moran_score': uniformity_moran,
                'uniformity_score': uniformity_overall,
            })

        mean_density_grid = np.mean(np.stack(density_grids, axis=0), axis=0) if density_grids else np.zeros((10, 10))

        return {
            'label': group_label,
            'image_count': len(results),
            'results': results,
            'file_details': file_details,
            'count_stats': self._summarize_numeric_series(count_values),
            'dispersed_count_stats': self._summarize_numeric_series(dispersed_count_values),
            'agglomerated_count_stats': self._summarize_numeric_series(agglomerated_count_values),
            'dispersed_ratio_stats': self._summarize_numeric_series(dispersed_ratio_values),
            'length_mean_stats': self._summarize_numeric_series(length_mean_values),
            'dispersed_length_mean_stats': self._summarize_numeric_series(dispersed_length_mean_values),
            'spatial_stats': {
                'nearest_neighbor_cv': self._summarize_numeric_series(nn_values),
                'nearest_neighbor_index': self._summarize_numeric_series(nn_index_values),
                'grid_density_cv': self._summarize_numeric_series(grid_values),
                'morans_i': self._summarize_numeric_series(moran_values),
                'grid_entropy': self._summarize_numeric_series(entropy_values),
                'occupancy_ratio': self._summarize_numeric_series(occupancy_values),
                'aggregation_nn_score': self._summarize_numeric_series(aggregation_nn_values),
                'aggregation_grid_score': self._summarize_numeric_series(aggregation_grid_values),
                'aggregation_moran_score': self._summarize_numeric_series(aggregation_moran_values),
                'aggregation_score': self._summarize_numeric_series(aggregation_values),
                'shadow_aggregation_score': self._summarize_numeric_series(shadow_aggregation_values),
                'uniformity_nn_score': self._summarize_numeric_series(uniformity_nn_values),
                'uniformity_grid_score': self._summarize_numeric_series(uniformity_grid_values),
                'uniformity_moran_score': self._summarize_numeric_series(uniformity_moran_values),
                'uniformity_score': self._summarize_numeric_series(uniformity_values),
            },
            'mean_density_grid': mean_density_grid.tolist(),
        }

    def _compute_two_group_tests(self, base_values: List[float], exp_values: List[float]) -> dict:
        """计算两组样本的t检验与Mann-Whitney U检验"""
        base = np.array(base_values, dtype=float)
        exp = np.array(exp_values, dtype=float)

        result = {
            't_stat': None,
            't_pvalue': None,
            'mw_stat': None,
            'mw_pvalue': None,
        }

        if base.size == 0 or exp.size == 0:
            return result

        try:
            t_stat, t_pvalue = ttest_ind(base, exp, equal_var=False, nan_policy='omit')
            result['t_stat'] = float(t_stat) if np.isfinite(t_stat) else None
            result['t_pvalue'] = float(t_pvalue) if np.isfinite(t_pvalue) else None
        except (TypeError, ValueError, FloatingPointError) as exc:
            logger.debug("Welch t-test failed for comparison inputs: %s", exc)

        try:
            mw_stat, mw_pvalue = mannwhitneyu(base, exp, alternative='two-sided')
            result['mw_stat'] = float(mw_stat) if np.isfinite(mw_stat) else None
            result['mw_pvalue'] = float(mw_pvalue) if np.isfinite(mw_pvalue) else None
        except (TypeError, ValueError, FloatingPointError) as exc:
            logger.debug("Mann-Whitney U test failed for comparison inputs: %s", exc)

        return result

    def _format_pvalue(self, value: Optional[float]) -> str:
        """格式化p值"""
        if value is None:
            return "N/A"
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"

    def _get_significance_marker(self, value: Optional[float]) -> str:
        """根据p值返回显著性星号"""
        if value is None:
            return "n.s."
        if value < 0.001:
            return "***"
        if value < 0.01:
            return "**"
        if value < 0.05:
            return "*"
        return "n.s."

    def _get_preferred_pvalue(self, test_result: Optional[dict]) -> Optional[float]:
        """优先返回更能代表差异证据的p值。"""
        if not test_result:
            return None

        candidates = [
            value for value in (
                test_result.get('t_pvalue'),
                test_result.get('mw_pvalue'),
            )
            if value is not None
        ]
        if not candidates:
            return None
        return float(min(candidates))

    def _is_test_significant(self, test_result: Optional[dict], alpha: float = 0.05) -> bool:
        """任一检验达到阈值即视为存在统计学证据。"""
        pvalue = self._get_preferred_pvalue(test_result)
        return pvalue is not None and pvalue < alpha

    def _get_group_detail_series(self, group_summary: dict, key: str) -> List[float]:
        """提取组内逐图数值序列。"""
        return [float(detail.get(key, 0.0)) for detail in group_summary.get('file_details', [])]

    def _compute_group_comparison_tests(self, base_group: dict, exp_group: dict) -> dict:
        """统一计算组别对比中的统计检验结果。"""
        return {
            'count': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'count'),
                self._get_group_detail_series(exp_group, 'count'),
            ),
            'dispersed_count': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'dispersed_count'),
                self._get_group_detail_series(exp_group, 'dispersed_count'),
            ),
            'dispersed_ratio': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'dispersed_ratio'),
                self._get_group_detail_series(exp_group, 'dispersed_ratio'),
            ),
            'nearest_neighbor_cv': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'nearest_neighbor_cv'),
                self._get_group_detail_series(exp_group, 'nearest_neighbor_cv'),
            ),
            'grid_density_cv': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'grid_density_cv'),
                self._get_group_detail_series(exp_group, 'grid_density_cv'),
            ),
            'uniformity_nn_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'uniformity_nn_score'),
                self._get_group_detail_series(exp_group, 'uniformity_nn_score'),
            ),
            'uniformity_grid_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'uniformity_grid_score'),
                self._get_group_detail_series(exp_group, 'uniformity_grid_score'),
            ),
            'uniformity_moran_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'uniformity_moran_score'),
                self._get_group_detail_series(exp_group, 'uniformity_moran_score'),
            ),
            'uniformity_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'uniformity_score'),
                self._get_group_detail_series(exp_group, 'uniformity_score'),
            ),
            'aggregation_nn_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'aggregation_nn_score'),
                self._get_group_detail_series(exp_group, 'aggregation_nn_score'),
            ),
            'aggregation_grid_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'aggregation_grid_score'),
                self._get_group_detail_series(exp_group, 'aggregation_grid_score'),
            ),
            'aggregation_moran_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'aggregation_moran_score'),
                self._get_group_detail_series(exp_group, 'aggregation_moran_score'),
            ),
            'aggregation_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'aggregation_score'),
                self._get_group_detail_series(exp_group, 'aggregation_score'),
            ),
            'shadow_aggregation_score': self._compute_two_group_tests(
                self._get_group_detail_series(base_group, 'shadow_aggregation_score'),
                self._get_group_detail_series(exp_group, 'shadow_aggregation_score'),
            ),
        }

    def _collect_group_image_paths(self, group_dir: Path) -> List[str]:
        """收集目录下支持的图片文件"""
        valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
        if not group_dir.exists():
            return []

        return [
            str(path) for path in sorted(group_dir.rglob('*'))
            if path.is_file() and path.suffix.lower() in valid_ext
        ]

    def _select_representative_result(self, group_summary: dict) -> dict:
        """选择最接近组平均水平的典型图像"""
        details = group_summary.get('file_details', [])
        if not details:
            raise ValueError(f"{group_summary.get('label', '该组')}没有可用结果")

        shadow_mean = group_summary['spatial_stats'].get('shadow_aggregation_score', {}).get('mean', 0.0)
        shadow_std = max(group_summary['spatial_stats'].get('shadow_aggregation_score', {}).get('std', 0.0), 1.0)
        uniformity_mean = group_summary['spatial_stats']['uniformity_score']['mean']
        uniformity_std = max(group_summary['spatial_stats']['uniformity_score']['std'], 1.0)

        def score(detail: dict) -> float:
            return (
                abs(detail.get('shadow_aggregation_score', 0.0) - shadow_mean) / shadow_std +
                abs(detail.get('uniformity_score', 0.0) - uniformity_mean) / uniformity_std
            )

        representative_detail = min(details, key=score)
        representative_result = next(
            result for result in group_summary['results']
            if result['name'] == representative_detail['name']
        )
        if representative_result.get('visualization') is not None:
            return representative_result
        context = representative_result.get('analysis_context') or self._build_compare_analysis_context()
        return self._analyze_image_with_context(
            representative_result['path'],
            context,
            include_visualization=True,
            preview_visualization=True,
        )

    def _annotate_heatmap_cells(self, ax, grid: np.ndarray):
        """在热图格子上标注数值"""
        if grid.size == 0:
            return

        max_value = float(np.max(grid)) if grid.size else 0.0
        threshold = max_value * 0.55
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value = float(grid[row, col])
                text_color = 'white' if value >= threshold and max_value > 0 else self.MODERN_COLORS['text_primary']
                ax.text(col, row, f"{value:.0f}",
                        ha='center', va='center',
                        fontsize=6.5, color=text_color)

    def _legacy_format_group_detail_lines(self, group_summary: dict) -> List[str]:
        """生成组内逐图统计明细"""
        lines = [f"{group_summary['label']}逐图结果:"]
        for detail in group_summary['file_details']:
            lines.append(
                f"  - {detail['name']}: CNT={detail['count']:.0f}，团聚风险总分={detail.get('aggregation_score', 0.0):.1f}，"
                f"综合均匀性得分={detail.get('uniformity_score', 0.0):.1f}，Moran's I={detail['morans_i']:.3f}，"
                f"最近邻CV={detail['nearest_neighbor_cv']:.3f}，网格CNT数CV={detail['grid_density_cv']:.3f}"
            )
        return lines

    def _legacy_format_group_comparison_summary(self,
                                         base_group: dict,
                                         exp_group: dict,
                                         note: Optional[str] = None,
                                         failures: Optional[List[str]] = None) -> str:
        """生成组别对比摘要"""
        compare_context = self._get_compare_display_context(
            *(base_group.get('results') or []),
            *(exp_group.get('results') or []),
        )
        base_count = base_group['count_stats']
        exp_count = exp_group['count_stats']
        base_spatial = base_group['spatial_stats']
        exp_spatial = exp_group['spatial_stats']
        base_counts = [detail['count'] for detail in base_group['file_details']]
        exp_counts = [detail['count'] for detail in exp_group['file_details']]
        base_nn = [detail['nearest_neighbor_cv'] for detail in base_group['file_details']]
        exp_nn = [detail['nearest_neighbor_cv'] for detail in exp_group['file_details']]
        base_grid = [detail['grid_density_cv'] for detail in base_group['file_details']]
        exp_grid = [detail['grid_density_cv'] for detail in exp_group['file_details']]
        base_uniformity = [detail.get('uniformity_score', 0.0) for detail in base_group['file_details']]
        exp_uniformity = [detail.get('uniformity_score', 0.0) for detail in exp_group['file_details']]
        count_tests = self._compute_two_group_tests(base_counts, exp_counts)
        nn_tests = self._compute_two_group_tests(base_nn, exp_nn)
        grid_tests = self._compute_two_group_tests(base_grid, exp_grid)
        uniformity_tests = self._compute_two_group_tests(base_uniformity, exp_uniformity)

        split_mode_label = self.split_mode_var.get()
        profile_label = self.detect_profile_var.get()
        count_diff = exp_count['mean'] - base_count['mean']
        count_ratio = (count_diff / base_count['mean'] * 100.0) if base_count['mean'] > 0 else 0.0

        uniformity_diff = exp_spatial['uniformity_score']['mean'] - base_spatial['uniformity_score']['mean']
        exp_more_uniform = uniformity_diff > 1.0
        base_more_clustered = base_spatial['morans_i']['mean'] > exp_spatial['morans_i']['mean']

        lines: List[str] = []
        if note:
            lines.append(note)
            lines.append("")

        lines.extend([
            f"相同识别条件: 模糊={self.blur_kernel_var.get()} / 自适应块={self.adaptive_block_var.get()} / C={self.adaptive_c_var.get()} / 桥接={self.bridge_strength_var.get()} / 最小长度={self.min_length_um_var.get():.1f}μm / 最小长宽比={self.min_slenderness_var.get():.1f} / 检测风格={profile_label} / 拆分模式={split_mode_label} / 合并距离={self.merge_distance_px_var.get()}px",
            f"base组: 共 {base_group['image_count']} 张图，总CNT {int(round(base_count['total']))}，平均 {base_count['mean']:.2f}，标准差 {base_count['std']:.2f}，方差 {base_count['var']:.2f}，范围 {base_count['min']:.0f}~{base_count['max']:.0f}",
            f"实验组: 共 {exp_group['image_count']} 张图，总CNT {int(round(exp_count['total']))}，平均 {exp_count['mean']:.2f}，标准差 {exp_count['std']:.2f}，方差 {exp_count['var']:.2f}，范围 {exp_count['min']:.0f}~{exp_count['max']:.0f}",
        ])

        if count_diff >= 0:
            lines.append(f"CNT数量均值差异: 实验组比base组高 {count_diff:.2f}（+{count_ratio:.1f}%）")
        else:
            lines.append(f"CNT数量均值差异: base组比实验组高 {abs(count_diff):.2f}（+{abs(count_ratio):.1f}%）")

        lines.extend([
            "",
            "组别均匀性统计:",
            f"综合均匀性得分: base组 {base_spatial['uniformity_score']['mean']:.1f}±{base_spatial['uniformity_score']['std']:.1f}，实验组 {exp_spatial['uniformity_score']['mean']:.1f}±{exp_spatial['uniformity_score']['std']:.1f}。该得分范围 0-100，越大越均匀。",
            f"方法一 中心点最近邻CV: base组 {base_spatial['nearest_neighbor_cv']['mean']:.3f}±{base_spatial['nearest_neighbor_cv']['std']:.3f}，实验组 {exp_spatial['nearest_neighbor_cv']['mean']:.3f}±{exp_spatial['nearest_neighbor_cv']['std']:.3f}。该值越小越均匀。",
            f"补充指标 最近邻指数NNI: base组 {base_spatial['nearest_neighbor_index']['mean']:.3f}±{base_spatial['nearest_neighbor_index']['std']:.3f}，实验组 {exp_spatial['nearest_neighbor_index']['mean']:.3f}±{exp_spatial['nearest_neighbor_index']['std']:.3f}。该值大于 1 表示比随机分布更均匀。",
            f"方法二 网格CNT数CV: base组 {base_spatial['grid_density_cv']['mean']:.3f}±{base_spatial['grid_density_cv']['std']:.3f}，实验组 {exp_spatial['grid_density_cv']['mean']:.3f}±{exp_spatial['grid_density_cv']['std']:.3f}。该值越小越均匀。",
            f"方法三 Moran's I: base组 {base_spatial['morans_i']['mean']:.3f}±{base_spatial['morans_i']['std']:.3f}，实验组 {exp_spatial['morans_i']['mean']:.3f}±{exp_spatial['morans_i']['std']:.3f}。该值越大越聚集。",
            "",
            "统计检验:",
            f"CNT数量 t检验 p={self._format_pvalue(count_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(count_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(count_tests['t_pvalue'])}",
            f"综合均匀性得分 t检验 p={self._format_pvalue(uniformity_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(uniformity_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(uniformity_tests['t_pvalue'])}",
            f"最近邻CV t检验 p={self._format_pvalue(nn_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(nn_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(nn_tests['t_pvalue'])}",
            f"网格CNT数CV t检验 p={self._format_pvalue(grid_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(grid_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(grid_tests['t_pvalue'])}",
            "",
        ])

        if exp_count['mean'] > base_count['mean'] and exp_more_uniform:
            conclusion = "结论: 在多图同参数统计下，实验组平均识别CNT数量更多，且整体分布更均匀。"
            if base_more_clustered:
                conclusion += " base组的空间自相关更高，CNT团聚更明显。"
        elif exp_count['mean'] < base_count['mean'] and not exp_more_uniform:
            conclusion = "结论: 在多图同参数统计下，base组平均识别CNT数量更多，且整体分布更均匀。"
            if not base_more_clustered:
                conclusion += " 实验组的空间自相关更高，CNT团聚更明显。"
        else:
            conclusion = "结论: 当前这组参数下，两组在数量和均匀性上的差异未完全同向拉开，可继续微调识别条件。"
        lines.append(conclusion)
        lines.append("")

        lines.extend(self._format_group_detail_lines(base_group))
        lines.append("")
        lines.extend(self._format_group_detail_lines(exp_group))

        if failures:
            lines.append("")
            lines.append("未成功分析的文件:")
            lines.extend([f"  - {item}" for item in failures])

        return "\n".join(lines)

    def _get_comparison_image_aspect(self, image: Optional[np.ndarray]) -> float:
        """返回对比图像的宽高比。"""
        if image is None or getattr(image, 'size', 0) == 0:
            return 1.0
        height, width = image.shape[:2]
        if height <= 0:
            return 1.0
        return width / height

    def _should_stack_comparison_images(self,
                                        *images: Optional[np.ndarray],
                                        threshold: float = 1.3) -> bool:
        """宽图优先采用上下排布，避免代表图被压扁。"""
        aspects = [
            self._get_comparison_image_aspect(image)
            for image in images
            if image is not None and getattr(image, 'size', 0) > 0
        ]
        if not aspects:
            return False
        return max(aspects) >= threshold or (sum(aspects) / len(aspects)) >= (threshold - 0.15)

    def _build_comparison_layout(self,
                                 *images: Optional[np.ndarray],
                                 stacked: bool,
                                 variant: str) -> dict:
        """返回稳定的预设对比布局，减少不同图像宽高比带来的抖动。"""
        if variant == 'pair':
            if stacked:
                return {
                    'figsize': (13.4, 11.5),
                    'height_ratios': [0.75, 1.6, 1.6],
                    'hspace': 0.40,
                    'wspace': 0.25,
                    'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.05},
                }
            return {
                'figsize': (13.2, 9.5),
                'height_ratios': [0.85, 1.4],
                'hspace': 0.32,
                'wspace': 0.25,
                'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.06},
            }

        if stacked:
            return {
                'figsize': (14.5, 14.2),
                'height_ratios': [0.80, 0.70, 0.75, 1.5, 1.5],
                'hspace': 0.44,
                'wspace': 0.32,
                'adjust': {'left': 0.055, 'right': 0.97, 'top': 0.97, 'bottom': 0.04},
            }
        return {
            'figsize': (14.5, 11.8),
            'height_ratios': [0.90, 0.80, 1.5],
            'hspace': 0.36,
            'wspace': 0.35,
            'adjust': {'left': 0.055, 'right': 0.97, 'top': 0.96, 'bottom': 0.045},
        }

    def _annotate_bar_values(self,
                             ax,
                             bars,
                             fmt: str = "{:.0f}",
                             offset_ratio: float = 0.03) -> None:
        """为柱状图补充数值标签。"""
        bar_list = list(bars)
        if not bar_list:
            return
        heights = [bar.get_height() for bar in bar_list]
        span = max(abs(value) for value in heights) or 1.0
        offset = max(span * offset_ratio, 0.015)
        for bar in bar_list:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset if height >= 0 else height - offset,
                fmt.format(height),
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=8.5,
                color=self.MODERN_COLORS['text_primary'],
            )

    def _prepare_comparison_display_image(self,
                                          image: Optional[np.ndarray],
                                          max_width: int = 1100,
                                          max_height: int = 650) -> Optional[np.ndarray]:
        """为对比页预缩放大图，降低渲染压力。"""
        if image is None or getattr(image, 'size', 0) == 0:
            return image
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)
        if scale >= 0.95:
            return image
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _format_compact_params(self) -> str:
        """生成精简版识别参数描述。"""
        parts = [
            f"模糊{self.blur_kernel_var.get()}/块{self.adaptive_block_var.get()}/C{self.adaptive_c_var.get()}",
            f"长度≥{self.min_length_um_var.get():.1f}μm",
            f"长宽比≥{self.min_slenderness_var.get():.1f}",
            self.detect_profile_var.get(),
        ]
        bridge_strength = self.bridge_strength_var.get()
        if bridge_strength:
            parts.append(f"桥接{bridge_strength}")
        split_mode = self.split_mode_var.get()
        if split_mode and split_mode != "不拆分":
            parts.append(split_mode)
        merge_distance = self.merge_distance_px_var.get()
        if merge_distance:
            parts.append(f"合并{merge_distance}px")
        return "识别参数: " + " | ".join(str(part) for part in parts if part)

    def _format_compare_fixed_params(self, context: Optional[dict] = None) -> str:
        """Describe the comparison settings from the actual stored context."""
        context = context or self._build_compare_analysis_context()
        preprocess_settings = context.get('preprocess_settings') or {}
        detect_settings = context.get('detect_settings') or {}
        analysis_roi = context.get('analysis_roi') or {}
        roi_label = str(analysis_roi.get('label', '中部75%')).strip() or '中部75%'
        parts = [
            (
                "对比预处理: "
                f"模糊{int(preprocess_settings.get('blur_kernel', CALIBRATED_BLUR_KERNEL))}"
                f"/块{int(preprocess_settings.get('adaptive_block', CALIBRATED_ADAPTIVE_BLOCK))}"
                f"/C{int(preprocess_settings.get('adaptive_c', CALIBRATED_ADAPTIVE_C))}"
            ),
            f"分析区域={roi_label}",
            f"长度≥{float(detect_settings.get('min_length_um', self.min_length_um_var.get())):.1f}μm",
            f"长宽比≥{float(detect_settings.get('min_slenderness', self.min_slenderness_var.get())):.1f}",
            self._get_detection_profile_label(
                str(detect_settings.get('detection_profile', self._get_detection_profile_key()))
            ),
        ]
        bridge_strength = int(preprocess_settings.get('bridge_strength', self.bridge_strength_var.get()))
        if bridge_strength:
            parts.append(f"桥接{bridge_strength}")
        split_mode = self._get_split_mode_label(
            str(detect_settings.get('split_mode', self.split_mode_var.get()))
        )
        if split_mode and split_mode != "不拆分":
            parts.append(split_mode)
        merge_distance = float(detect_settings.get('merge_distance_px', self.merge_distance_px_var.get()))
        if merge_distance:
            if float(merge_distance).is_integer():
                parts.append(f"合并{int(merge_distance)}px")
            else:
                parts.append(f"合并{merge_distance:.1f}px")
        return " | ".join(str(part) for part in parts if part)

    def _format_compact_significance(self, test_result: Optional[dict]) -> str:
        """将统计检验压缩成单行 p 值展示。"""
        pvalue = self._get_preferred_pvalue(test_result)
        marker = self._get_significance_marker(pvalue)
        if marker == "n.s.":
            return f"p={self._format_pvalue(pvalue)}"
        return f"p={self._format_pvalue(pvalue)} {marker}"

    def _format_metric_comparison(self,
                                  label: str,
                                  left_label: str,
                                  left_value: float,
                                  right_label: str,
                                  right_value: float,
                                  *,
                                  direction: str,
                                  qualifier: Optional[str] = None,
                                  precision: int = 1,
                                  left_std: Optional[float] = None,
                                  right_std: Optional[float] = None,
                                  test_result: Optional[dict] = None) -> str:
        """格式化精简版指标对比行。"""
        direction_suffix = {'up': '↑', 'down': '↓', 'cluster': '↑聚集'}.get(direction, '')
        title = f"{label}({qualifier})" if qualifier else (f"{label}({direction_suffix})" if direction_suffix else label)
        value_fmt = f"{{:.{precision}f}}"
        if left_std is None or right_std is None:
            line = f"{title}: {left_label} {value_fmt.format(left_value)} vs {right_label} {value_fmt.format(right_value)}"
        else:
            line = (
                f"{title}: {left_label} {value_fmt.format(left_value)}±{value_fmt.format(left_std)} vs "
                f"{right_label} {value_fmt.format(right_value)}±{value_fmt.format(right_std)}"
            )
        if test_result is not None:
            line += f" | {self._format_compact_significance(test_result)}"
        return line

    def _summarize_spatial_hotspots_for_caption(self, spatial: Optional[dict]) -> str:
        """为代表图标题补充热点摘要。"""
        if not spatial:
            return ""
        _, hotspot_info = self._build_spatial_hotspot_grid(spatial)
        hotspot_regions = int(hotspot_info.get('hotspot_regions', 0) or 0)
        if hotspot_regions <= 0:
            return ""
        metrics = self._get_shadow_aggregation_metrics(spatial)
        shadow_support_ratio = float(hotspot_info.get('shadow_support_ratio', 0.0) or 0.0)
        return (
            f"\n阴影团聚 {metrics['score']:.1f} | "
            f"均匀度 {metrics['uniformity_score']:.1f} | "
            f"热点 {hotspot_regions} 处 | "
            f"阴影支撑 {shadow_support_ratio:.1%}"
        )

    def _configure_comparison_image_axis(self,
                                         ax,
                                         image: np.ndarray,
                                         title: str,
                                         spatial: Optional[dict] = None):
        """以正确比例显示对比中的代表图，并附加热点摘要。"""
        if image is None or getattr(image, 'size', 0) == 0:
            ax.text(0.5, 0.5, "暂无图像", ha='center', va='center', transform=ax.transAxes, color=self.MODERN_COLORS['text_secondary'])
            ax.axis('off')
            return

        display_image = self._prepare_comparison_display_image(image)
        ax.imshow(display_image, interpolation='bilinear', aspect='equal')
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.0)
        ax.set_anchor('C')
        ax.text(
            0.02,
            0.98,
            title + self._summarize_spatial_hotspots_for_caption(spatial),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9.5,
            linespacing=1.2,
            color=self.MODERN_COLORS['text_primary'],
            bbox={
                'boxstyle': 'round,pad=0.35',
                'facecolor': self.MODERN_COLORS['bg_secondary'],
                'edgecolor': self.MODERN_COLORS['border'],
                'alpha': 0.90,
                'linewidth': 1.5,
            },
        )
        ax.axis('off')

    def _estimate_comparison_summary_height(self, summary_text: str) -> int:
        """根据摘要文本估算紧凑摘要区高度。"""
        logical_lines = summary_text.splitlines() or [summary_text]
        wrapped_lines = sum(max(1, (len(line) + 71) // 72) for line in logical_lines)
        return min(220, max(128, 34 + wrapped_lines * 14))

    def _fit_comparison_figure_to_frame(self,
                                        figure: Figure,
                                        chart_frame: Optional[tk.Widget],
                                        min_width_px: int = 680,
                                        padding_px: int = 30,
                                        max_width_px: int = 1400,
                                        allow_expand: bool = False) -> None:
        """按中间栏可用宽度等比缩放对比图。"""
        if chart_frame is None:
            return
        chart_frame.update_idletasks()
        available_width = max(
            int(chart_frame.winfo_width()) - padding_px,
            int(getattr(self, 'center_notebook', chart_frame).winfo_width()) - padding_px,
            min_width_px,
        )
        target_width_px = max(min_width_px, min(max_width_px, available_width))
        current_width_px = max(1, int(round(figure.get_figwidth() * figure.dpi)))
        current_height_px = max(1, int(round(figure.get_figheight() * figure.dpi)))
        tolerance = 12
        if abs(current_width_px - target_width_px) <= tolerance:
            return
        if current_width_px <= target_width_px and not allow_expand:
            return
        scale = target_width_px / current_width_px
        scale = max(0.5, min(1.5, scale))
        resized_width_px = max(min_width_px, int(round(current_width_px * scale)))
        resized_height_px = max(480, int(round(current_height_px * scale)))
        figure.set_size_inches(resized_width_px / figure.dpi, resized_height_px / figure.dpi, forward=False)

    def _schedule_comparison_layout_refresh(self, delay_ms: int = 80) -> None:
        """防抖刷新对比分析页的图表尺寸与滚动区域。"""
        if getattr(self, '_comparison_layout_job', None) is not None:
            self.root.after_cancel(self._comparison_layout_job)
        self._comparison_layout_job = self.root.after(delay_ms, self._refresh_comparison_layout)

    def _refresh_comparison_layout(self) -> None:
        """在标签页显示或窗口尺寸变化后，重新贴合对比图尺寸。"""
        self._comparison_layout_job = None
        if not self.comparison_panel or not hasattr(self, 'center_notebook'):
            return
        comparison_tab = self._center_tabs.get('comparison')
        if comparison_tab is None or str(self.center_notebook.select()) != str(comparison_tab):
            return
        chart = self._charts.get('comparison') or {}
        figure = chart.get('fig')
        canvas = chart.get('canvas')
        chart_frame = self.comparison_panel.get_chart_frame('comparison')
        if figure is None or canvas is None or chart_frame is None:
            self.comparison_panel.refresh_layout()
            return
        old_size = (
            int(round(figure.get_figwidth() * figure.dpi)),
            int(round(figure.get_figheight() * figure.dpi)),
        )
        self._fit_comparison_figure_to_frame(figure, chart_frame, allow_expand=False)
        new_height = min(1800, max(800, int(round(figure.get_figheight() * figure.dpi)) + 50))
        self.comparison_panel.set_section_height('comparison', new_height)
        new_size = (
            int(round(figure.get_figwidth() * figure.dpi)),
            int(round(figure.get_figheight() * figure.dpi)),
        )
        if abs(new_size[0] - old_size[0]) > 5 or abs(new_size[1] - old_size[1]) > 5:
            canvas.draw()
        self.comparison_panel.refresh_layout()

    def _render_comparison_figure(self, summary_text: str, figure: Figure):
        """将对比摘要和图表渲染到对比分析面板。"""
        if not self.comparison_panel:
            return
        self._select_center_tab('comparison')
        self.comparison_panel.refresh_layout()
        chart_frame = self.comparison_panel.get_chart_frame('comparison')
        self._fit_comparison_figure_to_frame(figure, chart_frame, allow_expand=True)
        summary_height = self._estimate_comparison_summary_height(summary_text)
        chart_height = min(1600, max(720, int(figure.get_size_inches()[1] * figure.dpi) + 40))
        self.comparison_panel.set_section_height('comparison_summary', summary_height)
        self.comparison_panel.set_section_height('comparison', chart_height)
        self.comparison_panel.set_text_content('comparison_summary', summary_text)
        if chart_frame is None:
            return
        for child in chart_frame.winfo_children():
            child.destroy()
        chart = self._charts['comparison']
        self._dispose_chart('comparison')
        canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        chart['fig'] = figure
        chart['ax'] = None
        chart['canvas'] = canvas
        chart['draw_count'] = 1
        self.comparison_panel.refresh_layout()
        self.comparison_panel.scroll_to_top()
        self._schedule_comparison_layout_refresh(delay_ms=60)
        self._schedule_comparison_layout_refresh(delay_ms=180)

    def _ensure_result_visualization(self, result: dict) -> dict:
        """确保结果中带有检测可视化图像。"""
        if result.get('visualization') is not None:
            return result
        context = result.get('analysis_context') or self._build_analysis_context()
        return self._analyze_image_with_context(
            result['path'],
            context,
            include_visualization=True,
            preview_visualization=True,
        )

    def _plot_group_aggregation_risk_chart(self, ax, base_group: dict, exp_group: dict, tests: dict) -> None:
        """绘制组别阴影团聚与均匀度主图卡。"""
        metric_names = ["阴影团聚", "均匀度"]
        base_metric_means = [base_group['spatial_stats']['shadow_aggregation_score']['mean'], base_group['spatial_stats']['uniformity_score']['mean']]
        exp_metric_means = [exp_group['spatial_stats']['shadow_aggregation_score']['mean'], exp_group['spatial_stats']['uniformity_score']['mean']]
        metric_tests = [tests['shadow_aggregation_score'], tests['uniformity_score']]
        x = np.arange(len(metric_names))
        width = 0.34
        base_metric_bars = ax.bar(x - width / 2, base_metric_means, width, label='base组', color=self.MODERN_COLORS['accent_rose'], alpha=0.95)
        exp_metric_bars = ax.bar(x + width / 2, exp_metric_means, width, label='实验组', color=self.MODERN_COLORS['accent_teal'], alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel('得分 (0-100)', color=self.MODERN_COLORS['text_secondary'])
        ax.set_title('阴影团聚 / 均匀度组间对比', color=self.MODERN_COLORS['text_primary'])
        ax.legend(frameon=False)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        self._annotate_bar_values(ax, base_metric_bars, fmt="{:.1f}", offset_ratio=0.01)
        self._annotate_bar_values(ax, exp_metric_bars, fmt="{:.1f}", offset_ratio=0.01)
        for idx, test in enumerate(metric_tests):
            pvalue = self._get_preferred_pvalue(test)
            marker = self._get_significance_marker(pvalue)
            annotation_y = min(97.5, max(base_metric_means[idx], exp_metric_means[idx]) + 7.0)
            ax.text(x[idx], annotation_y, f"p={self._format_pvalue(pvalue)}\n{marker}", ha='center', va='bottom', fontsize=8, color=self.MODERN_COLORS['text_primary'])
        ax.set_ylim(0, 100)

    def _shorten_distribution_label(self, label: str, max_length: int = 22) -> str:
        """缩短逐图标签，避免文件名把坐标轴挤坏。"""
        if len(label) <= max_length:
            return label
        return label[:max_length - 3] + "..."

    def _render_group_count_distribution_views(self,
                                               ax_box,
                                               ax_detail,
                                               base_group: dict,
                                               exp_group: dict,
                                               count_tests: dict) -> None:
        """根据样本量自适应显示组内 CNT 数量分布。"""
        base_counts = self._get_group_detail_series(base_group, 'count')
        exp_counts = self._get_group_detail_series(exp_group, 'count')
        base_names = [self._shorten_distribution_label(detail['name']) for detail in base_group.get('file_details', [])]
        exp_names = [self._shorten_distribution_label(detail['name']) for detail in exp_group.get('file_details', [])]
        labels = ['base组', '实验组']
        colors = [self.MODERN_COLORS['accent_rose'], self.MODERN_COLORS['accent_teal']]
        min_samples = min(len(base_counts), len(exp_counts)) if base_counts and exp_counts else 0

        if min_samples <= 1:
            for x_pos, values, color in zip((1, 2), (base_counts, exp_counts), colors):
                if not values:
                    continue
                ax_box.scatter([x_pos] * len(values), values, s=48, alpha=0.82, color=color, edgecolors='white', linewidths=0.6, zorder=3)
                mean_value = float(np.mean(values))
                ax_box.hlines(mean_value, x_pos - 0.18, x_pos + 0.18, colors=color, linewidth=2.0, zorder=2)
                ax_box.text(x_pos, mean_value + max(mean_value * 0.03, 1.0), f"{mean_value:.1f}", ha='center', va='bottom', fontsize=8.5, color=self.MODERN_COLORS['text_primary'])
            ax_box.set_xticks([1, 2])
            ax_box.set_xticklabels(labels)
            ax_box.set_title('组内CNT样本点（样本量较少）', color=self.MODERN_COLORS['text_primary'])
            ax_box.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
            ax_box.grid(True, axis='y', alpha=0.25, linestyle='--')
            max_count = max(base_counts + exp_counts) if (base_counts or exp_counts) else 1.0
            ax_box.text(1.5, max_count * 1.08 if max_count > 0 else 0.5, f"每组仅 {min_samples} 张图，已切换为单点概览 | {self._format_compact_significance(count_tests)}", ha='center', va='bottom', fontsize=8.5, color=self.MODERN_COLORS['text_primary'])
            ax_box.set_ylim(0, max_count * 1.22 if max_count > 0 else 1.0)

            detail_labels = [f"base | {name}" for name in base_names] + [f"实验 | {name}" for name in exp_names]
            detail_counts = base_counts + exp_counts
            detail_colors = [colors[0]] * len(base_counts) + [colors[1]] * len(exp_counts)
            positions = np.arange(len(detail_counts))
            ax_detail.barh(positions, detail_counts, color=detail_colors, alpha=0.88)
            ax_detail.set_yticks(positions)
            ax_detail.set_yticklabels(detail_labels)
            ax_detail.invert_yaxis()
            ax_detail.set_title('逐图CNT概览', color=self.MODERN_COLORS['text_primary'])
            ax_detail.set_xlabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
            ax_detail.grid(True, axis='x', alpha=0.25, linestyle='--')
            for y_pos, value in zip(positions, detail_counts):
                ax_detail.text(value + max(max(detail_counts, default=1.0) * 0.015, 0.6), y_pos, f"{value:.0f}", va='center', fontsize=8.5, color=self.MODERN_COLORS['text_primary'])
            return

        box = ax_box.boxplot([base_counts, exp_counts], tick_labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax_box.set_title('组内CNT数量分布', color=self.MODERN_COLORS['text_primary'])
        ax_box.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_box.grid(True, axis='y', alpha=0.25, linestyle='--')
        rng = np.random.default_rng(42)
        ax_box.scatter(1 + rng.normal(0, 0.04, len(base_counts)), base_counts, s=24, alpha=0.75, color=colors[0], edgecolors='white', linewidths=0.4, zorder=3)
        ax_box.scatter(2 + rng.normal(0, 0.04, len(exp_counts)), exp_counts, s=24, alpha=0.75, color=colors[1], edgecolors='white', linewidths=0.4, zorder=3)
        box_top = max(max(base_counts), max(exp_counts)) if base_counts and exp_counts else 1.0
        ax_box.text(1.5, box_top * 1.05 if box_top > 0 else 0.5, f"分布检验 {self._format_compact_significance(count_tests)}", ha='center', va='bottom', fontsize=8.5, color=self.MODERN_COLORS['text_primary'])
        ax_box.set_ylim(0, box_top * 1.18 if box_top > 0 else 1.0)
        base_x = np.arange(1, len(base_counts) + 1)
        exp_x = np.arange(1, len(exp_counts) + 1)
        if max(len(base_counts), len(exp_counts)) <= 3:
            ax_detail.scatter(base_x, base_counts, s=42, color=colors[0], label='base组', zorder=3)
            ax_detail.scatter(exp_x, exp_counts, s=42, color=colors[1], label='实验组', zorder=3)
            ax_detail.set_title('组内逐图CNT样本点', color=self.MODERN_COLORS['text_primary'])
        else:
            ax_detail.plot(base_x, base_counts, marker='o', linewidth=1.8, color=colors[0], label='base组')
            ax_detail.plot(exp_x, exp_counts, marker='o', linewidth=1.8, color=colors[1], label='实验组')
            ax_detail.set_title('组内逐图CNT趋势', color=self.MODERN_COLORS['text_primary'])
        ax_detail.set_xlabel('组内图像序号', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.legend(frameon=False)
        ax_detail.grid(True, alpha=0.25, linestyle='--')

    def _get_comparison_palette(self, left_label: str, right_label: str) -> Tuple[str, str]:
        """根据对比对象标签返回更符合语义的配色。"""
        left_key = (left_label or "").lower()
        right_key = (right_label or "").lower()
        if "base" in left_key or "实验" in right_label or "exp" in right_key:
            return self.MODERN_COLORS['accent_rose'], self.MODERN_COLORS['accent_teal']
        return self.MODERN_COLORS['accent_teal'], self.MODERN_COLORS['accent_rose']

    def _plot_dispersion_bar_chart(self,
                                   ax,
                                   left_group: dict,
                                   right_group: dict,
                                   stats_key: str,
                                   title: str,
                                   y_label: str,
                                   value_fmt: str,
                                   scale: float = 1.0,
                                   test_result: Optional[dict] = None) -> None:
        """绘制双对象分散指标柱状图。"""
        left_label = left_group.get('label', '对象A')
        right_label = right_group.get('label', '对象B')
        left_stats = left_group.get(stats_key, self._summarize_numeric_series([]))
        right_stats = right_group.get(stats_key, self._summarize_numeric_series([]))
        left_color, right_color = self._get_comparison_palette(left_label, right_label)

        values = [float(left_stats.get('mean', 0.0)) * scale, float(right_stats.get('mean', 0.0)) * scale]
        errors = [float(left_stats.get('std', 0.0)) * scale, float(right_stats.get('std', 0.0)) * scale]
        use_errorbar = any(error > 0 for error in errors)
        bars = ax.bar(
            [left_label, right_label],
            values,
            yerr=errors if use_errorbar else None,
            capsize=6 if use_errorbar else 0,
            color=[left_color, right_color],
            alpha=0.9,
        )
        ax.set_title(title, color=self.MODERN_COLORS['text_primary'])
        ax.set_ylabel(y_label, color=self.MODERN_COLORS['text_secondary'])
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')

        bar_tops = [
            max(0.0, value) + (max(0.0, error) if use_errorbar else 0.0)
            for value, error in zip(values, errors)
        ]
        content_top = max(bar_tops) if bar_tops else 0.0
        if content_top <= 0.0:
            content_top = max(
                max((max(0.0, value) for value in values), default=0.0),
                max((max(0.0, error) for error in errors), default=0.0),
            )

        base_floor = 1.0 if scale == 1.0 else 10.0
        top_padding = max(content_top * 0.18, 1.0 if scale == 1.0 else 3.0)
        label_padding = max(content_top * 0.08, 0.4 if scale == 1.0 else 1.5)
        upper_bound = max(content_top + top_padding + label_padding, base_floor)

        if test_result is not None:
            pvalue = self._get_preferred_pvalue(test_result)
            if pvalue is not None:
                marker = self._get_significance_marker(pvalue)
                annotation_y = content_top + top_padding * 0.55
                ax.text(
                    0.5,
                    annotation_y,
                    f"p={self._format_pvalue(pvalue)} | {marker}",
                    transform=ax.get_yaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=8.5,
                    color=self.MODERN_COLORS['text_primary'],
                )
                upper_bound = max(upper_bound, annotation_y + top_padding * 0.85)

        ax.set_ylim(0, upper_bound)
        self._annotate_bar_values(ax, bars, fmt=value_fmt, offset_ratio=0.03)

    def _format_representative_caption(self, group_summary: dict, representative_result: dict) -> str:
        """生成代表图标题，聚焦分散数量与分散比例。"""
        dispersed_stats = representative_result.get('dispersed_stats') or {}
        total_count = int(round(float(representative_result.get('stats', {}).get('count', 0.0))))
        dispersed_count = int(round(float(dispersed_stats.get('dispersed_count', total_count))))
        dispersed_ratio = float(
            dispersed_stats.get(
                'dispersed_ratio',
                1.0 if total_count > 0 and dispersed_count == total_count else 0.0,
            )
        )
        return (
            f"{group_summary.get('label', '对象')}典型分布\n"
            f"{representative_result.get('name', '未命名图像')}\n"
            f"分散CNT={dispersed_count} | 分散比例={dispersed_ratio:.1%}"
        )

    def _render_dispersion_comparison_dashboard(self,
                                                left_group: dict,
                                                right_group: dict,
                                                summary_text: str,
                                                tests: Optional[dict] = None) -> None:
        """统一渲染双图/组别的分散对比面板。"""
        left_typical = self._select_representative_result(left_group)
        right_typical = self._select_representative_result(right_group)

        left_display_image = self._prepare_comparison_display_image(left_typical.get('visualization'), max_width=1100, max_height=650)
        right_display_image = self._prepare_comparison_display_image(right_typical.get('visualization'), max_width=1100, max_height=650)
        stack_images = self._should_stack_comparison_images(left_display_image, right_display_image, threshold=1.65)
        layout = self._build_comparison_layout(left_display_image, right_display_image, stacked=stack_images, variant='pair')
        figure_width, figure_height = layout['figsize']
        figure = Figure(figsize=(min(13.4, figure_width), min(11.2, figure_height)), dpi=92)
        figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])

        if stack_images:
            grid_spec = figure.add_gridspec(
                3,
                2,
                height_ratios=layout['height_ratios'],
                hspace=layout['hspace'],
                wspace=layout['wspace'],
            )
            ax_dispersed_count = figure.add_subplot(grid_spec[0, 0])
            ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 1])
            ax_left = figure.add_subplot(grid_spec[1, :])
            ax_right = figure.add_subplot(grid_spec[2, :])
        else:
            grid_spec = figure.add_gridspec(
                2,
                2,
                height_ratios=layout['height_ratios'],
                hspace=layout['hspace'],
                wspace=layout['wspace'],
            )
            ax_dispersed_count = figure.add_subplot(grid_spec[0, 0])
            ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 1])
            ax_left = figure.add_subplot(grid_spec[1, 0])
            ax_right = figure.add_subplot(grid_spec[1, 1])
        figure.subplots_adjust(**layout['adjust'])

        dispersed_count_test = tests.get('dispersed_count') if tests else None
        dispersed_ratio_test = tests.get('dispersed_ratio') if tests else None

        self._plot_dispersion_bar_chart(
            ax_dispersed_count,
            left_group,
            right_group,
            'dispersed_count_stats',
            '分散CNT数量对比',
            '数量',
            "{:.1f}",
            test_result=dispersed_count_test,
        )
        self._plot_dispersion_bar_chart(
            ax_dispersed_ratio,
            left_group,
            right_group,
            'dispersed_ratio_stats',
            '分散比例对比',
            '比例 (%)',
            "{:.1f}%",
            scale=100.0,
            test_result=dispersed_ratio_test,
        )

        self._configure_comparison_image_axis(
            ax_left,
            left_display_image,
            self._format_representative_caption(left_group, left_typical),
            spatial=left_typical.get('stats', {}).get('spatial_distribution'),
        )
        self._configure_comparison_image_axis(
            ax_right,
            right_display_image,
            self._format_representative_caption(right_group, right_typical),
            spatial=right_typical.get('stats', {}).get('spatial_distribution'),
        )
        self._render_comparison_figure(summary_text, figure)

    def _show_comparison_window(self,
                                left_result: dict,
                                right_result: dict,
                                left_label: str,
                                right_label: str,
                                note: Optional[str] = None):
        """将双图对比结果显示到对比分析面板。"""
        left_result = self._ensure_result_visualization(left_result)
        right_result = self._ensure_result_visualization(right_result)
        summary_text = self._format_comparison_summary(left_result, right_result, left_label, right_label, note)
        left_group = self._summarize_group_results(left_label, [left_result])
        right_group = self._summarize_group_results(right_label, [right_result])
        tests = self._compute_group_comparison_tests(left_group, right_group)
        self._render_dispersion_comparison_dashboard(left_group, right_group, summary_text, tests)

    def _show_group_comparison_window(self,
                                      base_group: dict,
                                      exp_group: dict,
                                      note: Optional[str] = None,
                                      failures: Optional[List[str]] = None):
        """将 base 组与实验组的多图对比结果显示到对比分析面板。"""
        tests = self._compute_group_comparison_tests(base_group, exp_group)
        summary_text = self._format_group_comparison_summary(base_group, exp_group, note, failures)
        self._render_dispersion_comparison_dashboard(base_group, exp_group, summary_text, tests)

    def _format_dispersion_summary_lines(self,
                                         label: str,
                                         total_count: float,
                                         dispersed_count: float,
                                         agglomerated_count: float,
                                         dispersed_ratio: float,
                                         dispersed_length_mean: float,
                                         uniformity_score: float,
                                         length_std: Optional[float] = None) -> List[str]:
        """格式化对比分散/团聚统计摘要。"""
        length_text = f"{dispersed_length_mean:.1f}"
        if length_std is not None:
            length_text = f"{length_text}±{length_std:.1f}"
        return [
            f"{label}: 分散CNT数量 {int(round(dispersed_count))} | 总CNT数量 {int(round(total_count))} | 团聚区域CNT数量 {int(round(agglomerated_count))}",
            f"{label}: 分散比例 {dispersed_ratio:.1%} | 分散CNT长度统计 {length_text} μm | 空间分布均匀性 {uniformity_score:.1f}",
        ]

    def _format_group_detail_lines(self, group_summary: dict) -> List[str]:
        """生成精简版逐图明细。"""
        lines = [f"{group_summary['label']}逐图明细:"]
        for detail in group_summary.get('file_details', []):
            lines.append(
                f"  - {detail['name']}: 分散CNT={detail.get('dispersed_count', 0.0):.0f} | 总CNT={detail.get('count', 0.0):.0f} | "
                f"团聚CNT={detail.get('agglomerated_count', 0.0):.0f} | 分散比例={detail.get('dispersed_ratio', 0.0):.1%} | "
                f"分散长度={detail.get('dispersed_length_mean', 0.0):.1f}μm | 均匀性={detail.get('uniformity_score', 0.0):.1f}"
            )
        return lines

    def _format_group_comparison_summary(self,
                                         base_group: dict,
                                         exp_group: dict,
                                         note: Optional[str] = None,
                                         failures: Optional[List[str]] = None) -> str:
        """生成精简版组别对比摘要。"""
        base_count = base_group['count_stats']
        exp_count = exp_group['count_stats']
        base_dispersed = base_group.get('dispersed_count_stats', self._summarize_numeric_series([]))
        exp_dispersed = exp_group.get('dispersed_count_stats', self._summarize_numeric_series([]))
        base_agglomerated = base_group.get('agglomerated_count_stats', self._summarize_numeric_series([]))
        exp_agglomerated = exp_group.get('agglomerated_count_stats', self._summarize_numeric_series([]))
        base_ratio = base_group.get('dispersed_ratio_stats', self._summarize_numeric_series([]))
        exp_ratio = exp_group.get('dispersed_ratio_stats', self._summarize_numeric_series([]))
        base_dispersed_length = base_group.get('dispersed_length_mean_stats', self._summarize_numeric_series([]))
        exp_dispersed_length = exp_group.get('dispersed_length_mean_stats', self._summarize_numeric_series([]))
        base_spatial = base_group['spatial_stats']
        exp_spatial = exp_group['spatial_stats']
        compare_context = self._get_compare_display_context(
            *(base_group.get('results') or []),
            *(exp_group.get('results') or []),
        )
        tests = self._compute_group_comparison_tests(base_group, exp_group)

        count_tests = tests['count']
        shadow_tests = tests['shadow_aggregation_score']
        uniformity_tests = tests['uniformity_score']
        shadow_diff = base_spatial['shadow_aggregation_score']['mean'] - exp_spatial['shadow_aggregation_score']['mean']
        uniformity_diff = exp_spatial['uniformity_score']['mean'] - base_spatial['uniformity_score']['mean']
        evidence_significant = (
            self._is_test_significant(shadow_tests) or
            self._is_test_significant(uniformity_tests)
        )

        strong_exp_advantage = shadow_diff >= 4.0 and uniformity_diff >= 4.0 and evidence_significant
        trend_exp_advantage = shadow_diff > 0 and uniformity_diff > 0
        strong_base_advantage = shadow_diff <= -4.0 and uniformity_diff <= -4.0 and evidence_significant
        trend_base_advantage = shadow_diff < 0 and uniformity_diff < 0

        if strong_exp_advantage:
            conclusion = f"结论: 实验组更优，阴影团聚低 {shadow_diff:.1f} 分，均匀度高 {uniformity_diff:.1f} 分。"
        elif trend_exp_advantage:
            conclusion = "结论: 实验组趋势更优，阴影团聚更轻且均匀度更高，但统计证据仍偏弱。"
        elif strong_base_advantage:
            conclusion = f"结论: base组更优，阴影团聚低 {abs(shadow_diff):.1f} 分，均匀度高 {abs(uniformity_diff):.1f} 分。"
        elif trend_base_advantage:
            conclusion = "结论: base组趋势更优，阴影团聚更轻且均匀度更高，但统计证据仍偏弱。"
        else:
            conclusion = "结论: 两组在阴影团聚与均匀度上接近，当前更适合解读为趋势对比。"

        lines: List[str] = []
        if note:
            lines.extend([note, ""])

        lines.extend([
            self._format_compare_fixed_params(compare_context),
            "",
            "CNT数量",
            (
                f"base: 均值{base_count['mean']:.1f}±{base_count['std']:.1f} | "
                f"总计{int(round(base_count['total']))} | n={base_group['image_count']}"
            ),
            (
                f"实验: 均值{exp_count['mean']:.1f}±{exp_count['std']:.1f} | "
                f"总计{int(round(exp_count['total']))} | n={exp_group['image_count']} | "
                f"{self._format_compact_significance(count_tests)}"
            ),
            "",
            "分散 / 团聚统计",
        ])
        lines.extend(self._format_dispersion_summary_lines(
            'base',
            base_count['mean'],
            base_dispersed['mean'],
            base_agglomerated['mean'],
            base_ratio['mean'],
            base_dispersed_length['mean'],
            base_spatial['uniformity_score']['mean'],
            length_std=base_dispersed_length['std'],
        ))
        lines.extend(self._format_dispersion_summary_lines(
            '实验',
            exp_count['mean'],
            exp_dispersed['mean'],
            exp_agglomerated['mean'],
            exp_ratio['mean'],
            exp_dispersed_length['mean'],
            exp_spatial['uniformity_score']['mean'],
            length_std=exp_dispersed_length['std'],
        ))
        lines.extend([
            "",
            "核心指标",
            self._format_metric_comparison(
                '阴影团聚',
                'base',
                base_spatial['shadow_aggregation_score']['mean'],
                '实验',
                exp_spatial['shadow_aggregation_score']['mean'],
                direction='down',
                qualifier='0-100，越低越好',
                precision=1,
                left_std=base_spatial['shadow_aggregation_score']['std'],
                right_std=exp_spatial['shadow_aggregation_score']['std'],
                test_result=shadow_tests,
            ),
            self._format_metric_comparison(
                '均匀度',
                'base',
                base_spatial['uniformity_score']['mean'],
                '实验',
                exp_spatial['uniformity_score']['mean'],
                direction='up',
                qualifier='0-100，越高越好',
                precision=1,
                left_std=base_spatial['uniformity_score']['std'],
                right_std=exp_spatial['uniformity_score']['std'],
                test_result=uniformity_tests,
            ),
            "",
            conclusion,
            "",
        ])

        lines.extend(self._format_group_detail_lines(base_group))
        lines.append("")
        lines.extend(self._format_group_detail_lines(exp_group))

        if failures:
            lines.extend(["", "未成功分析的文件:"])
            lines.extend([f"  - {item}" for item in failures])

        return "\n".join(lines)

    def _format_comparison_summary(self,
                                   left_result: dict,
                                   right_result: dict,
                                   left_label: str,
                                   right_label: str,
                                   note: Optional[str] = None) -> str:
        """生成精简版双图对比摘要。"""
        compare_context = self._get_compare_display_context(left_result, right_result)
        left_spatial = left_result['stats'].get('spatial_distribution') or {}
        right_spatial = right_result['stats'].get('spatial_distribution') or {}
        left_metrics = self._get_shadow_aggregation_metrics(left_spatial)
        right_metrics = self._get_shadow_aggregation_metrics(right_spatial)
        left_count = int(left_result['stats'].get('count', 0))
        right_count = int(right_result['stats'].get('count', 0))
        left_dispersed = left_result.get('dispersed_stats') or {}
        right_dispersed = right_result.get('dispersed_stats') or {}
        count_diff = left_count - right_count
        count_ratio = (count_diff / right_count * 100.0) if right_count > 0 else 0.0
        left_uniformity_score = left_metrics['uniformity_score']
        right_uniformity_score = right_metrics['uniformity_score']
        uniformity_diff = left_uniformity_score - right_uniformity_score
        shadow_diff = right_metrics['score'] - left_metrics['score']

        left_better = shadow_diff > 0 and uniformity_diff > 0
        right_better = shadow_diff < 0 and uniformity_diff < 0

        if left_better and shadow_diff >= 4.0 and uniformity_diff >= 4.0:
            conclusion = f"结论: {left_label}更优，阴影团聚低 {shadow_diff:.1f} 分，均匀度高 {uniformity_diff:.1f} 分。"
        elif right_better and abs(shadow_diff) >= 4.0 and abs(uniformity_diff) >= 4.0:
            conclusion = f"结论: {right_label}更优，阴影团聚低 {abs(shadow_diff):.1f} 分，均匀度高 {abs(uniformity_diff):.1f} 分。"
        elif left_better:
            conclusion = f"结论: {left_label}趋势更优，阴影团聚更轻且均匀度更高。"
        elif right_better:
            conclusion = f"结论: {right_label}趋势更优，阴影团聚更轻且均匀度更高。"
        else:
            conclusion = "结论: 两张图在阴影团聚与均匀度上接近，当前更适合看作趋势对比。"

        left_dispersed_length = (left_dispersed.get('dispersed_length_stats') or {}).get('length_mean', left_result['stats'].get('length_mean', 0.0))
        right_dispersed_length = (right_dispersed.get('dispersed_length_stats') or {}).get('length_mean', right_result['stats'].get('length_mean', 0.0))

        lines: List[str] = []
        if note:
            lines.extend([note, ""])

        lines.extend([
            self._format_compare_fixed_params(compare_context),
            "",
            "CNT数量",
            f"{left_label}: {left_result['name']} | CNT={left_count}",
            f"{right_label}: {right_result['name']} | CNT={right_count}",
            f"差异: {left_label if count_diff >= 0 else right_label} {abs(count_ratio):.1f}%",
            "",
            "分散 / 团聚统计",
        ])
        lines.extend(self._format_dispersion_summary_lines(
            left_label,
            left_count,
            float(left_dispersed.get('dispersed_count', left_count)),
            float(left_dispersed.get('agglomerated_count', 0.0)),
            float(left_dispersed.get('dispersed_ratio', 1.0 if left_count > 0 else 0.0)),
            float(left_dispersed_length),
            left_uniformity_score,
        ))
        lines.extend(self._format_dispersion_summary_lines(
            right_label,
            right_count,
            float(right_dispersed.get('dispersed_count', right_count)),
            float(right_dispersed.get('agglomerated_count', 0.0)),
            float(right_dispersed.get('dispersed_ratio', 1.0 if right_count > 0 else 0.0)),
            float(right_dispersed_length),
            right_uniformity_score,
        ))
        lines.extend([
            "",
            "核心指标",
            self._format_metric_comparison(
                '阴影团聚',
                left_label,
                left_metrics['score'],
                right_label,
                right_metrics['score'],
                direction='down',
                qualifier='0-100，越低越好',
                precision=1,
            ),
            self._format_metric_comparison(
                '均匀度',
                left_label,
                left_uniformity_score,
                right_label,
                right_uniformity_score,
                direction='up',
                qualifier='0-100，越高越好',
                precision=1,
            ),
            "",
            conclusion,
        ])

        return "\n".join(lines)

    def _get_compare_initial_dir(self) -> str:
        """返回对比模式默认打开的目录"""
        data_root = Path(__file__).resolve().parent / "DATA"
        return str(data_root) if data_root.exists() else str(Path(__file__).resolve().parent)

    def _get_supported_image_filetypes(self) -> List[Tuple[str, str]]:
        """返回对比模式使用的图像过滤器"""
        return [
            ("图像文件", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp"),
            ("所有文件", "*.*"),
        ]

    def _open_compare_mode_dialog(self):
        """打开统一的对比分析入口"""
        window = tk.Toplevel(self.root)
        window.title("选择对比模式")
        window.geometry("720x390")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()
        window.configure(bg=self.MODERN_COLORS['bg_primary'])

        container = ttk.Frame(window, style='Card.TFrame')
        container.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        header = ttk.Frame(container, style='Card.TFrame')
        header.pack(fill=tk.X, padx=16, pady=(16, 8))

        tk.Label(
            header,
            text="选择对比模式",
            font=('Segoe UI', 13, 'bold'),
            bg=self.MODERN_COLORS['bg_secondary'],
            fg=self.MODERN_COLORS['accent_primary'],
        ).pack(anchor='w')
        tk.Label(
            header,
            text="对比分析会沿用当前界面的预处理和识别参数，并仅分析中部 75% 区域；分析会在后台执行，便于重复对比而不把界面卡死。",
            font=('Segoe UI', 9),
            bg=self.MODERN_COLORS['bg_secondary'],
            fg=self.MODERN_COLORS['text_secondary'],
            justify='left',
            wraplength=640,
        ).pack(anchor='w', pady=(6, 0))

        mode_container = tk.Frame(container, bg=self.MODERN_COLORS['bg_secondary'])
        mode_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))

        def launch(action):
            window.destroy()
            self.root.after(0, action)

        modes = [
            (
                "任意两图对比",
                "手动选择两张图，在相同识别条件下比较分散CNT数量、分散比例和典型分布。",
                "选择两张图",
                self._compare_two_images,
            ),
            (
                "组别统计对比",
                "分别选择 base 组和实验组的多张图，输出分散CNT数量、分散比例、显著性检验和典型分布。",
                "选择两组图",
                self._compare_image_groups,
            ),
        ]

        for title, description, button_text, action in modes:
            card = tk.Frame(
                mode_container,
                bg=self.MODERN_COLORS['bg_secondary'],
                highlightbackground=self.MODERN_COLORS['border'],
                highlightthickness=1,
                bd=0,
            )
            card.pack(fill=tk.X, pady=6)

            text_frame = tk.Frame(card, bg=self.MODERN_COLORS['bg_secondary'])
            text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=14, pady=12)

            tk.Label(
                text_frame,
                text=title,
                font=('Segoe UI', 10, 'bold'),
                bg=self.MODERN_COLORS['bg_secondary'],
                fg=self.MODERN_COLORS['text_primary'],
            ).pack(anchor='w')
            tk.Label(
                text_frame,
                text=description,
                font=('Segoe UI', 9),
                bg=self.MODERN_COLORS['bg_secondary'],
                fg=self.MODERN_COLORS['text_secondary'],
                justify='left',
                wraplength=460,
            ).pack(anchor='w', pady=(4, 0))

            ttk.Button(
                card,
                text=button_text,
                style='Accent.TButton',
                command=lambda selected_action=action: launch(selected_action),
            ).pack(side=tk.RIGHT, padx=14)

        footer = ttk.Frame(container, style='Card.TFrame')
        footer.pack(fill=tk.X, padx=16, pady=(0, 14))
        ttk.Button(footer, text="关闭", command=window.destroy).pack(side=tk.RIGHT)

    def _invoke_compare_batch_analysis(self,
                                       image_paths: List[str],
                                       group_label: str,
                                       include_visualization: bool = True,
                                       progress_callback: Optional[Callable[[int, int, str], None]] = None,
                                       context: Optional[dict] = None) -> Tuple[List[dict], List[str]]:
        """Analyze compare inputs under the current comparison context while remaining compatible with test doubles."""
        analyze_image_files = self._analyze_image_files
        compare_context = context or self._build_compare_analysis_context()
        signature = inspect.signature(analyze_image_files).parameters
        kwargs = {}
        if 'context' in signature:
            kwargs['context'] = compare_context
        if 'include_visualization' in signature:
            kwargs['include_visualization'] = include_visualization
        if 'preview_visualization' in signature:
            kwargs['preview_visualization'] = include_visualization
        if 'progress_callback' in signature:
            kwargs['progress_callback'] = progress_callback
        return analyze_image_files(image_paths, group_label, **kwargs)

    @staticmethod
    def _make_compare_progress_state(total: int, message: str = "准备开始...") -> dict:
        """Create a thread-safe progress snapshot for a compare-analysis request."""
        return {
            'current': 0,
            'total': max(0, int(total)),
            'message': str(message),
        }

    def _build_compare_note(self, mode: str) -> str:
        """Build a user-facing note for the current compare-analysis mode."""
        if str(mode).lower() == 'group':
            return (
                "本次组别对比使用当前界面的预处理与识别参数，"
                "并仅分析中部 75% 区域以避开比例尺区域干扰；"
                "第一组按 base组 统计，第二组按 实验组 统计。"
            )
        return (
            "本次双图对比使用当前界面的预处理与识别参数，"
            "并仅分析中部 75% 区域以避开比例尺区域干扰。"
        )

    def _run_compare_analysis_request(self, request: dict, progress_state: dict) -> dict:
        """Execute a compare-analysis request off the main Tk thread."""
        mode = str(request.get('mode', '')).lower()
        context = request.get('context') or self._build_compare_analysis_context()

        def report(current: int, total: int, message: str = "") -> None:
            progress_state['current'] = max(0, int(current))
            progress_state['total'] = max(0, int(total))
            progress_state['message'] = str(message or progress_state.get('message') or "准备开始...")

        if mode == 'group':
            base_paths = list(request.get('base_paths') or [])
            exp_paths = list(request.get('exp_paths') or [])
            total_images = len(base_paths) + len(exp_paths)
            report(0, total_images, "准备开始...")

            base_results, base_failures = self._invoke_compare_batch_analysis(
                base_paths,
                "base组",
                include_visualization=False,
                context=context,
                progress_callback=lambda current, total, message: report(
                    current,
                    total_images,
                    f"base组 | {message}" if message else "base组分析中",
                ),
            )
            exp_results, exp_failures = self._invoke_compare_batch_analysis(
                exp_paths,
                "实验组",
                include_visualization=False,
                context=context,
                progress_callback=lambda current, total, message: report(
                    len(base_paths) + current,
                    total_images,
                    f"实验组 | {message}" if message else "实验组分析中",
                ),
            )

            report(total_images, total_images, "分析完成")
            return {
                'mode': 'group',
                'base_group': self._summarize_group_results("base组", base_results),
                'exp_group': self._summarize_group_results("实验组", exp_results),
                'failures': base_failures + exp_failures,
                'note': self._build_compare_note('group'),
            }

        if mode == 'pair':
            left_path = str(request.get('left_path', ''))
            right_path = str(request.get('right_path', ''))
            image_paths = [left_path, right_path]
            total_images = len(image_paths)
            report(0, total_images, "准备开始...")
            results, failures = self._invoke_compare_batch_analysis(
                image_paths,
                "双图对比",
                include_visualization=True,
                context=context,
                progress_callback=report,
            )
            result_by_path = {str(Path(result['path']).resolve()): result for result in results}
            left_result = result_by_path.get(str(Path(left_path).resolve()))
            right_result = result_by_path.get(str(Path(right_path).resolve()))
            if left_result is None or right_result is None:
                failure_text = "；".join(failures) if failures else "两张图像中至少有一张未成功完成分析"
                raise ValueError(f"双图对比需要两张图都分析成功: {failure_text}")

            report(total_images, total_images, "分析完成")
            return {
                'mode': 'pair',
                'left_result': left_result,
                'right_result': right_result,
                'note': self._build_compare_note('pair'),
            }

        raise ValueError(f"Unsupported compare-analysis mode: {request.get('mode')}")

    def _start_compare_analysis(self, request: dict) -> None:
        """Submit a compare-analysis request to the dedicated background executor."""
        self._discard_compare_analysis_state(include_completed=True, notify=False)
        request = {
            **dict(request),
            'context': request.get('context') or self._build_compare_analysis_context(),
        }

        total_images = int(request.get('total_images', 0))
        progress_state = self._make_compare_progress_state(total_images)
        self._compare_token += 1
        task_token = self._compare_token
        self._compare_snapshot = {
            'request': dict(request),
            'progress_state': progress_state,
        }

        if self.comparison_panel is not None:
            self._select_center_tab('comparison')
            self.comparison_panel.show_progress()
            self.comparison_panel.update_progress(0, max(1, total_images), "准备开始...")

        self._set_compare_analysis_busy_state(True)
        self._compare_future = self._compare_executor.submit(
            self._run_compare_analysis_request,
            dict(request),
            progress_state,
        )
        self.root.after(80, self._poll_compare_analysis_result, task_token)

    def _poll_compare_analysis_result(self, task_token: int) -> None:
        """Poll a background compare-analysis task and update the UI from the main thread."""
        if task_token != getattr(self, '_compare_token', None):
            return

        future = getattr(self, '_compare_future', None)
        snapshot = getattr(self, '_compare_snapshot', None)
        if future is None or snapshot is None:
            return

        progress_state = snapshot.get('progress_state') or {}
        total = max(1, int(progress_state.get('total', 0) or 0))
        current = max(0, int(progress_state.get('current', 0) or 0))
        message = str(progress_state.get('message') or "准备开始...")
        if self.comparison_panel is not None:
            self.comparison_panel.show_progress()
            self.comparison_panel.update_progress(current, total, message)

        if not future.done():
            self.root.after(80, self._poll_compare_analysis_result, task_token)
            return

        try:
            payload = future.result()
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as exc:
            logger.exception("对比分析失败")
            messagebox.showerror("错误", f"对比分析失败: {exc}")
        except Exception as exc:
            logger.exception("对比分析失败")
            messagebox.showerror("错误", f"对比分析失败: {exc}")
        else:
            mode = str(payload.get('mode', '')).lower()
            if mode == 'group':
                self._show_group_comparison_window(
                    payload['base_group'],
                    payload['exp_group'],
                    payload.get('note'),
                    payload.get('failures'),
                )
            elif mode == 'pair':
                self._show_comparison_window(
                    payload['left_result'],
                    payload['right_result'],
                    "图像A",
                    "图像B",
                    payload.get('note'),
                )
        finally:
            if task_token == getattr(self, '_compare_token', None):
                self._compare_future = None
                self._compare_snapshot = None
                if self.comparison_panel is not None:
                    self.comparison_panel.hide_progress()
                self._set_compare_analysis_busy_state(False)

    def _compare_image_groups(self):
        """按组批量选择图像并进行组别对比"""
        initial_dir = self._get_compare_initial_dir()
        filetypes = self._get_supported_image_filetypes()

        base_paths = list(filedialog.askopenfilenames(
            title="选择 base组 图片（可多选）",
            initialdir=initial_dir,
            filetypes=filetypes,
        ))
        if not base_paths:
            return

        exp_initial_dir = str(Path(base_paths[0]).parent)

        exp_paths = list(filedialog.askopenfilenames(
            title="选择 实验组 图片（可多选）",
            initialdir=exp_initial_dir,
            filetypes=filetypes,
        ))
        if not exp_paths:
            return

        base_set = {str(Path(path).resolve()) for path in base_paths}
        exp_set = {str(Path(path).resolve()) for path in exp_paths}
        overlap = sorted(base_set & exp_set)
        if overlap:
            messagebox.showwarning("提示", "base组与实验组存在重复图片，请去除重复后再进行组别对比。")
            return

        try:
            self._start_compare_analysis({
                'mode': 'group',
                'base_paths': base_paths,
                'exp_paths': exp_paths,
                'total_images': len(base_paths) + len(exp_paths),
            })
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as e:
            logger.exception("组别对比启动失败")
            messagebox.showerror("错误", f"组别对比失败: {e}")
        except Exception as e:
            logger.exception("组别对比启动失败")
            messagebox.showerror("错误", f"组别对比失败: {e}")

    def _compare_two_images(self):
        """任选两张图，在同一识别条件下进行双图对比"""
        initial_dir = self._get_compare_initial_dir()
        filetypes = self._get_supported_image_filetypes()

        left_path = filedialog.askopenfilename(
            title="选择第一张图片",
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        if not left_path:
            return

        right_initial_dir = str(Path(left_path).parent)

        right_path = filedialog.askopenfilename(
            title="选择第二张图片",
            initialdir=right_initial_dir,
            filetypes=filetypes,
        )
        if not right_path:
            return

        if Path(left_path).resolve() == Path(right_path).resolve():
            messagebox.showwarning("提示", "两次选择的是同一张图片，请重新选择两张不同的图像进行对比。")
            return

        try:
            self._start_compare_analysis({
                'mode': 'pair',
                'left_path': left_path,
                'right_path': right_path,
                'total_images': 2,
            })
        except GUI_EXPECTED_ANALYSIS_EXCEPTIONS as e:
            logger.exception("双图对比启动失败")
            messagebox.showerror("错误", f"双图对比失败: {e}")
        except Exception as e:
            logger.exception("双图对比启动失败")
            messagebox.showerror("错误", f"双图对比失败: {e}")

    # ===== 保存和导出 =====
    def _save_results(self):
        """保存分析结果"""
        measurements = self._get_active_measurements()

        if not measurements:
            self.image_panel.show_status("当前上下文没有可保存的检测结果")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("CSV文件", "*.csv")]
        )

        if file_path:
            try:
                stats = self.analyzer.get_statistics(self.current_roi)

                if file_path.endswith('.json'):
                    spatial = stats.get('spatial_distribution') or {}
                    data = {
                        'roi': self.current_roi.name if self.current_roi else "Full Image",
                        'statistics': {
                            'count': int(stats['count']),
                            'length_mean': float(stats['length_mean']),
                            'length_std': float(stats['length_std']),
                            'length_min': float(stats['length_min']),
                            'length_max': float(stats['length_max']),
                            'scale_um_per_pixel': float(self.analyzer.scale_um_per_pixel),
                            'spatial_distribution': {
                                'grid_size': int(spatial.get('grid_size', 0)),
                                'nearest_neighbor_cv': float(spatial.get('nearest_neighbor_cv', 0.0)),
                                'nearest_neighbor_index': float(spatial.get('nearest_neighbor_index', 0.0)),
                                'grid_density_cv': float(spatial.get('grid_density_cv', 0.0)),
                                'grid_entropy': float(spatial.get('grid_entropy', 0.0)),
                                'occupancy_ratio': float(spatial.get('occupancy_ratio', 0.0)),
                                'morans_i': float(spatial.get('morans_i', 0.0)),
                                'uniformity_scores': spatial.get('uniformity_scores', {}),
                                'density_grid': spatial.get('density_grid', []),
                            },
                        },
                    'measurements': [
                            {
                                'id': int(m.id),
                                'length_um': float(m.length_um),
                                'width_mean_um': float(m.width_mean_um) if m.width_mean_um else None,
                                'width_median_um': float(m.width_median_um) if m.width_median_um else None,
                                'width_iqr_um': float(m.width_iqr_um) if m.width_iqr_um else None,
                                'slenderness': float(m.slenderness) if m.slenderness else None
                            }
                            for m in measurements
                        ]
                    }
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                elif file_path.endswith('.csv'):
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['ID', '长度(μm)', '宽度均值(μm)', '宽度中位数(μm)', '宽度IQR(μm)', '长宽比'])
                        for m in measurements:
                            writer.writerow([
                                m.id,
                                f"{m.length_um:.2f}",
                                f"{m.width_mean_um:.2f}" if m.width_mean_um else "N/A",
                                f"{m.width_median_um:.2f}" if m.width_median_um else "N/A",
                                f"{m.width_iqr_um:.2f}" if m.width_iqr_um else "N/A",
                                f"{m.slenderness:.2f}" if m.slenderness else "N/A"
                            ])

                messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")

            except (OSError, ValueError, TypeError) as e:
                logger.exception("保存结果失败")
                messagebox.showerror("错误", f"保存失败: {e}")
            except Exception as e:
                logger.exception("保存结果时发生未预期的错误")
                messagebox.showerror("错误", f"保存失败: {e}")

    def _export_report(self):
        """导出分析报告"""
        measurements = self._get_active_measurements()

        if not measurements:
            self.image_panel.show_status("当前上下文没有可导出的检测结果")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                stats = self.analyzer.get_statistics(self.current_roi)
                spatial = stats.get('spatial_distribution') or {}

                report = f"""
========================================
    CNT图像分析报告
========================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析区域: {self.current_roi.name if self.current_roi else '全图'}

----------------------------------------
    比例尺信息
----------------------------------------
比例尺: {self.analyzer.scale_um_per_pixel:.4f} μm/pixel

----------------------------------------
    统计摘要
----------------------------------------
检测到的CNT数量: {stats['count']}

长度统计:
  - 平均值: {stats['length_mean']:.2f} μm
  - 标准差: {stats['length_std']:.2f} μm
  - 最小值: {stats['length_min']:.2f} μm
  - 最大值: {stats['length_max']:.2f} μm
  - 范围: {stats['length_max'] - stats['length_min']:.2f} μm

长度分布:
"""
                for label, count in stats['length_distribution'].items():
                    percentage = (count / stats['count'] * 100) if stats['count'] > 0 else 0
                    report += f"  - {label}: {count}根 ({percentage:.1f}%)\n"

                if spatial:
                    report += f"""

空间分布均匀性:
  - 综合均匀性得分: {(spatial.get('uniformity_scores') or {}).get('overall', 0.0):.1f} / 100（越大越均匀）
  - 中心点最近邻CV: {spatial.get('nearest_neighbor_cv', 0.0):.3f}（越小越均匀）
  - 最近邻指数NNI: {spatial.get('nearest_neighbor_index', 0.0):.3f}（大于1更均匀）
  - {spatial.get('grid_size', 0)}×{spatial.get('grid_size', 0)}网格CNT数CV: {spatial.get('grid_density_cv', 0.0):.3f}（越小越均匀）
  - 空间熵: {spatial.get('grid_entropy', 0.0):.3f}（越大越均匀）
  - Moran's I: {spatial.get('morans_i', 0.0):.3f}（越大越聚集）
  - 网格占用率: {spatial.get('occupancy_ratio', 0.0):.1%}
"""

                report += """
----------------------------------------
    详细测量数据
----------------------------------------
ID      长度(μm)    宽度(μm)    长宽比
----------------------------------------
"""
                for m in measurements:
                    width_str = f"{m.width_mean_um:.2f}" if m.width_mean_um else "N/A"
                    slenderness_str = f"{m.slenderness:.2f}" if m.slenderness else "N/A"
                    report += f"{m.id:<8}{m.length_um:<12.2f}{width_str:<12}{slenderness_str}\n"

                report += """
========================================
            报告结束
========================================
"""

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                messagebox.showinfo("成功", f"报告已导出到:\n{file_path}")

            except (OSError, ValueError, TypeError) as e:
                logger.exception("导出报告失败")
                messagebox.showerror("错误", f"导出失败: {e}")
            except Exception as e:
                logger.exception("导出报告时发生未预期的错误")
                messagebox.showerror("错误", f"导出失败: {e}")
