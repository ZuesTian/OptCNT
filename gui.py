"""
GUI主控制器模块 - 负责协调各个面板和核心分析功能
"""
import json
import logging
import csv
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Tuple
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from models import ROIRegion, CNTMeasurement
from analyzer_core import CNTAnalyzer
from utils import (
    DEBOUNCE_DELAY_MS,
    CHART_REBUILD_DRAW_LIMIT,
    SCALE_BAR_DEFAULT_UM,
    CNT_BRIDGE_STRENGTH_DEFAULT,
    CNT_MERGE_DISTANCE_DEFAULT_PX,
    get_length_histogram_bins,
)
from widgets import SortableTreeview
from panels import ControlPanel, ImagePanel, ResultPanel, AdvancedAnalysisPanel, ComparisonAnalysisPanel

logger = logging.getLogger(__name__)


class CNTAnalyzerGUI:
    """CNT分析器图形界面主控制器"""

    # Modern Vibrant 风格配色方案 - 丰富色彩层次
    MODERN_COLORS = {
        'bg_primary': '#FAFBFC',       # 整体背景（极淡灰）
        'bg_secondary': '#FFFFFF',     # 卡片/面板背景（纯白）
        'bg_tertiary': '#F1F5F9',      # 控件背景
        'text_primary': '#1E293B',     # 主要文字（深灰）
        'text_secondary': '#64748B',   # 次要文字
        'text_muted': '#94A3B8',       # 提示文字
        'border': '#E2E8F0',           # 边框
        'border_light': '#CBD5E0',
        'separator': '#E2E8F0',
        'accent_primary': '#6366F1',   # 主强调色（靛蓝紫）
        'accent_primary_light': '#818CF8',
        'accent_primary_dark': '#4F46E5',
        'accent_secondary': '#8B5CF6', # 次强调色（紫色）
        'accent_tertiary': '#EC4899',  # 第三强调色（粉色）
        'accent_teal': '#14B8A6',      # 青色
        'accent_amber': '#F59E0B',     # 琥珀色
        'accent_rose': '#F43F5E',      # 玫瑰红
        'success': '#10B981',          # 绿色
        'warning': '#F59E0B',          # 橙色
        'error': '#EF4444',            # 红色
        'info': '#06B6D4',             # 青色
        'button_bg': '#FFFFFF',        # 按钮背景
        'button_active': '#EEF2FF',    # 按钮激活
        'input_bg': '#FFFFFF',         # 输入框背景
        'input_border': '#CBD5E0',     # 输入框边框
        'hover_bg': '#F1F5F9',         # 悬停背景
        'selected_bg': '#E0E7FF',      # 选中背景
        'gradient_start': '#6366F1',   # 渐变起始色
        'gradient_end': '#8B5CF6',     # 渐变结束色
        'card_shadow': '#E2E8F0',      # 卡片阴影
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CNT图像分析系统 - 现代化骨架预览版")

        # 应用Modern样式
        self._apply_modern_style()

        # 核心分析器
        self.analyzer = CNTAnalyzer()

        # 状态变量
        self.current_image = None
        self.photo = None
        self.current_roi: Optional[ROIRegion] = None
        self.roi_counter = 0
        self.zoom_level = 1.0
        self._preprocess_job = None
        self._layout_job = None
        self._layout_retry_count = 0
        self._comparison_layout_job = None
        self.main_paned: Optional[tk.PanedWindow] = None
        self.current_image_path: Optional[str] = None
        self._analysis_cache = OrderedDict()
        self._analysis_cache_limit = 48
        self._last_auto_suggest_result = None
        self._last_root_size: Optional[Tuple[int, int]] = None
        
        # 图表缓存
        self._charts = {
            'histogram': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
            'pie': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
            'cluster': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
            'heatmap': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
            'comparison': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
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

        # 快捷键：从剪贴板粘贴图像
        self.root.bind_all("<Control-v>", self._paste_image_from_clipboard)
        self.root.bind_all("<Control-V>", self._paste_image_from_clipboard)

    def _init_variables(self):
        """初始化Tkinter变量"""
        self.blur_kernel_var = tk.IntVar(value=9)
        self.adaptive_block_var = tk.IntVar(value=11)
        self.adaptive_c_var = tk.IntVar(value=3)
        self.bridge_strength_var = tk.IntVar(value=CNT_BRIDGE_STRENGTH_DEFAULT)
        self.min_length_um_var = tk.DoubleVar(value=4.0)
        self.max_length_um_var = tk.DoubleVar(value=200.0)
        self.min_slenderness_var = tk.DoubleVar(value=3.0)
        self.merge_distance_px_var = tk.IntVar(value=CNT_MERGE_DISTANCE_DEFAULT_PX)
        self.detect_profile_var = tk.StringVar(value="标准（推荐）")
        self.split_mode_var = tk.StringVar(value="不拆分")
        self.scale_pixels_var = tk.DoubleVar(value=0)
        self.scale_um_var = tk.DoubleVar(value=SCALE_BAR_DEFAULT_UM)
        self.live_preview_var = tk.BooleanVar(value=True)
        self.display_var = tk.StringVar(value="original")
        self._last_preprocess_signature = None

    def _apply_modern_style(self):
        """应用Modern风格样式"""
        c = self.MODERN_COLORS

        self.root.configure(bg=c['bg_primary'])
        style = ttk.Style()
        
        try:
            style.theme_use('clam')
        except tk.TclError:
            logger.debug("clam主题不可用，使用默认主题")

        default_font = ('Segoe UI', 9)
        heading_font = ('Segoe UI', 10, 'bold')
        
        style.configure('.',
                        background=c['bg_primary'],
                        foreground=c['text_primary'],
                        font=default_font)

        style.configure('TFrame', background=c['bg_primary'])
        style.configure('Card.TFrame', background=c['bg_secondary'])

        style.configure('TLabel', background=c['bg_primary'], foreground=c['text_primary'], font=default_font)
        style.configure('Card.TLabel', background=c['bg_secondary'], foreground=c['text_primary'], font=default_font)
        style.configure('Header.TLabel', background=c['bg_secondary'], foreground=c['accent_primary'], font=('Segoe UI', 11, 'bold'))
        style.configure('Secondary.TLabel', background=c['bg_primary'], foreground=c['text_secondary'], font=('Segoe UI', 9, 'italic'))

        style.configure('TButton',
                        background=c['button_bg'],
                        foreground=c['accent_primary'],
                        borderwidth=1,
                        relief='flat',
                        font=('Segoe UI', 9, 'bold'),
                        padding=5)
        
        style.map('TButton',
                  background=[('active', c['button_active']),
                              ('pressed', c['accent_primary'])],
                  foreground=[('active', c['accent_primary_dark']),
                              ('pressed', '#FFFFFF')],
                  relief=[('pressed', 'flat')])
        
        style.configure('Accent.TButton',
                        background=c['accent_primary'],
                        foreground='#FFFFFF',
                        borderwidth=0,
                        relief='flat',
                        font=('Segoe UI', 9, 'bold'),
                        padding=6)
        
        style.map('Accent.TButton',
                  background=[('active', c['accent_primary_light']),
                              ('pressed', c['accent_primary_dark'])],
                  foreground=[('active', '#FFFFFF'),
                              ('pressed', '#FFFFFF')])

        style.configure('Success.TButton',
                        background=c['success'],
                        foreground='#FFFFFF',
                        borderwidth=0,
                        relief='flat',
                        font=('Segoe UI', 9, 'bold'),
                        padding=6)
        
        style.map('Success.TButton',
                  background=[('active', '#059669'),
                              ('pressed', '#047857')])

        style.configure('Warning.TButton',
                        background=c['warning'],
                        foreground='#FFFFFF',
                        borderwidth=0,
                        relief='flat',
                        font=('Segoe UI', 9, 'bold'),
                        padding=6)
        
        style.map('Warning.TButton',
                  background=[('active', '#D97706'),
                              ('pressed', '#B45309')])

        style.configure('Danger.TButton',
                        background=c['error'],
                        foreground='#FFFFFF',
                        borderwidth=0,
                        relief='flat',
                        font=('Segoe UI', 9, 'bold'),
                        padding=6)
        
        style.map('Danger.TButton',
                  background=[('active', '#DC2626'),
                              ('pressed', '#B91C1C')])

        style.configure('TEntry',
                        fieldbackground=c['input_bg'],
                        foreground=c['text_primary'],
                        borderwidth=1,
                        relief='solid',
                        padding=5)
        
        style.configure('TLabelframe',
                        background=c['bg_primary'],
                        borderwidth=1,
                        relief='solid',
                        bordercolor=c['border'])
        
        style.configure('TLabelframe.Label',
                        background=c['bg_primary'],
                        foreground=c['accent_secondary'],
                        font=heading_font)

        style.configure('TNotebook', background=c['bg_primary'], tabmargins=[2, 5, 2, 0], borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=c['bg_tertiary'],
                        foreground=c['text_secondary'],
                        padding=[15, 8],
                        font=('Segoe UI', 9),
                        borderwidth=0)
        
        style.map('TNotebook.Tab',
                  background=[('selected', c['bg_secondary']),
                              ('active', c['hover_bg'])],
                  foreground=[('selected', c['accent_primary']),
                              ('active', c['text_primary'])],
                  expand=[('selected', [1, 1, 1, 0])])

        style.configure('TScale', background=c['bg_primary'], troughcolor=c['border'], sliderlength=20)
        
        style.configure('TScrollbar', 
                        background=c['bg_tertiary'], 
                        troughcolor=c['bg_primary'], 
                        borderwidth=0,
                        arrowsize=12)
        style.map('TScrollbar',
                  background=[('active', c['accent_teal']), 
                              ('pressed', c['accent_primary'])])

        style.configure('TCheckbutton', background=c['bg_primary'], foreground=c['text_primary'], font=default_font)
        style.configure('TRadiobutton', background=c['bg_primary'], foreground=c['text_primary'], font=default_font)
        
        style.map('TCheckbutton', background=[('active', c['bg_primary'])])
        style.map('TRadiobutton', background=[('active', c['bg_primary'])])

        style.configure('Treeview',
                        background=c['bg_secondary'],
                        foreground=c['text_primary'],
                        fieldbackground=c['bg_secondary'],
                        borderwidth=0,
                        font=default_font,
                        rowheight=28)
        
        style.configure('Treeview.Heading',
                        background=c['bg_tertiary'],
                        foreground=c['text_secondary'],
                        font=('Segoe UI', 9, 'bold'),
                        borderwidth=0,
                        relief='flat')
        
        style.map('Treeview.Heading',
                  background=[('active', c['hover_bg'])],
                  foreground=[('active', c['accent_primary'])])

        style.configure('Horizontal.TProgressbar',
                        background=c['accent_primary'],
                        troughcolor=c['bg_tertiary'])

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

    def _get_pane_layout_profile(self, tab_key: Optional[str] = None) -> dict:
        """根据当前标签页返回三栏布局配置。"""
        active_key = tab_key or self._get_active_center_tab_key()
        profiles = {
            'image': {
                'left_ratio': 0.20,
                'right_ratio': 0.20,
                'left_floor': 260,
                'right_floor': 220,
                'center_target': 620,
            },
            'analysis': {
                'left_ratio': 0.19,
                'right_ratio': 0.18,
                'left_floor': 250,
                'right_floor': 220,
                'center_target': 680,
            },
            'comparison': {
                'left_ratio': 0.17,
                'right_ratio': 0.15,
                'left_floor': 240,
                'right_floor': 200,
                'center_target': 760,
            },
        }
        return profiles.get(active_key, profiles['image'])

    def _calculate_pane_widths(self, total_w: int, tab_key: Optional[str] = None) -> Tuple[int, int, int]:
        """根据当前可用宽度和活动标签页计算左/中/右三栏宽度。"""
        profile = self._get_pane_layout_profile(tab_key)
        total_w = max(1, int(total_w))
        left_floor = int(profile['left_floor'])
        right_floor = int(profile['right_floor'])
        left_w = max(left_floor, int(total_w * float(profile['left_ratio'])))
        right_w = max(right_floor, int(total_w * float(profile['right_ratio'])))
        center_target = min(
            int(profile['center_target']),
            max(420, total_w - left_floor - right_floor),
        )

        center_w = total_w - left_w - right_w
        if center_w < center_target:
            shortage = center_target - center_w
            left_reducible = max(0, left_w - left_floor)
            reduce_left = min(shortage // 2, left_reducible)
            left_w -= reduce_left
            shortage -= reduce_left

            right_reducible = max(0, right_w - right_floor)
            reduce_right = min(shortage, right_reducible)
            right_w -= reduce_right

        center_w = max(420, total_w - left_w - right_w)
        if left_w + center_w + right_w != total_w:
            right_w = max(right_floor, total_w - left_w - center_w)

        return left_w, center_w, right_w

    def _optimize_window_distribution(self):
        """自适应优化窗口分布，优先保证中间图像区域"""
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
        tab_key = self._get_active_center_tab_key()
        profile = self._get_pane_layout_profile(tab_key)
        panes = paned.panes()
        if len(panes) >= 3:
            try:
                paned.paneconfigure(panes[0], minsize=int(profile['left_floor']))
                paned.paneconfigure(panes[1], minsize=420)
                paned.paneconfigure(panes[2], minsize=int(profile['right_floor']))
            except tk.TclError:
                pass

        left_w, _, right_w = self._calculate_pane_widths(total_w, tab_key)

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

        ttk.Button(button_frame, text="📂 打开图像", style='Accent.TButton',
                   command=self._open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📋 粘贴图像", style='Accent.TButton',
                   command=self._paste_image_from_clipboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="💾 保存结果", style='Success.TButton',
                   command=self._save_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="📊 导出报告", style='Warning.TButton',
                   command=self._export_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🔬 对比分析", style='Accent.TButton',
                   command=self._open_compare_mode_dialog).pack(side=tk.LEFT, padx=2)
        
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
        """在切换到对比分析页后，按可用宽度重新贴合图表。"""
        if event.widget is self.center_notebook:
            self._schedule_window_distribution(delay_ms=60)
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

    # ===== 文件操作 =====
    def _load_image_common(self):
        """加载图像后的通用流程"""
        self._reset_display()
        self._update_display()

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
        self._update_results()
        self._refresh_scale_status_ui()
        self._refresh_analysis_status_ui()

    # ===== 比例尺操作 =====
    def _select_scale_on_image(self):
        """在图像上选择比例尺"""
        if self.analyzer.image is None:
            messagebox.showwarning("警告", "请先打开图像！")
            return

        def on_scale_selected(length):
            # 画布像素 → 原图像素（消除缩放影响）
            real_length = length / self.zoom_level
            self.scale_pixels_var.set(real_length)
            messagebox.showinfo("比例尺选择",
                                f"已选择比例尺长度: {real_length:.1f}像素\n"
                                "请输入对应的微米数并点击'应用比例尺'")

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
            self.analyzer.record_manual_scale(pixels, micrometers, source='manual', confidence='high')
            new_scale = self.analyzer.scale_um_per_pixel
            
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
            
            # 刷新结果显示
            self._update_results()
            self._update_display()
            
            messagebox.showinfo("成功", "比例尺已应用，测量结果已更新！")

        except (ValueError, tk.TclError) as e:
            logger.exception("应用比例尺失败")
            messagebox.showerror("错误", f"应用比例尺失败: {e}")
        except Exception as e:
            logger.exception("应用比例尺时发生未预期的错误")
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

    # ===== ROI操作 =====
    def _select_roi(self):
        """选择ROI"""
        if self.analyzer.image is None:
            messagebox.showwarning("警告", "请先打开图像！")
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
            else:
                self._update_display()

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
            else:
                self._update_display()
            self._update_results()

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
            else:
                self._update_display()
            self._update_results()

    def _clear_all_rois(self):
        """清空所有ROI"""
        self.analyzer.clear_rois()
        self.control_panel.clear_roi_list()
        self.current_roi = None
        self._last_preprocess_signature = None
        if self._is_preprocess_mode():
            self._apply_preprocessing(force=True)
        else:
            self._update_display()
        self._update_results()

    # ===== 自适应参数推荐 =====
    def _get_detection_profile_key(self) -> str:
        """将中文检测风格映射为核心算法配置键"""
        return {
            "严格（少误检）": "precision",
            "标准（推荐）": "balanced",
            "敏感（少漏检）": "recall",
        }.get(self.detect_profile_var.get(), "balanced")

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
        except Exception as e:
            self._last_auto_suggest_result = None
            logger.debug(f"自适应参数推荐失败，使用默认值: {e}")
            return None

    def _on_reapply_auto_suggest(self):
        """手动触发一次自动参数推荐"""
        if self.analyzer.image is None:
            messagebox.showwarning("警告", "请先打开图像！")
            return
        params = self._auto_suggest_params()
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        elif self._is_preprocess_mode():
            self._apply_preprocessing(force=True)
        if params:
            self.image_panel.show_status(
                f"已重新推荐参数: {params['blur_kernel']}/{params['adaptive_block']}/{params['adaptive_c']}"
            )
        else:
            self.image_panel.show_status("自动推荐失败，已保留当前参数")

    def _on_profile_change(self, event=None):
        """识别策略变化时刷新推荐参数和状态"""
        if self.analyzer.image is None:
            return
        self._auto_suggest_params()
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._refresh_analysis_status_ui()

    def _on_split_mode_change(self, event=None):
        """粘连拆分模式变化时刷新状态"""
        self._refresh_analysis_status_ui()

    # ===== 预处理参数 =====
    def _is_preprocess_mode(self) -> bool:
        """当前显示模式是否需要预处理结果"""
        return self.display_var.get() in ("binary", "skeleton_preview")

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
        return (
            int(self.blur_kernel_var.get()),
            int(self.adaptive_block_var.get()),
            int(self.adaptive_c_var.get()),
            int(self.bridge_strength_var.get()),
            True,  # threshold_invert
            roi_signature,
        )

    def _needs_preprocessing(self) -> bool:
        """判断当前参数/ROI是否需要重新预处理"""
        if self.analyzer.binary_image is None:
            return True
        return self._get_preprocess_signature() != self._last_preprocess_signature

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

            if not force and self.analyzer.binary_image is not None and signature == self._last_preprocess_signature:
                self._update_display()
                return

            self.analyzer.preprocess(
                blur_kernel=blur_kernel,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                bridge_strength=int(self.bridge_strength_var.get()),
                threshold_invert=True,
                roi=roi_to_use
            )
            self._last_preprocess_signature = signature
            self._update_display()
        except (ValueError, cv2.error) as e:
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
        self._refresh_analysis_status_ui()

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
        self._refresh_analysis_status_ui()

    def _on_c_change(self, value):
        """自适应常数C变化"""
        val = int(float(value))
        self.adaptive_c_var.set(val)
        self.control_panel.update_c_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._refresh_analysis_status_ui()

    def _on_bridge_change(self, value):
        """桥接强度变化"""
        val = int(float(value))
        self.bridge_strength_var.set(val)
        self.control_panel.update_bridge_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()
        self._refresh_analysis_status_ui()

    def _on_merge_distance_change(self, value):
        """近邻合并距离变化"""
        val = int(float(value))
        self.merge_distance_px_var.set(val)
        self.control_panel.update_merge_distance_label(str(val))
        self._refresh_analysis_status_ui()

    # ===== CNT检测 =====
    def _detect_cnt(self):
        """检测CNT"""
        if self.analyzer.image is None:
            messagebox.showwarning("警告", "请先打开图像！")
            return

        try:
            min_length = self.min_length_um_var.get()
            max_length = self.max_length_um_var.get()
            min_slenderness = self.min_slenderness_var.get()
            merge_distance_px = self.merge_distance_px_var.get()

            # 修复1: 强制校验并重算预处理，确保二值图与当前ROI一致
            current_signature = self._get_preprocess_signature()
            if self.analyzer.binary_image is None or current_signature != self._last_preprocess_signature:
                self._apply_preprocessing(force=True)

            measurements = self.analyzer.detect_cnts_hybrid(
                min_length_um=min_length,
                max_length_um=max_length,
                min_slenderness=min_slenderness,
                detection_profile=self._get_detection_profile_key(),
                split_mode={
                    "不拆分": "off",
                    "标准拆分": "conservative",
                    "强力拆分": "aggressive",
                }.get(self.split_mode_var.get(), self.split_mode_var.get()),
                merge_distance_px=float(merge_distance_px),
                roi=self.current_roi
            )

            self._update_results()
            self._update_advanced_analysis()
            self._update_display()

            roi_text = f" ({self.current_roi.name})" if self.current_roi else ""
            messagebox.showinfo("检测完成",
                                f"在{roi_text if self.current_roi else '全图'}中检测到 {len(measurements)} 个CNT")

        except (ValueError, cv2.error, tk.TclError) as e:
            logger.exception("CNT检测失败")
            messagebox.showerror("错误", f"CNT检测失败: {e}")
        except Exception as e:
            logger.exception("CNT检测时发生未预期的错误")
            messagebox.showerror("错误", f"发生未预期的错误: {e}")

    # ===== 显示更新 =====
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
                if self.analyzer.binary_image is not None:
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
                    image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                else:
                    return

            elif mode == "skeleton_preview":
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

        measurements = self.current_roi.measurements if self.current_roi else self.analyzer.measurements
        if not measurements:
            return

        stats = self.analyzer.get_statistics(self.current_roi)

        text_widget = self.result_panel.stats_text
        
        text_widget.insert(tk.END, "检测到的CNT数量: ", 'header')
        text_widget.insert(tk.END, f"{stats['count']}\n\n", 'value')
        
        text_widget.insert(tk.END, "===== 长度统计 (μm) =====\n", 'header')
        text_widget.insert(tk.END, "平均值: ", 'header')
        text_widget.insert(tk.END, f"{stats['length_mean']:.2f}\n", 'value')
        text_widget.insert(tk.END, "标准差: ", 'header')
        text_widget.insert(tk.END, f"{stats['length_std']:.2f}\n", 'value')
        text_widget.insert(tk.END, "最小值: ", 'header')
        text_widget.insert(tk.END, f"{stats['length_min']:.2f}\n", 'value')
        text_widget.insert(tk.END, "最大值: ", 'header')
        text_widget.insert(tk.END, f"{stats['length_max']:.2f}\n\n", 'value')
        
        text_widget.insert(tk.END, "===== 长度分布 =====\n", 'header')
        for label, count in stats['length_distribution'].items():
            text_widget.insert(tk.END, f"{label}: ", 'header')
            text_widget.insert(tk.END, f"{count}根\n", 'value')

        # 宽度鲁棒统计汇总
        widths_median = [m.width_median_um for m in measurements if m.width_median_um]
        if widths_median:
            text_widget.insert(tk.END, "\n===== 宽度统计 (μm) =====\n", 'header')
            text_widget.insert(tk.END, "中位数均值: ", 'header')
            text_widget.insert(tk.END, f"{np.mean(widths_median):.3f}\n", 'value')
            widths_iqr = [m.width_iqr_um for m in measurements if m.width_iqr_um]
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
            self.result_panel.add_measurement((m.id, f"{m.length_um:.2f}"))

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
        measurements = self.current_roi.measurements if self.current_roi else self.analyzer.measurements

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
        measurements = self.current_roi.measurements if self.current_roi else self.analyzer.measurements
        if not measurements:
            return

        stats = self.analyzer.get_statistics(self.current_roi)

        # 绘制图表
        self._draw_distribution_chart(measurements)
        self._draw_pie_chart(stats['length_distribution'])
        self._draw_cluster_analysis(measurements)
        self._draw_spatial_heatmap(stats.get('spatial_distribution') or {})
        
        # 强制刷新布局
        self.analysis_panel.refresh_layout()

    def _dispose_chart(self, key: str):
        """销毁图表对象，避免长时间运行时累积旧 Figure。"""
        chart = self._charts[key]
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
            chart['fig'] = Figure(figsize=figsize, dpi=100)
            chart['fig'].patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            chart['ax'] = chart['fig'].add_subplot(111)
            chart['canvas'] = FigureCanvasTkAgg(chart['fig'], master=frame)
            chart['canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        elif chart['ax'] is not None:
            chart['ax'].clear()

        chart['draw_count'] = chart.get('draw_count', 0) + 1
        return chart

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

            bins = get_length_histogram_bins(lengths)

            counts, _, _ = ax.hist(
                lengths,
                bins=bins,
                edgecolor='white',
                alpha=0.8,
                color=self.MODERN_COLORS['accent_primary']
            )

            # 若数据全部未落入分箱（极端边界情况下），给出明确提示
            if np.sum(counts) == 0:
                ax.text(0.5, 0.5, "当前分箱下无可视柱形，请检查比例尺或过滤参数",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['warning'])

            ax.set_xlabel('长度 (μm)', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_ylabel('数量', fontsize=9, color=self.MODERN_COLORS['text_secondary'])

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

        except Exception as e:
            logger.exception(f"绘制聚类图错误: {e}")

    def _draw_spatial_heatmap(self, spatial: dict):
        """绘制CNT空间分布热图"""
        try:
            chart = self._init_chart('heatmap')
            ax = chart['ax']
            canvas = chart['canvas']
            if not canvas:
                return

            density_grid = np.array(spatial.get('density_grid') or [])
            if density_grid.size == 0:
                ax.text(0.5, 0.5, "暂无空间分布数据",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=self.MODERN_COLORS['text_muted'])
                canvas.draw()
                return

            heatmap = ax.imshow(density_grid, cmap='YlOrRd', interpolation='nearest')
            ax.set_title('CNT网格数量热图', fontsize=10, color=self.MODERN_COLORS['text_primary'])
            ax.set_xlabel('X网格', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.set_ylabel('Y网格', fontsize=9, color=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])

            if 'colorbar' not in chart or chart['colorbar'] is None:
                chart['colorbar'] = chart['fig'].colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            else:
                chart['colorbar'].update_normal(heatmap)
            chart['colorbar'].ax.tick_params(colors=self.MODERN_COLORS['text_secondary'])
            chart['colorbar'].set_label('每格CNT数量', color=self.MODERN_COLORS['text_secondary'])

            chart['fig'].tight_layout()
            canvas.draw()

        except Exception as e:
            logger.exception(f"绘制空间热图错误: {e}")

    def _get_current_analysis_settings(self) -> Tuple[dict, dict]:
        """读取当前界面的识别条件，确保多张图在同一条件下比较"""
        preprocess_settings = {
            'blur_kernel': int(self.blur_kernel_var.get()),
            'adaptive_block': int(self.adaptive_block_var.get()),
            'adaptive_c': int(self.adaptive_c_var.get()),
            'bridge_strength': int(self.bridge_strength_var.get()),
            'threshold_invert': True,
        }
        detect_settings = {
            'min_length_um': float(self.min_length_um_var.get()),
            'max_length_um': float(self.max_length_um_var.get()),
            'min_slenderness': float(self.min_slenderness_var.get()),
            'detection_profile': self._get_detection_profile_key(),
            'merge_distance_px': float(self.merge_distance_px_var.get()),
            'split_mode': {
                "不拆分": "off",
                "标准拆分": "conservative",
                "强力拆分": "aggressive",
            }.get(self.split_mode_var.get(), self.split_mode_var.get()),
            'roi': None,
        }
        return preprocess_settings, detect_settings

    def _build_analysis_context(self) -> dict:
        """快照当前分析参数，避免重复读取界面变量"""
        preprocess_settings, detect_settings = self._get_current_analysis_settings()
        scale_um = float(self.scale_um_var.get()) if self.scale_um_var.get() > 0 else float(SCALE_BAR_DEFAULT_UM)
        manual_scale_pixels = float(self.scale_pixels_var.get()) if self.scale_pixels_var.get() > 0 else 0.0
        return {
            'preprocess_settings': preprocess_settings,
            'detect_settings': detect_settings,
            'scale_um': scale_um,
            'manual_scale_pixels': manual_scale_pixels,
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

    def _get_cached_analysis_result(self, cache_key: tuple) -> Optional[dict]:
        """读取缓存结果，并刷新其 LRU 顺序"""
        cached = self._analysis_cache.get(cache_key)
        if cached is None:
            return None
        self._analysis_cache.move_to_end(cache_key)
        return cached

    def _store_analysis_result(self, cache_key: tuple, result: dict) -> dict:
        """写入分析缓存，并限制缓存大小"""
        self._analysis_cache[cache_key] = result
        self._analysis_cache.move_to_end(cache_key)
        while len(self._analysis_cache) > self._analysis_cache_limit:
            self._analysis_cache.popitem(last=False)
        return result

    def _run_image_analysis(self, image_path: str, context: dict, include_visualization: bool = False) -> dict:
        """执行单张图片分析，支持缓存复用"""
        analyzer = CNTAnalyzer()
        analyzer.load_image(image_path)

        scale_um = float(context['scale_um'])
        manual_scale_pixels = float(context['manual_scale_pixels'])
        scale_result = analyzer.apply_detected_scale(default_micrometers=scale_um)
        scale_info = scale_result.get('scale_info')
        if not scale_result.get('applied') and manual_scale_pixels > 0 and scale_um > 0:
            analyzer.record_manual_scale(manual_scale_pixels, scale_um, source='batch_manual', confidence='low')

        analyzer.preprocess(**context['preprocess_settings'])
        analyzer.detect_cnts_hybrid(**context['detect_settings'])
        stats = analyzer.get_statistics()

        visualization = None
        if include_visualization:
            visualization = cv2.cvtColor(analyzer.get_visualization(), cv2.COLOR_BGR2RGB)

        return {
            'path': image_path,
            'name': Path(image_path).name,
            'stats': stats,
            'scale_info': scale_info,
            'scale_status': analyzer.get_scale_status(),
            'visualization': visualization,
        }

    def _analyze_image_file(self, image_path: str, include_visualization: bool = False) -> dict:
        """在不影响当前主界面的前提下分析单张图像"""
        context = self._build_analysis_context()
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

        result = self._run_image_analysis(image_path, context, include_visualization=include_visualization)
        stored = self._store_analysis_result(cache_key, result)
        if include_visualization:
            self._store_analysis_result(base_cache_key, stored)
        return stored

    def _analyze_image_files(self, image_paths: List[str], group_label: str) -> Tuple[List[dict], List[str]]:
        """批量分析一组图像，返回成功结果与失败信息"""
        results: List[dict] = []
        failures: List[str] = []

        for image_path in image_paths:
            try:
                results.append(self._analyze_image_file(image_path))
            except Exception as e:
                failures.append(f"{Path(image_path).name}: {e}")

        if not results:
            raise ValueError(f"{group_label}未成功分析任何图像")

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
        length_mean_values: List[float] = []
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
        density_grids: List[np.ndarray] = []
        file_details: List[dict] = []

        for result in results:
            stats = result.get('stats', {})
            spatial = stats.get('spatial_distribution') or {}
            uniformity_scores = spatial.get('uniformity_scores') or {}
            aggregation_scores = spatial.get('aggregation_scores') or {}

            count = float(stats.get('count', 0))
            length_mean = float(stats.get('length_mean', 0.0))
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

            count_values.append(count)
            length_mean_values.append(length_mean)
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

            density_grid = np.array(spatial.get('density_grid') or np.zeros((10, 10)), dtype=float)
            density_grids.append(density_grid)

            file_details.append({
                'name': result['name'],
                'count': count,
                'length_mean': length_mean,
                'nearest_neighbor_cv': nn_cv,
                'nearest_neighbor_index': nn_index,
                'grid_density_cv': grid_cv,
                'morans_i': morans_i,
                'aggregation_nn_score': aggregation_nn,
                'aggregation_grid_score': aggregation_grid,
                'aggregation_moran_score': aggregation_moran,
                'aggregation_score': aggregation_overall,
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
            'length_mean_stats': self._summarize_numeric_series(length_mean_values),
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
            t_stat, t_pvalue = scipy_stats.ttest_ind(base, exp, equal_var=False, nan_policy='omit')
            result['t_stat'] = float(t_stat) if np.isfinite(t_stat) else None
            result['t_pvalue'] = float(t_pvalue) if np.isfinite(t_pvalue) else None
        except (TypeError, ValueError, FloatingPointError) as exc:
            logger.debug("Welch t-test failed for comparison inputs: %s", exc)

        try:
            mw_stat, mw_pvalue = scipy_stats.mannwhitneyu(base, exp, alternative='two-sided')
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

        count_mean = group_summary['count_stats']['mean']
        count_std = max(group_summary['count_stats']['std'], 1.0)
        uniformity_mean = group_summary['spatial_stats']['uniformity_score']['mean']
        uniformity_std = max(group_summary['spatial_stats']['uniformity_score']['std'], 1.0)

        def score(detail: dict) -> float:
            return (
                abs(detail['count'] - count_mean) / count_std +
                abs(detail.get('uniformity_score', 0.0) - uniformity_mean) / uniformity_std
            )

        representative_detail = min(details, key=score)
        representative_path = next(
            result['path'] for result in group_summary['results']
            if result['name'] == representative_detail['name']
        )
        return self._analyze_image_file(representative_path, include_visualization=True)

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

    def _format_comparison_summary(self,
                                   left_result: dict,
                                   right_result: dict,
                                   left_label: str,
                                   right_label: str,
                                   note: Optional[str] = None) -> str:
        """生成双图对比摘要"""
        left_spatial = left_result['stats'].get('spatial_distribution') or {}
        right_spatial = right_result['stats'].get('spatial_distribution') or {}
        left_uniformity = left_spatial.get('uniformity_scores') or {}
        right_uniformity = right_spatial.get('uniformity_scores') or {}

        left_count = int(left_result['stats']['count'])
        right_count = int(right_result['stats']['count'])
        count_diff = left_count - right_count
        count_ratio = (count_diff / right_count * 100.0) if right_count > 0 else 0.0

        left_uniformity_score = float(left_uniformity.get('overall', 0.0))
        right_uniformity_score = float(right_uniformity.get('overall', 0.0))
        uniformity_diff = left_uniformity_score - right_uniformity_score
        left_more_uniform = uniformity_diff > 1.0
        right_more_uniform = uniformity_diff < -1.0
        right_more_clustered = right_spatial.get('morans_i', 0) > left_spatial.get('morans_i', 0)

        split_mode_label = self.split_mode_var.get()
        profile_label = self.detect_profile_var.get()
        lines = []
        if note:
            lines.append(note)
            lines.append("")

        lines.extend([
            f"相同识别条件: 模糊={self.blur_kernel_var.get()} / 自适应块={self.adaptive_block_var.get()} / C={self.adaptive_c_var.get()} / 桥接={self.bridge_strength_var.get()} / 最小长度={self.min_length_um_var.get():.1f}μm / 最小长宽比={self.min_slenderness_var.get():.1f} / 检测风格={profile_label} / 拆分模式={split_mode_label} / 合并距离={self.merge_distance_px_var.get()}px",
            f"{left_label}: {left_result['name']}，识别到 {left_count} 根CNT",
            f"{right_label}: {right_result['name']}，识别到 {right_count} 根CNT",
        ])

        if count_diff >= 0:
            lines.append(f"CNT数量差异: {left_label}多 {count_diff} 根（+{count_ratio:.1f}%）")
        else:
            lines.append(f"CNT数量差异: {right_label}多 {abs(count_diff)} 根（+{abs(count_ratio):.1f}%）")

        lines.extend([
            "",
            "分布均匀性分析:",
            f"综合均匀性得分: {left_label} {left_uniformity_score:.1f}，{right_label} {right_uniformity_score:.1f}。该得分范围 0-100，越大越均匀。",
            f"方法一 中心点最近邻CV: {left_label} {left_spatial.get('nearest_neighbor_cv', 0):.3f}，{right_label} {right_spatial.get('nearest_neighbor_cv', 0):.3f}。该值越小越均匀。",
            f"补充指标 最近邻指数NNI: {left_label} {left_spatial.get('nearest_neighbor_index', 0):.3f}，{right_label} {right_spatial.get('nearest_neighbor_index', 0):.3f}。该值大于 1 表示更均匀。",
            f"方法二 网格CNT数CV: {left_label} {left_spatial.get('grid_density_cv', 0):.3f}，{right_label} {right_spatial.get('grid_density_cv', 0):.3f}。该值越小越均匀。",
            f"方法三 Moran's I: {left_label} {left_spatial.get('morans_i', 0):.3f}，{right_label} {right_spatial.get('morans_i', 0):.3f}。该值越大越聚集。",
            "",
        ])

        if left_count > right_count and left_more_uniform:
            conclusion = f"结论: 在相同识别条件下，{left_label}识别CNT数量更多，且分布更均匀。"
            if right_more_clustered:
                conclusion += f" {right_label}的空间自相关更高，CNT团聚更明显。"
        elif left_count < right_count and right_more_uniform:
            conclusion = f"结论: 在相同识别条件下，{right_label}识别CNT数量更多，且分布更均匀。"
            if not right_more_clustered:
                conclusion += f" {left_label}的空间自相关更高，CNT团聚更明显。"
        else:
            conclusion = "结论: 当前这组参数下，两张图的数量和均匀性差异没有同时完全拉开，可继续微调识别条件。"
        lines.append(conclusion)

        return "\n".join(lines)

    def _select_center_tab(self, tab_key: str):
        """切换到指定的中间页签"""
        if not hasattr(self, 'center_notebook'):
            return

        tab = self._center_tabs.get(tab_key)
        if tab is not None:
            self.center_notebook.select(tab)

    def _apply_standard_axes_style(self, axes: List):
        """统一普通图表坐标轴样式"""
        for ax in axes:
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            ax.tick_params(axis='x', colors=self.MODERN_COLORS['text_secondary'])
            ax.tick_params(axis='y', colors=self.MODERN_COLORS['text_secondary'])
            for spine in ax.spines.values():
                spine.set_color(self.MODERN_COLORS['border'])

    def _get_comparison_image_aspect(self, image: Optional[np.ndarray]) -> float:
        """返回对比图像的宽高比"""
        if image is None or getattr(image, 'size', 0) == 0:
            return 1.0

        height, width = image.shape[:2]
        if height <= 0:
            return 1.0
        return width / height

    def _should_stack_comparison_images(self,
                                        *images: Optional[np.ndarray],
                                        threshold: float = 1.3) -> bool:
        """宽图优先采用上下排布，避免代表图被压扁"""
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
        """根据图像宽高比动态分配图表与图像区域。"""
        aspects = [
            self._get_comparison_image_aspect(image)
            for image in images
            if image is not None and getattr(image, 'size', 0) > 0
        ]
        avg_aspect = (sum(aspects) / len(aspects)) if aspects else 1.0
        max_aspect = max(aspects) if aspects else 1.0

        if variant == 'pair':
            if stacked:
                wide_boost = max(0.0, max(max_aspect, avg_aspect) - 1.3)
                image_ratio = min(1.75, 1.20 + wide_boost * 0.55)
                top_ratio = max(0.60, 0.78 - wide_boost * 0.18)
                figure_height = 11.8 + wide_boost * 3.2
                return {
                    'figsize': (13.4, figure_height),
                    'height_ratios': [top_ratio, image_ratio, image_ratio],
                    'hspace': 0.36,
                    'wspace': 0.22,
                    'adjust': {'left': 0.05, 'right': 0.98, 'top': 0.968, 'bottom': 0.04},
                }

            tall_boost = max(0.0, 1.05 - avg_aspect)
            image_ratio = min(1.55, 1.30 + tall_boost * 0.35)
            figure_height = 9.7 + tall_boost * 1.6
            return {
                'figsize': (13.2, figure_height),
                'height_ratios': [0.80, image_ratio],
                'hspace': 0.30,
                'wspace': 0.22,
                'adjust': {'left': 0.055, 'right': 0.98, 'top': 0.962, 'bottom': 0.05},
            }

        if stacked:
            wide_boost = max(0.0, max(max_aspect, avg_aspect) - 1.25)
            top_ratio = max(0.68, 0.84 - wide_boost * 0.16)
            detail_ratio = max(0.60, 0.72 - wide_boost * 0.10)
            image_ratio = min(1.85, 1.22 + wide_boost * 0.55)
            figure_height = 14.6 + wide_boost * 3.8
            return {
                'figsize': (15.0, figure_height),
                'height_ratios': [top_ratio, detail_ratio, image_ratio, image_ratio],
                'hspace': 0.38,
                'wspace': 0.30,
                'adjust': {'left': 0.05, 'right': 0.98, 'top': 0.972, 'bottom': 0.035},
            }

        tall_boost = max(0.0, 1.05 - avg_aspect)
        image_ratio = min(1.60, 1.45 + tall_boost * 0.30)
        figure_height = 12.0 + tall_boost * 1.8
        return {
            'figsize': (14.8, figure_height),
            'height_ratios': [0.92, 0.82, image_ratio],
            'hspace': 0.34,
            'wspace': 0.32,
            'adjust': {'left': 0.05, 'right': 0.98, 'top': 0.966, 'bottom': 0.04},
        }

    def _annotate_bar_values(self,
                             ax,
                             bars,
                             fmt: str = "{:.0f}",
                             offset_ratio: float = 0.03) -> None:
        """为柱状图补充数值标签"""
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

    def _configure_comparison_image_axis(self, ax, image: np.ndarray, title: str):
        """以正确比例显示对比中的代表图"""
        if image is None or getattr(image, 'size', 0) == 0:
            ax.text(
                0.5, 0.5, "暂无图像",
                ha='center', va='center',
                transform=ax.transAxes,
                color=self.MODERN_COLORS['text_secondary'],
            )
            ax.axis('off')
            return

        display_image = self._prepare_comparison_display_image(image)
        ax.imshow(display_image, interpolation='nearest', aspect='equal')
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.0)
        ax.set_anchor('C')
        ax.text(
            0.02,
            0.98,
            title,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            linespacing=1.15,
            color=self.MODERN_COLORS['text_primary'],
            bbox={
                'boxstyle': 'round,pad=0.28',
                'facecolor': self.MODERN_COLORS['bg_secondary'],
                'edgecolor': self.MODERN_COLORS['border'],
                'alpha': 0.84,
            },
        )
        ax.axis('off')

    def _estimate_comparison_summary_height(self, summary_text: str) -> int:
        """根据摘要文本估算更紧凑的摘要区高度。"""
        logical_lines = summary_text.splitlines() or [summary_text]
        wrapped_lines = sum(max(1, (len(line) + 71) // 72) for line in logical_lines)
        return min(220, max(128, 34 + wrapped_lines * 14))

    def _render_comparison_figure(self, summary_text: str, figure: Figure):
        """将对比摘要和图表渲染到对比分析面板"""
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
        """确保结果中带有检测可视化图像"""
        if result.get('visualization') is not None:
            return result
        return self._analyze_image_file(result['path'], include_visualization=True)

    def _show_comparison_window(self,
                                left_result: dict,
                                right_result: dict,
                                left_label: str,
                                right_label: str,
                                note: Optional[str] = None):
        """将双图对比结果显示到对比分析面板"""
        left_result = self._ensure_result_visualization(left_result)
        right_result = self._ensure_result_visualization(right_result)

        left_spatial = left_result['stats'].get('spatial_distribution') or {}
        right_spatial = right_result['stats'].get('spatial_distribution') or {}
        left_uniformity = left_spatial.get('uniformity_scores') or {}
        right_uniformity = right_spatial.get('uniformity_scores') or {}
        summary_text = self._format_comparison_summary(left_result, right_result, left_label, right_label, note)
        left_display_image = self._prepare_comparison_display_image(
            left_result.get('visualization'),
            max_width=1000,
            max_height=560,
        )
        right_display_image = self._prepare_comparison_display_image(
            right_result.get('visualization'),
            max_width=1000,
            max_height=560,
        )
        stack_images = self._should_stack_comparison_images(
            left_display_image,
            right_display_image,
            threshold=1.7,
        )

        layout = self._build_comparison_layout(
            left_display_image,
            right_display_image,
            stacked=stack_images,
            variant='pair',
        )
        pair_width, pair_height = layout['figsize']
        figure = Figure(figsize=(min(13.2, pair_width), min(10.8, pair_height)), dpi=90)
        figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])
        if stack_images:
            grid_spec = figure.add_gridspec(
                3,
                2,
                height_ratios=layout['height_ratios'],
                hspace=layout['hspace'],
                wspace=layout['wspace'],
            )
            ax_count = figure.add_subplot(grid_spec[0, 0])
            ax_metric = figure.add_subplot(grid_spec[0, 1])
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
            ax_count = figure.add_subplot(grid_spec[0, 0])
            ax_metric = figure.add_subplot(grid_spec[0, 1])
            ax_left = figure.add_subplot(grid_spec[1, 0])
            ax_right = figure.add_subplot(grid_spec[1, 1])
        figure.subplots_adjust(**layout['adjust'])

        labels = [left_label, right_label]
        counts = [left_result['stats']['count'], right_result['stats']['count']]
        count_colors = [self.MODERN_COLORS['accent_teal'], self.MODERN_COLORS['accent_rose']]
        bars = ax_count.bar(labels, counts, color=count_colors, alpha=0.9)
        ax_count.set_title('CNT数量对比', color=self.MODERN_COLORS['text_primary'])
        ax_count.set_ylabel('数量', color=self.MODERN_COLORS['text_secondary'])
        ax_count.grid(True, axis='y', alpha=0.25, linestyle='--')
        count_top = max(counts) if counts else 1.0
        ax_count.set_ylim(0, count_top * 1.28 if count_top > 0 else 1.0)
        for bar, value in zip(bars, counts):
            ax_count.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(count_top * 0.03, 1.0),
                f"{int(value)}",
                ha='center',
                va='bottom',
                fontsize=9,
                color=self.MODERN_COLORS['text_primary'],
            )

        metric_names = ["最近邻得分", "网格得分", "Moran得分", "综合得分"]
        left_metrics = [
            left_uniformity.get('nearest_neighbor', 0.0),
            left_uniformity.get('grid_density', 0.0),
            left_uniformity.get('moran', 0.0),
            left_uniformity.get('overall', 0.0),
        ]
        right_metrics = [
            right_uniformity.get('nearest_neighbor', 0.0),
            right_uniformity.get('grid_density', 0.0),
            right_uniformity.get('moran', 0.0),
            right_uniformity.get('overall', 0.0),
        ]
        x = np.arange(len(metric_names))
        width = 0.35
        left_metric_bars = ax_metric.bar(
            x - width / 2,
            left_metrics,
            width,
            label=left_label,
            color=self.MODERN_COLORS['accent_teal'],
        )
        right_metric_bars = ax_metric.bar(
            x + width / 2,
            right_metrics,
            width,
            label=right_label,
            color=self.MODERN_COLORS['accent_rose'],
        )
        ax_metric.set_xticks(x)
        ax_metric.set_xticklabels(metric_names)
        ax_metric.set_ylabel('得分 (0-100，越大越均匀)', color=self.MODERN_COLORS['text_secondary'])
        ax_metric.set_title('均匀性得分对比', color=self.MODERN_COLORS['text_primary'])
        ax_metric.legend(frameon=False)
        ax_metric.grid(True, axis='y', alpha=0.25, linestyle='--')
        self._annotate_bar_values(ax_metric, left_metric_bars, fmt="{:.1f}", offset_ratio=0.012)
        self._annotate_bar_values(ax_metric, right_metric_bars, fmt="{:.1f}", offset_ratio=0.012)
        ax_metric.set_ylim(0, 105)

        self._configure_comparison_image_axis(
            ax_left,
            left_result['visualization'],
            f"{left_label}典型CNT分布\n{left_result['name']}\nCNT={int(left_result['stats']['count'])}",
        )

        self._configure_comparison_image_axis(
            ax_right,
            right_result['visualization'],
            f"{right_label}典型CNT分布\n{right_result['name']}\nCNT={int(right_result['stats']['count'])}",
        )

        self._apply_standard_axes_style([ax_count, ax_metric])
        for ax in (ax_left, ax_right):
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            for spine in ax.spines.values():
                spine.set_color(self.MODERN_COLORS['border'])

        self._render_comparison_figure(summary_text, figure)
        return

    def _prepare_comparison_display_image(self,
                                          image: Optional[np.ndarray],
                                          max_width: int = 1200,
                                          max_height: int = 700) -> Optional[np.ndarray]:
        """为对比页显示预先缩小大图，降低 Tk + Matplotlib 渲染压力。"""
        if image is None or getattr(image, 'size', 0) == 0:
            return image

        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return image

        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 0.999:
            return image

        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    def _fit_comparison_figure_to_frame(self,
                                        figure: Figure,
                                        chart_frame: Optional[tk.Widget],
                                        min_width_px: int = 620,
                                        padding_px: int = 24,
                                        max_width_px: int = 1260,
                                        allow_expand: bool = False) -> None:
        """按当前中间栏可用宽度等比缩放对比图，避免切页后显示不全。"""
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
        if current_width_px <= target_width_px + 8 and (not allow_expand or current_width_px >= target_width_px - 24):
            return

        scale = target_width_px / current_width_px
        resized_width_px = max(min_width_px, int(round(current_width_px * scale)))
        resized_height_px = max(420, int(round(current_height_px * scale)))
        figure.set_size_inches(
            resized_width_px / figure.dpi,
            resized_height_px / figure.dpi,
            forward=False,
        )

    def _schedule_comparison_layout_refresh(self, delay_ms: int = 80) -> None:
        """防抖刷新对比分析页的图表尺寸与滚动区域。"""
        if self._comparison_layout_job is not None:
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
        self._fit_comparison_figure_to_frame(figure, chart_frame, allow_expand=True)
        new_height = min(1600, max(720, int(round(figure.get_figheight() * figure.dpi)) + 40))
        self.comparison_panel.set_section_height('comparison', new_height)

        new_size = (
            int(round(figure.get_figwidth() * figure.dpi)),
            int(round(figure.get_figheight() * figure.dpi)),
        )
        if new_size != old_size:
            canvas.draw()

        self.comparison_panel.refresh_layout()

    def _legacy_show_group_comparison_window(self,
                                      base_group: dict,
                                      exp_group: dict,
                                      note: Optional[str] = None,
                                      failures: Optional[List[str]] = None):
        """将base组与实验组的多图对比结果显示到对比分析面板"""
        base_counts = [detail['count'] for detail in base_group['file_details']]
        exp_counts = [detail['count'] for detail in exp_group['file_details']]
        count_tests = self._compute_two_group_tests(base_counts, exp_counts)
        nn_tests = self._compute_two_group_tests(
            [detail['nearest_neighbor_cv'] for detail in base_group['file_details']],
            [detail['nearest_neighbor_cv'] for detail in exp_group['file_details']],
        )
        grid_tests = self._compute_two_group_tests(
            [detail['grid_density_cv'] for detail in base_group['file_details']],
            [detail['grid_density_cv'] for detail in exp_group['file_details']],
        )
        uniformity_nn_tests = self._compute_two_group_tests(
            [detail.get('uniformity_nn_score', 0.0) for detail in base_group['file_details']],
            [detail.get('uniformity_nn_score', 0.0) for detail in exp_group['file_details']],
        )
        uniformity_grid_tests = self._compute_two_group_tests(
            [detail.get('uniformity_grid_score', 0.0) for detail in base_group['file_details']],
            [detail.get('uniformity_grid_score', 0.0) for detail in exp_group['file_details']],
        )
        uniformity_moran_tests = self._compute_two_group_tests(
            [detail.get('uniformity_moran_score', 0.0) for detail in base_group['file_details']],
            [detail.get('uniformity_moran_score', 0.0) for detail in exp_group['file_details']],
        )
        uniformity_tests = self._compute_two_group_tests(
            [detail.get('uniformity_score', 0.0) for detail in base_group['file_details']],
            [detail.get('uniformity_score', 0.0) for detail in exp_group['file_details']],
        )
        base_typical = self._select_representative_result(base_group)
        exp_typical = self._select_representative_result(exp_group)
        summary_text = self._format_group_comparison_summary(base_group, exp_group, note, failures)
        stack_typical_images = self._should_stack_comparison_images(
            base_typical.get('visualization'),
            exp_typical.get('visualization'),
            threshold=1.25,
        )

        layout = self._build_comparison_layout(
            base_typical.get('visualization'),
            exp_typical.get('visualization'),
            stacked=stack_typical_images,
            variant='group',
        )
        figure = Figure(figsize=layout['figsize'], dpi=100)
        figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])
        if stack_typical_images:
            grid_spec = figure.add_gridspec(
                4,
                6,
                height_ratios=layout['height_ratios'],
                hspace=layout['hspace'],
                wspace=layout['wspace'],
            )
            ax_mean = figure.add_subplot(grid_spec[0, 0:2])
            ax_box = figure.add_subplot(grid_spec[0, 2:4])
            ax_metric = figure.add_subplot(grid_spec[0, 4:6])
            ax_detail = figure.add_subplot(grid_spec[1, 0:6])
            ax_base_typical = figure.add_subplot(grid_spec[2, 0:6])
            ax_exp_typical = figure.add_subplot(grid_spec[3, 0:6])
        else:
            grid_spec = figure.add_gridspec(
                3,
                6,
                height_ratios=layout['height_ratios'],
                hspace=layout['hspace'],
                wspace=layout['wspace'],
            )
            ax_mean = figure.add_subplot(grid_spec[0, 0:2])
            ax_box = figure.add_subplot(grid_spec[0, 2:4])
            ax_metric = figure.add_subplot(grid_spec[0, 4:6])
            ax_detail = figure.add_subplot(grid_spec[1, 0:6])
            ax_base_typical = figure.add_subplot(grid_spec[2, 0:3])
            ax_exp_typical = figure.add_subplot(grid_spec[2, 3:6])
        figure.subplots_adjust(**layout['adjust'])

        base_count = base_group['count_stats']
        exp_count = exp_group['count_stats']
        labels = ['base组', '实验组']
        means = [base_count['mean'], exp_count['mean']]
        stds = [base_count['std'], exp_count['std']]

        mean_bars = ax_mean.bar(
            labels,
            means,
            yerr=stds,
            capsize=6,
            color=[self.MODERN_COLORS['accent_rose'], self.MODERN_COLORS['accent_teal']],
            alpha=0.9,
        )
        ax_mean.set_title('每图CNT数量均值 ± 标准差', color=self.MODERN_COLORS['text_primary'])
        ax_mean.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_mean.grid(True, axis='y', alpha=0.25, linestyle='--')

        diff_ratio = ((exp_count['mean'] - base_count['mean']) / base_count['mean'] * 100.0) if base_count['mean'] > 0 else 0.0
        y_max = max(means[i] + stds[i] for i in range(len(means))) if means else 1.0
        for bar, mean, std in zip(mean_bars, means, stds):
            ax_mean.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + max(y_max * 0.03, 1.0),
                f"{mean:.1f}",
                ha='center',
                va='bottom',
                fontsize=8.5,
                color=self.MODERN_COLORS['text_primary'],
            )
        ax_mean.text(
            0.5,
            y_max * 1.08 if y_max > 0 else 0.5,
            f"实验组相对base组 {diff_ratio:+.1f}%\n"
            f"t检验 p={self._format_pvalue(count_tests['t_pvalue'])} ({self._get_significance_marker(count_tests['t_pvalue'])})",
            ha='center',
            va='bottom',
            fontsize=9,
            color=self.MODERN_COLORS['text_primary'],
        )
        ax_mean.set_ylim(0, y_max * 1.34 if y_max > 0 else 1.0)

        box = ax_box.boxplot([base_counts, exp_counts], tick_labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], [self.MODERN_COLORS['accent_rose'], self.MODERN_COLORS['accent_teal']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax_box.set_title('组内CNT数量分布', color=self.MODERN_COLORS['text_primary'])
        ax_box.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_box.grid(True, axis='y', alpha=0.25, linestyle='--')

        rng = np.random.default_rng(42)
        ax_box.scatter(
            1 + rng.normal(0, 0.04, len(base_counts)),
            base_counts,
            s=24,
            alpha=0.75,
            color=self.MODERN_COLORS['accent_rose'],
            edgecolors='white',
            linewidths=0.4,
            zorder=3,
        )
        ax_box.scatter(
            2 + rng.normal(0, 0.04, len(exp_counts)),
            exp_counts,
            s=24,
            alpha=0.75,
            color=self.MODERN_COLORS['accent_teal'],
            edgecolors='white',
            linewidths=0.4,
            zorder=3,
        )
        box_top = max(max(base_counts), max(exp_counts)) if base_counts and exp_counts else 1.0
        ax_box.text(
            1.5,
            box_top * 1.05 if box_top > 0 else 0.5,
            f"Mann-Whitney p={self._format_pvalue(count_tests['mw_pvalue'])}",
            ha='center',
            va='bottom',
            fontsize=8.5,
            color=self.MODERN_COLORS['text_primary'],
        )
        ax_box.set_ylim(0, box_top * 1.18 if box_top > 0 else 1.0)

        metric_names = ["最近邻得分", "网格得分", "Moran得分", "综合得分"]
        base_metric_means = [
            base_group['spatial_stats']['uniformity_nn_score']['mean'],
            base_group['spatial_stats']['uniformity_grid_score']['mean'],
            base_group['spatial_stats']['uniformity_moran_score']['mean'],
            base_group['spatial_stats']['uniformity_score']['mean'],
        ]
        exp_metric_means = [
            exp_group['spatial_stats']['uniformity_nn_score']['mean'],
            exp_group['spatial_stats']['uniformity_grid_score']['mean'],
            exp_group['spatial_stats']['uniformity_moran_score']['mean'],
            exp_group['spatial_stats']['uniformity_score']['mean'],
        ]
        x = np.arange(len(metric_names))
        width = 0.35
        base_metric_bars = ax_metric.bar(
            x - width / 2,
            base_metric_means,
            width,
            label='base组',
            color=self.MODERN_COLORS['accent_rose'],
        )
        exp_metric_bars = ax_metric.bar(
            x + width / 2,
            exp_metric_means,
            width,
            label='实验组',
            color=self.MODERN_COLORS['accent_teal'],
        )
        ax_metric.set_xticks(x)
        ax_metric.set_xticklabels(metric_names)
        ax_metric.set_ylabel('得分 (0-100，越大越均匀)', color=self.MODERN_COLORS['text_secondary'])
        ax_metric.set_title('组别均匀性得分均值', color=self.MODERN_COLORS['text_primary'])
        ax_metric.legend(frameon=False)
        ax_metric.grid(True, axis='y', alpha=0.25, linestyle='--')
        self._annotate_bar_values(ax_metric, base_metric_bars, fmt="{:.1f}", offset_ratio=0.012)
        self._annotate_bar_values(ax_metric, exp_metric_bars, fmt="{:.1f}", offset_ratio=0.012)

        metric_tests = [uniformity_nn_tests, uniformity_grid_tests, uniformity_moran_tests, uniformity_tests]
        metric_y = np.maximum(base_metric_means, exp_metric_means)
        metric_pad = max(float(np.max(metric_y)) * 0.05, 1.5) if len(metric_y) else 2.0
        for idx, test in enumerate(metric_tests):
            ax_metric.text(
                x[idx],
                metric_y[idx] + metric_pad,
                f"p={self._format_pvalue(test['t_pvalue'])}\n{self._get_significance_marker(test['t_pvalue'])}",
                ha='center',
                va='bottom',
                fontsize=8,
                color=self.MODERN_COLORS['text_primary'],
            )
        ax_metric.set_ylim(0, 108)

        base_x = np.arange(1, len(base_counts) + 1)
        exp_x = np.arange(1, len(exp_counts) + 1)
        ax_detail.plot(
            base_x,
            base_counts,
            marker='o',
            linewidth=1.8,
            color=self.MODERN_COLORS['accent_rose'],
            label='base组',
        )
        ax_detail.plot(
            exp_x,
            exp_counts,
            marker='o',
            linewidth=1.8,
            color=self.MODERN_COLORS['accent_teal'],
            label='实验组',
        )
        ax_detail.set_title('组内逐图CNT数量', color=self.MODERN_COLORS['text_primary'])
        ax_detail.set_xlabel('组内图像序号', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.legend(frameon=False)
        ax_detail.grid(True, alpha=0.25, linestyle='--')

        self._configure_comparison_image_axis(
            ax_base_typical,
            base_typical['visualization'],
            f"base组典型CNT分布\n{base_typical['name']}\nCNT={int(base_typical['stats']['count'])}",
        )

        self._configure_comparison_image_axis(
            ax_exp_typical,
            exp_typical['visualization'],
            f"实验组典型CNT分布\n{exp_typical['name']}\nCNT={int(exp_typical['stats']['count'])}",
        )

        self._apply_standard_axes_style([ax_mean, ax_box, ax_metric, ax_detail])
        for ax in (ax_base_typical, ax_exp_typical):
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            for spine in ax.spines.values():
                spine.set_color(self.MODERN_COLORS['border'])

        self._render_comparison_figure(summary_text, figure)
        return

    def _format_group_detail_lines(self, group_summary: dict) -> List[str]:
        """生成组内逐图统计明细，优先展示团聚风险与均匀性。"""
        lines = [f"{group_summary['label']}逐图结果:"]
        for detail in group_summary.get('file_details', []):
            lines.append(
                f"  - {detail['name']}: CNT={detail['count']:.0f}，团聚风险总分={detail.get('aggregation_score', 0.0):.1f}，"
                f"综合均匀性得分={detail.get('uniformity_score', 0.0):.1f}，Moran's I={detail.get('morans_i', 0.0):.3f}，"
                f"最近邻CV={detail.get('nearest_neighbor_cv', 0.0):.3f}，网格CNT数CV={detail.get('grid_density_cv', 0.0):.3f}"
            )
        return lines

    def _format_group_comparison_summary(self,
                                         base_group: dict,
                                         exp_group: dict,
                                         note: Optional[str] = None,
                                         failures: Optional[List[str]] = None) -> str:
        """生成组别对比摘要，优先强调团聚风险与均匀性差异。"""
        base_count = base_group['count_stats']
        exp_count = exp_group['count_stats']
        base_spatial = base_group['spatial_stats']
        exp_spatial = exp_group['spatial_stats']
        tests = self._compute_group_comparison_tests(base_group, exp_group)
        count_tests = tests['count']
        nn_tests = tests['nearest_neighbor_cv']
        grid_tests = tests['grid_density_cv']
        uniformity_tests = tests['uniformity_score']
        aggregation_tests = tests['aggregation_score']

        split_mode_label = self.split_mode_var.get()
        profile_label = self.detect_profile_var.get()
        count_diff = exp_count['mean'] - base_count['mean']
        count_ratio = (count_diff / base_count['mean'] * 100.0) if base_count['mean'] > 0 else 0.0

        base_aggregation_mean = base_spatial['aggregation_score']['mean']
        exp_aggregation_mean = exp_spatial['aggregation_score']['mean']
        aggregation_diff = base_aggregation_mean - exp_aggregation_mean
        uniformity_diff = exp_spatial['uniformity_score']['mean'] - base_spatial['uniformity_score']['mean']
        evidence_significant = (
            self._is_test_significant(aggregation_tests) or
            self._is_test_significant(uniformity_tests)
        )
        strong_conclusion = (
            aggregation_diff >= 5.0 and
            uniformity_diff >= 5.0 and
            evidence_significant
        )
        directional_advantage = aggregation_diff > 0 and uniformity_diff > 0

        if strong_conclusion:
            headline = (
                "主结论: [团聚更明显] [分布更均匀] [统计显著] "
                "base组团聚影响更明显，实验组CNT分布更均匀。"
            )
        elif directional_advantage:
            headline = (
                "主结论: [团聚更明显] [分布更均匀] [趋势存在但证据有限] "
                "实验组在均匀性方向上更优，base组存在更高团聚倾向，但统计证据有限/差异未完全拉开。"
            )
        else:
            headline = (
                "主结论: [方向未完全一致] "
                "当前两组在团聚风险和均匀性上的差异未形成一致优势，仍需结合更多图像或继续优化参数。"
            )

        aggregation_marker = self._get_significance_marker(self._get_preferred_pvalue(aggregation_tests))
        uniformity_marker = self._get_significance_marker(self._get_preferred_pvalue(uniformity_tests))
        count_marker = self._get_significance_marker(self._get_preferred_pvalue(count_tests))

        lines: List[str] = []
        if note:
            lines.append(note)
            lines.append("")

        lines.extend([
            headline,
            (
                f"团聚风险总分: base组 {base_aggregation_mean:.1f}±{base_spatial['aggregation_score']['std']:.1f}，"
                f"实验组 {exp_aggregation_mean:.1f}±{exp_spatial['aggregation_score']['std']:.1f}。"
                f"该得分范围 0-100，越大越团聚；当前 base 组相对实验组高 {aggregation_diff:.1f} 分。"
            ),
            (
                f"综合均匀性得分: base组 {base_spatial['uniformity_score']['mean']:.1f}±{base_spatial['uniformity_score']['std']:.1f}，"
                f"实验组 {exp_spatial['uniformity_score']['mean']:.1f}±{exp_spatial['uniformity_score']['std']:.1f}。"
                f"该得分范围 0-100，越大越均匀；当前实验组相对 base 组高 {uniformity_diff:.1f} 分。"
            ),
            (
                f"统计支持: 团聚风险总分 t检验 p={self._format_pvalue(aggregation_tests['t_pvalue'])}，"
                f"Mann-Whitney p={self._format_pvalue(aggregation_tests['mw_pvalue'])}，显著性 {aggregation_marker}；"
                f"综合均匀性得分 t检验 p={self._format_pvalue(uniformity_tests['t_pvalue'])}，"
                f"Mann-Whitney p={self._format_pvalue(uniformity_tests['mw_pvalue'])}，显著性 {uniformity_marker}。"
            ),
            "",
            f"相同识别条件: 模糊={self.blur_kernel_var.get()} / 自适应块={self.adaptive_block_var.get()} / C={self.adaptive_c_var.get()} / 桥接={self.bridge_strength_var.get()} / 最小长度={self.min_length_um_var.get():.1f}μm / 最小长宽比={self.min_slenderness_var.get():.1f} / 检测风格={profile_label} / 拆分模式={split_mode_label} / 合并距离={self.merge_distance_px_var.get()}px",
            f"base组: 共 {base_group['image_count']} 张图，总CNT {int(round(base_count['total']))}，平均 {base_count['mean']:.2f}，标准差 {base_count['std']:.2f}，方差 {base_count['var']:.2f}，范围 {base_count['min']:.0f}~{base_count['max']:.0f}",
            f"实验组: 共 {exp_group['image_count']} 张图，总CNT {int(round(exp_count['total']))}，平均 {exp_count['mean']:.2f}，标准差 {exp_count['std']:.2f}，方差 {exp_count['var']:.2f}，范围 {exp_count['min']:.0f}~{exp_count['max']:.0f}",
        ])

        if count_diff >= 0:
            lines.append(f"CNT数量均值差异: 实验组比base组高 {count_diff:.2f}（{count_ratio:.1f}%）")
        else:
            lines.append(f"CNT数量均值差异: base组比实验组高 {abs(count_diff):.2f}（{abs(count_ratio):.1f}%）")

        lines.extend([
            "",
            "组别空间分布统计:",
            f"方法一 中心点最近邻CV: base组 {base_spatial['nearest_neighbor_cv']['mean']:.3f}±{base_spatial['nearest_neighbor_cv']['std']:.3f}，实验组 {exp_spatial['nearest_neighbor_cv']['mean']:.3f}±{exp_spatial['nearest_neighbor_cv']['std']:.3f}。该值越小越均匀。",
            f"补充指标 最近邻指数NNI: base组 {base_spatial['nearest_neighbor_index']['mean']:.3f}±{base_spatial['nearest_neighbor_index']['std']:.3f}，实验组 {exp_spatial['nearest_neighbor_index']['mean']:.3f}±{exp_spatial['nearest_neighbor_index']['std']:.3f}。该值大于 1 表示比随机分布更均匀。",
            f"方法二 网格CNT数CV: base组 {base_spatial['grid_density_cv']['mean']:.3f}±{base_spatial['grid_density_cv']['std']:.3f}，实验组 {exp_spatial['grid_density_cv']['mean']:.3f}±{exp_spatial['grid_density_cv']['std']:.3f}。该值越小越均匀。",
            f"方法三 Moran's I: base组 {base_spatial['morans_i']['mean']:.3f}±{base_spatial['morans_i']['std']:.3f}，实验组 {exp_spatial['morans_i']['mean']:.3f}±{exp_spatial['morans_i']['std']:.3f}。该值越大越聚集。",
            "",
            "统计检验:",
            f"CNT数量 t检验 p={self._format_pvalue(count_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(count_tests['mw_pvalue'])}，显著性 {count_marker}",
            f"综合团聚风险 t检验 p={self._format_pvalue(aggregation_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(aggregation_tests['mw_pvalue'])}，显著性 {aggregation_marker}",
            f"综合均匀性得分 t检验 p={self._format_pvalue(uniformity_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(uniformity_tests['mw_pvalue'])}，显著性 {uniformity_marker}",
            f"最近邻CV t检验 p={self._format_pvalue(nn_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(nn_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(self._get_preferred_pvalue(nn_tests))}",
            f"网格CNT数CV t检验 p={self._format_pvalue(grid_tests['t_pvalue'])}，Mann-Whitney p={self._format_pvalue(grid_tests['mw_pvalue'])}，显著性 {self._get_significance_marker(self._get_preferred_pvalue(grid_tests))}",
            "",
        ])

        lines.extend(self._format_group_detail_lines(base_group))
        lines.append("")
        lines.extend(self._format_group_detail_lines(exp_group))

        if failures:
            lines.append("")
            lines.append("未成功分析的文件:")
            lines.extend([f"  - {item}" for item in failures])

        return "\n".join(lines)

    def _plot_group_aggregation_risk_chart(self, ax, base_group: dict, exp_group: dict, tests: dict) -> None:
        """绘制组别团聚风险主图卡。"""
        metric_names = ["团聚风险总分", "Moran 团聚风险", "最近邻离散风险", "网格密度离散风险"]
        base_metric_means = [
            base_group['spatial_stats']['aggregation_score']['mean'],
            base_group['spatial_stats']['aggregation_moran_score']['mean'],
            base_group['spatial_stats']['aggregation_nn_score']['mean'],
            base_group['spatial_stats']['aggregation_grid_score']['mean'],
        ]
        exp_metric_means = [
            exp_group['spatial_stats']['aggregation_score']['mean'],
            exp_group['spatial_stats']['aggregation_moran_score']['mean'],
            exp_group['spatial_stats']['aggregation_nn_score']['mean'],
            exp_group['spatial_stats']['aggregation_grid_score']['mean'],
        ]
        metric_tests = [
            tests['aggregation_score'],
            tests['aggregation_moran_score'],
            tests['aggregation_nn_score'],
            tests['aggregation_grid_score'],
        ]

        x = np.arange(len(metric_names))
        width = 0.34
        base_color = self.MODERN_COLORS['accent_rose']
        exp_color = self.MODERN_COLORS['accent_teal']

        base_metric_bars = ax.bar(
            x - width / 2,
            base_metric_means,
            width,
            label='base组',
            color=base_color,
            alpha=0.95,
        )
        exp_metric_bars = ax.bar(
            x + width / 2,
            exp_metric_means,
            width,
            label='实验组',
            color=exp_color,
            alpha=0.82,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel('团聚风险得分 (0-100)', color=self.MODERN_COLORS['text_secondary'])
        ax.set_title('团聚风险 vs 均匀性优势', color=self.MODERN_COLORS['text_primary'])
        legend = ax.legend(title='数值越大 = 越团聚', frameon=False, loc='upper right')
        if legend is not None:
            legend.get_title().set_color(self.MODERN_COLORS['text_secondary'])
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        self._annotate_bar_values(ax, base_metric_bars, fmt="{:.1f}", offset_ratio=0.01)
        self._annotate_bar_values(ax, exp_metric_bars, fmt="{:.1f}", offset_ratio=0.01)

        ax.text(
            0.01,
            0.97,
            "base高 / 实验低 = 实验组更均匀",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=8.5,
            color=self.MODERN_COLORS['text_secondary'],
            bbox={
                'boxstyle': 'round,pad=0.25',
                'facecolor': self.MODERN_COLORS['bg_tertiary'],
                'edgecolor': self.MODERN_COLORS['border'],
                'alpha': 0.95,
            },
        )

        for idx, test in enumerate(metric_tests):
            pvalue = self._get_preferred_pvalue(test)
            marker = self._get_significance_marker(pvalue)
            annotation_y = min(97.5, max(base_metric_means[idx], exp_metric_means[idx]) + 7.0)
            ax.text(
                x[idx],
                annotation_y,
                f"p={self._format_pvalue(pvalue)}\n{marker}",
                ha='center',
                va='bottom',
                fontsize=8,
                color=self.MODERN_COLORS['text_primary'],
            )

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
        """根据样本量自适应显示组内数量分布，避免单点时出现失真的箱线图/折线图。"""
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
                ax_box.scatter(
                    [x_pos] * len(values),
                    values,
                    s=48,
                    alpha=0.82,
                    color=color,
                    edgecolors='white',
                    linewidths=0.6,
                    zorder=3,
                )
                mean_value = float(np.mean(values))
                ax_box.hlines(mean_value, x_pos - 0.18, x_pos + 0.18, colors=color, linewidth=2.0, zorder=2)
                ax_box.text(
                    x_pos,
                    mean_value + max(mean_value * 0.03, 1.0),
                    f"{mean_value:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=8.5,
                    color=self.MODERN_COLORS['text_primary'],
                )
            ax_box.set_xticks([1, 2])
            ax_box.set_xticklabels(labels)
            ax_box.set_title('组内CNT样本点（样本量较少）', color=self.MODERN_COLORS['text_primary'])
            ax_box.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
            ax_box.grid(True, axis='y', alpha=0.25, linestyle='--')
            max_count = max(base_counts + exp_counts) if (base_counts or exp_counts) else 1.0
            ax_box.text(
                1.5,
                max_count * 1.08 if max_count > 0 else 0.5,
                f"每组仅 {min_samples} 张图，已切换为单点概览 | {self._format_compact_significance(count_tests)}",
                ha='center',
                va='bottom',
                fontsize=8.5,
                color=self.MODERN_COLORS['text_primary'],
            )
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
                ax_detail.text(
                    value + max(max(detail_counts, default=1.0) * 0.015, 0.6),
                    y_pos,
                    f"{value:.0f}",
                    va='center',
                    fontsize=8.5,
                    color=self.MODERN_COLORS['text_primary'],
                )
            return

        box = ax_box.boxplot([base_counts, exp_counts], tick_labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax_box.set_title('组内CNT数量分布', color=self.MODERN_COLORS['text_primary'])
        ax_box.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_box.grid(True, axis='y', alpha=0.25, linestyle='--')

        rng = np.random.default_rng(42)
        ax_box.scatter(
            1 + rng.normal(0, 0.04, len(base_counts)),
            base_counts,
            s=24,
            alpha=0.75,
            color=colors[0],
            edgecolors='white',
            linewidths=0.4,
            zorder=3,
        )
        ax_box.scatter(
            2 + rng.normal(0, 0.04, len(exp_counts)),
            exp_counts,
            s=24,
            alpha=0.75,
            color=colors[1],
            edgecolors='white',
            linewidths=0.4,
            zorder=3,
        )
        box_top = max(max(base_counts), max(exp_counts)) if base_counts and exp_counts else 1.0
        ax_box.text(
            1.5,
            box_top * 1.05 if box_top > 0 else 0.5,
            f"分布检验 {self._format_compact_significance(count_tests)}",
            ha='center',
            va='bottom',
            fontsize=8.5,
            color=self.MODERN_COLORS['text_primary'],
        )
        ax_box.set_ylim(0, box_top * 1.18 if box_top > 0 else 1.0)

        base_x = np.arange(1, len(base_counts) + 1)
        exp_x = np.arange(1, len(exp_counts) + 1)
        if max(len(base_counts), len(exp_counts)) <= 3:
            ax_detail.scatter(base_x, base_counts, s=42, color=colors[0], label='base组', zorder=3)
            ax_detail.scatter(exp_x, exp_counts, s=42, color=colors[1], label='实验组', zorder=3)
            ax_detail.set_title('组内逐图CNT样本点', color=self.MODERN_COLORS['text_primary'])
        else:
            ax_detail.plot(
                base_x,
                base_counts,
                marker='o',
                linewidth=1.8,
                color=colors[0],
                label='base组',
            )
            ax_detail.plot(
                exp_x,
                exp_counts,
                marker='o',
                linewidth=1.8,
                color=colors[1],
                label='实验组',
            )
            ax_detail.set_title('组内逐图CNT数量', color=self.MODERN_COLORS['text_primary'])
        ax_detail.set_xlabel('组内图像序号', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_detail.legend(frameon=False)
        ax_detail.grid(True, alpha=0.25, linestyle='--')

    def _show_group_comparison_window(self,
                                      base_group: dict,
                                      exp_group: dict,
                                      note: Optional[str] = None,
                                      failures: Optional[List[str]] = None):
        """将base组与实验组的多图对比结果显示到对比分析面板。"""
        tests = self._compute_group_comparison_tests(base_group, exp_group)
        count_tests = tests['count']
        base_typical = self._select_representative_result(base_group)
        exp_typical = self._select_representative_result(exp_group)
        summary_text = self._format_group_comparison_summary(base_group, exp_group, note, failures)
        base_display_image = self._prepare_comparison_display_image(
            base_typical.get('visualization'),
            max_width=1000,
            max_height=560,
        )
        exp_display_image = self._prepare_comparison_display_image(
            exp_typical.get('visualization'),
            max_width=1000,
            max_height=560,
        )
        stack_typical_images = self._should_stack_comparison_images(
            base_display_image,
            exp_display_image,
            threshold=1.8,
        )

        layout = self._build_comparison_layout(
            base_display_image,
            exp_display_image,
            stacked=stack_typical_images,
            variant='group',
        )
        figure_width, figure_height = layout['figsize']
        compact_width = min(13.8, figure_width)
        compact_height = min(12.6, figure_height + (1.1 if stack_typical_images else 0.7))
        figure = Figure(figsize=(compact_width, compact_height), dpi=88)
        figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])

        if stack_typical_images:
            base_ratios = layout['height_ratios']
            grid_spec = figure.add_gridspec(
                5,
                6,
                height_ratios=[
                    max(1.0, base_ratios[0] + 0.20),
                    max(0.72, base_ratios[1] * 0.95),
                    max(0.72, base_ratios[1]),
                    base_ratios[2],
                    base_ratios[3],
                ],
                hspace=max(layout['hspace'], 0.34),
                wspace=layout['wspace'],
            )
            ax_metric = figure.add_subplot(grid_spec[0, 0:6])
            ax_mean = figure.add_subplot(grid_spec[1, 0:3])
            ax_box = figure.add_subplot(grid_spec[1, 3:6])
            ax_detail = figure.add_subplot(grid_spec[2, 0:6])
            ax_base_typical = figure.add_subplot(grid_spec[3, 0:6])
            ax_exp_typical = figure.add_subplot(grid_spec[4, 0:6])
        else:
            base_ratios = layout['height_ratios']
            grid_spec = figure.add_gridspec(
                4,
                6,
                height_ratios=[
                    max(1.02, base_ratios[0] + 0.18),
                    max(0.74, base_ratios[1] * 0.95),
                    max(0.74, base_ratios[1]),
                    base_ratios[2],
                ],
                hspace=max(layout['hspace'], 0.32),
                wspace=layout['wspace'],
            )
            ax_metric = figure.add_subplot(grid_spec[0, 0:6])
            ax_mean = figure.add_subplot(grid_spec[1, 0:3])
            ax_box = figure.add_subplot(grid_spec[1, 3:6])
            ax_detail = figure.add_subplot(grid_spec[2, 0:6])
            ax_base_typical = figure.add_subplot(grid_spec[3, 0:3])
            ax_exp_typical = figure.add_subplot(grid_spec[3, 3:6])
        figure.subplots_adjust(**layout['adjust'])

        base_count = base_group['count_stats']
        exp_count = exp_group['count_stats']
        labels = ['base组', '实验组']
        means = [base_count['mean'], exp_count['mean']]
        stds = [base_count['std'], exp_count['std']]

        self._plot_group_aggregation_risk_chart(ax_metric, base_group, exp_group, tests)

        mean_bars = ax_mean.bar(
            labels,
            means,
            yerr=stds,
            capsize=6,
            color=[self.MODERN_COLORS['accent_rose'], self.MODERN_COLORS['accent_teal']],
            alpha=0.9,
        )
        ax_mean.set_title('每图CNT数量均值 ± 标准差', color=self.MODERN_COLORS['text_primary'])
        ax_mean.set_ylabel('CNT数量', color=self.MODERN_COLORS['text_secondary'])
        ax_mean.grid(True, axis='y', alpha=0.25, linestyle='--')

        diff_ratio = ((exp_count['mean'] - base_count['mean']) / base_count['mean'] * 100.0) if base_count['mean'] > 0 else 0.0
        y_max = max(means[i] + stds[i] for i in range(len(means))) if means else 1.0
        for bar, mean, std in zip(mean_bars, means, stds):
            ax_mean.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + max(y_max * 0.03, 1.0),
                f"{mean:.1f}",
                ha='center',
                va='bottom',
                fontsize=8.5,
                color=self.MODERN_COLORS['text_primary'],
            )
        ax_mean.text(
            0.5,
            y_max * 1.09 if y_max > 0 else 0.5,
            f"实验组相对base组 {diff_ratio:+.1f}%\n"
            f"p={self._format_pvalue(self._get_preferred_pvalue(count_tests))} ({self._get_significance_marker(self._get_preferred_pvalue(count_tests))})",
            ha='center',
            va='bottom',
            fontsize=9,
            color=self.MODERN_COLORS['text_primary'],
        )
        ax_mean.set_ylim(0, y_max * 1.36 if y_max > 0 else 1.0)

        self._render_group_count_distribution_views(
            ax_box,
            ax_detail,
            base_group,
            exp_group,
            count_tests,
        )

        self._configure_comparison_image_axis(
            ax_base_typical,
            base_typical['visualization'],
            f"base组典型CNT分布\n{base_typical['name']}\nCNT={int(base_typical['stats']['count'])}",
        )

        self._configure_comparison_image_axis(
            ax_exp_typical,
            exp_typical['visualization'],
            f"实验组典型CNT分布\n{exp_typical['name']}\nCNT={int(exp_typical['stats']['count'])}",
        )

        self._apply_standard_axes_style([ax_metric, ax_mean, ax_box, ax_detail])
        for ax in (ax_base_typical, ax_exp_typical):
            ax.set_facecolor(self.MODERN_COLORS['bg_secondary'])
            for spine in ax.spines.values():
                spine.set_color(self.MODERN_COLORS['border'])

        self._render_comparison_figure(summary_text, figure)
        return

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

    def _format_compact_significance(self, test_result: Optional[dict]) -> str:
        """将统计检验压缩成单一 p 值展示。"""
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
        direction_suffix = {
            'up': '↑',
            'down': '↓',
            'cluster': '↑聚集',
        }.get(direction, '')
        if qualifier:
            title = f"{label}({qualifier})"
        else:
            title = f"{label}({direction_suffix})" if direction_suffix else label
        value_fmt = f"{{:.{precision}f}}"

        if left_std is None or right_std is None:
            line = (
                f"{title}: {left_label} {value_fmt.format(left_value)} vs "
                f"{right_label} {value_fmt.format(right_value)}"
            )
        else:
            line = (
                f"{title}: {left_label} {value_fmt.format(left_value)}±{value_fmt.format(left_std)} vs "
                f"{right_label} {value_fmt.format(right_value)}±{value_fmt.format(right_std)}"
            )

        if test_result is not None:
            line += f" | {self._format_compact_significance(test_result)}"
        return line

    def _format_group_detail_lines(self, group_summary: dict) -> List[str]:
        """生成精简版逐图明细。"""
        lines = [f"{group_summary['label']}逐图明细:"]
        for detail in group_summary.get('file_details', []):
            lines.append(
                f"  - {detail['name']}: CNT={detail['count']:.0f} | 风险={detail.get('aggregation_score', 0.0):.1f} | "
                f"均匀={detail.get('uniformity_score', 0.0):.1f} | Moran={detail.get('morans_i', 0.0):.3f} | "
                f"NNCV={detail.get('nearest_neighbor_cv', 0.0):.3f} | GridCV={detail.get('grid_density_cv', 0.0):.3f}"
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
        base_spatial = base_group['spatial_stats']
        exp_spatial = exp_group['spatial_stats']
        tests = self._compute_group_comparison_tests(base_group, exp_group)

        count_tests = tests['count']
        aggregation_tests = tests['aggregation_score']
        uniformity_tests = tests['uniformity_score']
        nn_tests = tests['nearest_neighbor_cv']
        grid_tests = tests['grid_density_cv']

        count_ratio = ((exp_count['mean'] - base_count['mean']) / base_count['mean'] * 100.0) if base_count['mean'] > 0 else 0.0
        aggregation_diff = base_spatial['aggregation_score']['mean'] - exp_spatial['aggregation_score']['mean']
        uniformity_diff = exp_spatial['uniformity_score']['mean'] - base_spatial['uniformity_score']['mean']
        evidence_significant = (
            self._is_test_significant(aggregation_tests) or
            self._is_test_significant(uniformity_tests)
        )

        strong_exp_advantage = aggregation_diff >= 5.0 and uniformity_diff >= 5.0 and evidence_significant
        trend_exp_advantage = aggregation_diff > 0 and uniformity_diff > 0
        strong_base_advantage = aggregation_diff <= -5.0 and uniformity_diff <= -5.0 and evidence_significant
        trend_base_advantage = aggregation_diff < 0 and uniformity_diff < 0

        if strong_exp_advantage:
            conclusion = (
                f"结论: 实验组显著优于base组（数量{count_ratio:+.1f}%，均匀性+{uniformity_diff:.1f}分，"
                f"团聚风险-{aggregation_diff:.1f}分）。"
            )
        elif trend_exp_advantage:
            conclusion = (
                f"结论: 实验组趋势更优（数量{count_ratio:+.1f}%，均匀性+{uniformity_diff:.1f}分，"
                f"团聚风险-{aggregation_diff:.1f}分），但统计证据有限。"
            )
        elif strong_base_advantage:
            conclusion = (
                f"结论: base组显著优于实验组（数量{count_ratio:+.1f}%，均匀性{uniformity_diff:.1f}分，"
                f"团聚风险{aggregation_diff:+.1f}分）。"
            )
        elif trend_base_advantage:
            conclusion = (
                f"结论: base组趋势更优（数量{count_ratio:+.1f}%，均匀性{uniformity_diff:.1f}分，"
                f"团聚风险{aggregation_diff:+.1f}分），但统计证据有限。"
            )
        else:
            conclusion = "结论: 两组在数量与分布上未形成同向优势，当前更适合解读为趋势对比而非明确胜负。"

        lines: List[str] = []
        if note:
            lines.extend([note, ""])

        lines.extend([
            self._format_compact_params(),
            "",
            "数量统计",
            (
                f"base组(n={base_group['image_count']}): 总计{int(round(base_count['total']))}根，"
                f"均值{base_count['mean']:.1f}±{base_count['std']:.1f}，范围{base_count['min']:.0f}~{base_count['max']:.0f}"
            ),
            (
                f"实验组(n={exp_group['image_count']}): 总计{int(round(exp_count['total']))}根，"
                f"均值{exp_count['mean']:.1f}±{exp_count['std']:.1f}，范围{exp_count['min']:.0f}~{exp_count['max']:.0f}"
            ),
            f"差异: 实验组 {count_ratio:+.1f}% | {self._format_compact_significance(count_tests)}",
            "",
            "均匀性对比",
            self._format_metric_comparison(
                "团聚风险总分",
                "base",
                base_spatial['aggregation_score']['mean'],
                "实验",
                exp_spatial['aggregation_score']['mean'],
                direction='down',
                qualifier='0-100↓',
                precision=1,
                left_std=base_spatial['aggregation_score']['std'],
                right_std=exp_spatial['aggregation_score']['std'],
                test_result=aggregation_tests,
            ),
            self._format_metric_comparison(
                "综合均匀性",
                "base",
                base_spatial['uniformity_score']['mean'],
                "实验",
                exp_spatial['uniformity_score']['mean'],
                direction='up',
                qualifier='0-100↑',
                precision=1,
                left_std=base_spatial['uniformity_score']['std'],
                right_std=exp_spatial['uniformity_score']['std'],
                test_result=uniformity_tests,
            ),
            self._format_metric_comparison(
                "最近邻CV",
                "base",
                base_spatial['nearest_neighbor_cv']['mean'],
                "实验",
                exp_spatial['nearest_neighbor_cv']['mean'],
                direction='down',
                precision=3,
                left_std=base_spatial['nearest_neighbor_cv']['std'],
                right_std=exp_spatial['nearest_neighbor_cv']['std'],
                test_result=nn_tests,
            ),
            self._format_metric_comparison(
                "网格CV",
                "base",
                base_spatial['grid_density_cv']['mean'],
                "实验",
                exp_spatial['grid_density_cv']['mean'],
                direction='down',
                precision=3,
                left_std=base_spatial['grid_density_cv']['std'],
                right_std=exp_spatial['grid_density_cv']['std'],
                test_result=grid_tests,
            ),
            self._format_metric_comparison(
                "Moran's I",
                "base",
                base_spatial['morans_i']['mean'],
                "实验",
                exp_spatial['morans_i']['mean'],
                direction='cluster',
                precision=3,
                left_std=base_spatial['morans_i']['std'],
                right_std=exp_spatial['morans_i']['std'],
            ),
            "注: ↑越大越好，↓越小越好，↑聚集表示越大越团聚，* p<0.05",
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
        left_spatial = left_result['stats'].get('spatial_distribution') or {}
        right_spatial = right_result['stats'].get('spatial_distribution') or {}
        left_uniformity = left_spatial.get('uniformity_scores') or {}
        right_uniformity = right_spatial.get('uniformity_scores') or {}

        left_count = int(left_result['stats']['count'])
        right_count = int(right_result['stats']['count'])
        count_diff = left_count - right_count
        count_ratio = (count_diff / right_count * 100.0) if right_count > 0 else 0.0

        left_uniformity_score = float(left_uniformity.get('overall', 0.0))
        right_uniformity_score = float(right_uniformity.get('overall', 0.0))
        uniformity_diff = left_uniformity_score - right_uniformity_score
        left_aggregation_score = 100.0 - left_uniformity_score
        right_aggregation_score = 100.0 - right_uniformity_score
        aggregation_diff = left_aggregation_score - right_aggregation_score

        left_better = aggregation_diff < 0 and uniformity_diff > 0
        right_better = aggregation_diff > 0 and uniformity_diff < 0

        if left_better and count_diff >= 0:
            conclusion = (
                f"结论: {left_label}整体更优（CNT{count_ratio:+.1f}%，均匀性+{uniformity_diff:.1f}分），"
                f"{right_label}团聚更明显。"
            )
        elif right_better and count_diff <= 0:
            conclusion = (
                f"结论: {right_label}整体更优（CNT{abs(count_ratio):+.1f}%，均匀性+{abs(uniformity_diff):.1f}分），"
                f"{left_label}团聚更明显。"
            )
        elif left_better:
            conclusion = f"结论: {left_label}分布更均匀，但数量优势未与均匀性完全同向。"
        elif right_better:
            conclusion = f"结论: {right_label}分布更均匀，但数量优势未与均匀性完全同向。"
        else:
            conclusion = "结论: 两图在数量与分布上各有优劣，尚未形成明确的单边优势。"

        lines: List[str] = []
        if note:
            lines.extend([note, ""])

        lines.extend([
            self._format_compact_params(),
            "",
            f"{left_label}: {left_result['name']} | CNT={left_count}",
            f"{right_label}: {right_result['name']} | CNT={right_count}",
            f"数量差异: {left_label if count_diff >= 0 else right_label} {abs(count_ratio):+.1f}%",
            "",
            "分布对比",
            self._format_metric_comparison(
                "团聚风险总分",
                left_label,
                left_aggregation_score,
                right_label,
                right_aggregation_score,
                direction='down',
                qualifier='0-100↓',
                precision=1,
            ),
            self._format_metric_comparison(
                "综合均匀性",
                left_label,
                left_uniformity_score,
                right_label,
                right_uniformity_score,
                direction='up',
                qualifier='0-100↑',
                precision=1,
            ),
            self._format_metric_comparison(
                "最近邻CV",
                left_label,
                float(left_spatial.get('nearest_neighbor_cv', 0.0)),
                right_label,
                float(right_spatial.get('nearest_neighbor_cv', 0.0)),
                direction='down',
                precision=3,
            ),
            self._format_metric_comparison(
                "网格CV",
                left_label,
                float(left_spatial.get('grid_density_cv', 0.0)),
                right_label,
                float(right_spatial.get('grid_density_cv', 0.0)),
                direction='down',
                precision=3,
            ),
            self._format_metric_comparison(
                "Moran's I",
                left_label,
                float(left_spatial.get('morans_i', 0.0)),
                right_label,
                float(right_spatial.get('morans_i', 0.0)),
                direction='cluster',
                precision=3,
            ),
            "注: ↑越大越好，↓越小越好，↑聚集表示越大越团聚",
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
            text="所有模式都会严格使用当前界面上的同一组识别参数；区别只在于选图方式和统计粒度。",
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
                "手动选择两张图，在相同识别条件下比较 CNT 数量、均匀性指标和典型CNT分布。",
                "选择两张图",
                self._compare_two_images,
            ),
            (
                "组别统计对比",
                "分别选择 base 组和实验组的多张图，输出组均值、波动范围、显著性检验和典型CNT分布。",
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
            base_results, base_failures = self._analyze_image_files(base_paths, "base组")
            exp_results, exp_failures = self._analyze_image_files(exp_paths, "实验组")

            failures = base_failures + exp_failures
            base_group = self._summarize_group_results("base组", base_results)
            exp_group = self._summarize_group_results("实验组", exp_results)

            note = "本次组别对比严格使用当前界面上的同一组识别条件；第一组按 base组 统计，第二组按 实验组 统计。"
            self._show_group_comparison_window(base_group, exp_group, note, failures)
        except Exception as e:
            logger.exception("组别对比失败")
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
            left_result = self._analyze_image_file(left_path, include_visualization=True)
            right_result = self._analyze_image_file(right_path, include_visualization=True)
            note = "本次双图对比严格使用当前界面上的同一组识别条件。"
            self._show_comparison_window(left_result, right_result, "图像A", "图像B", note)
        except Exception as e:
            logger.exception("双图对比失败")
            messagebox.showerror("错误", f"双图对比失败: {e}")

    # ===== 保存和导出 =====
    def _save_results(self):
        """保存分析结果"""
        measurements = self.current_roi.measurements if self.current_roi else self.analyzer.measurements

        if not measurements:
            messagebox.showwarning("警告", "没有可保存的结果！")
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
        measurements = self.current_roi.measurements if self.current_roi else self.analyzer.measurements

        if not measurements:
            messagebox.showwarning("警告", "没有可导出的结果！")
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
