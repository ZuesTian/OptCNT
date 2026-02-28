"""
GUI主控制器模块 - 负责协调各个面板和核心分析功能
"""
import json
import logging
import csv
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

from models import ROIRegion, CNTMeasurement
from analyzer_core import CNTAnalyzer
from utils import DEBOUNCE_DELAY_MS, SCALE_BAR_DEFAULT_UM
from widgets import SortableTreeview
from panels import ControlPanel, ImagePanel, ResultPanel, AdvancedAnalysisPanel

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
        self.main_paned: Optional[tk.PanedWindow] = None
        
        # 图表缓存
        self._charts = {
            'histogram': {'fig': None, 'ax': None, 'canvas': None},
            'pie': {'fig': None, 'ax': None, 'canvas': None},
            'cluster': {'fig': None, 'ax': None, 'canvas': None}
        }

        # Tkinter变量
        self._init_variables()

        # 面板引用（在 _setup_ui 中初始化）
        self.control_panel: ControlPanel = None  # type: ignore[assignment]
        self.image_panel: ImagePanel = None  # type: ignore[assignment]
        self.result_panel: ResultPanel = None  # type: ignore[assignment]
        self.analysis_panel: AdvancedAnalysisPanel = None  # type: ignore[assignment]

        # 设置UI
        self._setup_ui()

        # 快捷键：从剪贴板粘贴图像
        self.root.bind_all("<Control-v>", self._paste_image_from_clipboard)
        self.root.bind_all("<Control-V>", self._paste_image_from_clipboard)

    def _init_variables(self):
        """初始化Tkinter变量"""
        self.blur_kernel_var = tk.IntVar(value=11)
        self.adaptive_block_var = tk.IntVar(value=15)
        self.adaptive_c_var = tk.IntVar(value=2)
        self.min_length_um_var = tk.DoubleVar(value=5.0)
        self.max_length_um_var = tk.DoubleVar(value=200.0)
        self.min_slenderness_var = tk.DoubleVar(value=5.0)
        self.split_mode_var = tk.StringVar(value="保守")
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
        main_paned.add(left_frame, minsize=280, width=left_width)
        self._setup_control_panel(left_frame)

        # 中间面板 - 图像显示
        center_frame = ttk.Frame(main_paned)
        center_width = int(window_width * 0.54)
        main_paned.add(center_frame, minsize=520, width=center_width)
        self._setup_center_panel(center_frame)

        # 右侧面板 - 结果面板
        right_frame = ttk.Frame(main_paned)
        right_width = int(window_width * 0.24)
        main_paned.add(right_frame, minsize=260, width=right_width)
        self._setup_result_panel(right_frame)

        # 根据窗口尺寸自动优化三栏分配：左控制/中图像/右结果
        self.root.after_idle(self._optimize_window_distribution)
        self.root.bind("<Configure>", self._on_root_resize, add="+")

    def _on_root_resize(self, event):
        """窗口尺寸变化时防抖重排三栏布局"""
        if event.widget is not self.root or self.main_paned is None:
            return
        if self._layout_job is not None:
            self.root.after_cancel(self._layout_job)
        self._layout_job = self.root.after(120, self._optimize_window_distribution)

    def _optimize_window_distribution(self):
        """自适应优化窗口分布，优先保证中间图像区域"""
        self._layout_job = None
        paned = self.main_paned
        if paned is None or not paned.winfo_exists() or len(paned.panes()) < 3:
            return

        total_w = max(1, paned.winfo_width())
        # 目标比例：左 22% / 中 54% / 右 24%
        left_w = max(280, int(total_w * 0.22))
        right_w = max(260, int(total_w * 0.24))
        center_min = 520

        center_w = total_w - left_w - right_w
        if center_w < center_min:
            shortage = center_min - center_w
            left_reducible = max(0, left_w - 260)
            reduce_left = min(shortage // 2, left_reducible)
            left_w -= reduce_left
            shortage -= reduce_left

            right_reducible = max(0, right_w - 220)
            reduce_right = min(shortage, right_reducible)
            right_w -= reduce_right

        left_sash = left_w
        right_sash = max(left_sash + 120, total_w - right_w)
        right_sash = min(right_sash, total_w - 1)

        try:
            paned.sash_place(0, left_sash, 0)
            paned.sash_place(1, right_sash, 0)
        except tk.TclError:
            return

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
            'min_length': self.min_length_um_var,
            'max_length': self.max_length_um_var,
            'min_slenderness': self.min_slenderness_var,
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

        # 图像显示标签页
        image_tab = ttk.Frame(self.center_notebook, style='Card.TFrame')
        self.center_notebook.add(image_tab, text="图像显示")
        self._setup_image_panel(image_tab)

        # 高级分析标签页
        analysis_tab = ttk.Frame(self.center_notebook, style='Card.TFrame')
        self.center_notebook.add(analysis_tab, text="高级分析")
        self._setup_advanced_analysis_panel(analysis_tab)

    def _setup_image_panel(self, parent):
        """设置图像显示面板"""
        callbacks = {
            'on_mousewheel': self._on_mousewheel,
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

    # ===== 文件操作 =====
    def _load_image_common(self):
        """加载图像后的通用流程"""
        self._reset_display()
        self._update_display()

        # 比例尺默认使用 SCALE_BAR_DEFAULT_UM（OCR 识别值仅作为提示信息）
        scale_info = self.analyzer.detect_scale_bar()
        self.scale_um_var.set(SCALE_BAR_DEFAULT_UM)
        if scale_info:
            self.scale_pixels_var.set(scale_info['pixels'])
            ocr_um = scale_info.get('micrometers')
            if ocr_um is not None:
                messagebox.showinfo(
                    "比例尺检测",
                    f"检测到比例尺长度: {scale_info['pixels']:.1f}像素\n"
                    f"默认按 {SCALE_BAR_DEFAULT_UM:g}μm 处理（OCR识别值: {ocr_um}μm，仅供参考）\n"
                    f"请确认后点击'应用比例尺'"
                )
            else:
                messagebox.showinfo(
                    "比例尺检测",
                    f"检测到比例尺长度: {scale_info['pixels']:.1f}像素\n"
                    f"默认按 {SCALE_BAR_DEFAULT_UM:g}μm 处理，请按实际情况修改后点击'应用比例尺'"
                )
        else:
            messagebox.showwarning(
                "比例尺检测",
                f"未能自动检测到比例尺，默认已设为 {SCALE_BAR_DEFAULT_UM:g}μm，请手动确认"
            )

        # 自适应推荐预处理参数
        self._auto_suggest_params()

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
            else:
                self._draw_status_indicator('error')
                messagebox.showwarning("提示", "剪贴板内容不是图像或图像文件")
                return "break"

            self._load_image_common()
            self._draw_status_indicator('ready')
        except Exception as e:
            self._draw_status_indicator('error')
            logger.exception("粘贴图像失败")
            messagebox.showerror("错误", f"粘贴图像失败: {e}")

        return "break"

    def _reset_display(self):
        """重置显示"""
        self.zoom_level = 1.0
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
            self.analyzer.set_scale(pixels, micrometers)
            new_scale = self.analyzer.scale_um_per_pixel
            
            # 重算全局测量结果
            for m in self.analyzer.measurements:
                m.length_um = m.length_pixels * new_scale
                if m.width_mean_um is not None:
                    width_px = m.width_mean_um / old_scale if old_scale > 0 else 0
                    m.width_mean_um = width_px * new_scale
            
            # 重算所有ROI的测量结果
            for roi in self.analyzer.rois:
                for m in roi.measurements:
                    m.length_um = m.length_pixels * new_scale
                    if m.width_mean_um is not None:
                        width_px = m.width_mean_um / old_scale if old_scale > 0 else 0
                        m.width_mean_um = width_px * new_scale
            
            scale_text = f"当前比例尺: {pixels:.1f}px = {micrometers:.1f}μm " \
                        f"({self.analyzer.scale_um_per_pixel:.4f}μm/pixel)"
            self.control_panel.update_scale_label(scale_text)
            
            # 刷新结果显示
            self._update_results()
            
            messagebox.showinfo("成功", "比例尺已应用，测量结果已更新！")

        except Exception as e:
            logger.exception("应用比例尺失败")
            messagebox.showerror("错误", f"应用比例尺失败: {e}")

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
    def _auto_suggest_params(self):
        """根据图像特征自动推荐预处理参数"""
        try:
            roi = self._get_active_preprocess_roi()
            params = self.analyzer.suggest_preprocess_params(roi)

            self.blur_kernel_var.set(params['blur_kernel'])
            self.adaptive_block_var.set(params['adaptive_block'])
            self.adaptive_c_var.set(params['adaptive_c'])

            self.control_panel.update_blur_label(str(params['blur_kernel']))
            self.control_panel.update_block_label(str(params['adaptive_block']))
            self.control_panel.update_c_label(str(params['adaptive_c']))

            self._last_preprocess_signature = None
        except Exception as e:
            logger.debug(f"自适应参数推荐失败，使用默认值: {e}")

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
                threshold_invert=True,
                roi=roi_to_use
            )
            self._last_preprocess_signature = signature
            self._update_display()
        except Exception as e:
            logger.exception(f"预处理错误: {e}")

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

    def _on_c_change(self, value):
        """自适应常数C变化"""
        val = int(float(value))
        self.adaptive_c_var.set(val)
        self.control_panel.update_c_label(str(val))
        self._last_preprocess_signature = None
        if self.live_preview_var.get() and self._is_preprocess_mode():
            self._schedule_preprocessing()

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

            # 修复1: 强制校验并重算预处理，确保二值图与当前ROI一致
            current_signature = self._get_preprocess_signature()
            if self.analyzer.binary_image is None or current_signature != self._last_preprocess_signature:
                self._apply_preprocessing(force=True)

            measurements = self.analyzer.detect_cnts_hybrid(
                min_length_um=min_length,
                max_length_um=max_length,
                min_slenderness=min_slenderness,
                split_mode={
                    "关闭": "off",
                    "保守": "conservative",
                    "激进": "aggressive",
                }.get(self.split_mode_var.get(), self.split_mode_var.get()),
                roi=self.current_roi
            )

            self._update_results()
            self._update_advanced_analysis()
            self._update_display()

            roi_text = f" ({self.current_roi.name})" if self.current_roi else ""
            messagebox.showinfo("检测完成",
                                f"在{roi_text if self.current_roi else '全图'}中检测到 {len(measurements)} 个CNT")

        except Exception as e:
            logger.exception("CNT检测失败")
            messagebox.showerror("错误", f"CNT检测失败: {e}")

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
                for r in self.analyzer.rois:
                    cv2.rectangle(image, (r.x, r.y), (r.x + r.width, r.y + r.height),
                                  r.color, 2)
                    cv2.putText(image, r.name, (r.x + 5, r.y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, r.color, 2)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif mode == "binary":
                if self.analyzer.binary_image is not None:
                    overlay = self.analyzer.image.copy()
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

        except Exception as e:
            logger.exception(f"显示更新错误: {e}")

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
        
        # 强制刷新布局
        self.analysis_panel.refresh_layout()

    def _init_chart(self, key: str, figsize=(6, 4)):
        """初始化或获取图表对象"""
        chart = self._charts[key]
        if chart['fig'] is None:
            frame = self.analysis_panel.get_chart_frame(key)
            if frame:
                chart['fig'] = Figure(figsize=figsize, dpi=100)
                chart['fig'].patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])
                chart['ax'] = chart['fig'].add_subplot(111)
                chart['canvas'] = FigureCanvasTkAgg(chart['fig'], master=frame)
                chart['canvas'].get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        else:
            chart['ax'].clear()
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

            max_len = max(lengths)
            min_len = min(lengths)

            # 动态分箱：避免固定到 200μm 导致长样本全部落在分箱外，从而“看不到柱形”
            if max_len <= 200 and min_len >= 0:
                bins = [0, 5, 10, 15, 20, 30, 50, 100, 200]
            else:
                right = max_len * 1.05 if max_len > 0 else 1.0
                left = min(0.0, min_len)
                if right <= left:
                    right = left + 1.0
                bins = np.linspace(left, right, 12)

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
                    data = {
                        'roi': self.current_roi.name if self.current_roi else "Full Image",
                        'statistics': {
                            'count': int(stats['count']),
                            'length_mean': float(stats['length_mean']),
                            'length_std': float(stats['length_std']),
                            'length_min': float(stats['length_min']),
                            'length_max': float(stats['length_max']),
                            'scale_um_per_pixel': float(self.analyzer.scale_um_per_pixel)
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

            except Exception as e:
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

            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {e}")
