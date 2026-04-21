"""
面板模块 - 包含各个功能面板类
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List
from datetime import datetime

from .widgets import SortableTreeview, ScrollableFrame
from .gui_styles import get_platform_font, get_cjk_font, scale_ui_value
from ..core.utils import (
    SCALE_BAR_DEFAULT_UM,
    CNT_BRIDGE_STRENGTH_DEFAULT,
    CNT_BRIDGE_STRENGTH_MAX,
    CNT_MERGE_DISTANCE_DEFAULT_PX,
    CNT_MERGE_DISTANCE_MAX_PX,
)

# 常量定义
MIN_ROI_SIZE = 10  # ROI最小尺寸 (像素)
MIN_SCALE_LENGTH = 5  # 比例尺最小长度 (像素)


def _scale_px(widget: tk.Widget, value: float, minimum: int = 1) -> int:
    """按当前顶层窗口 DPI 将逻辑尺寸换算为实际像素。"""
    return scale_ui_value(widget.winfo_toplevel(), value, minimum)


def _calculate_result_split_height(widget: tk.Widget, total_height: int) -> int:
    """按面板高度返回更稳定的统计区目标高度。

    策略：
    - 小窗口保持统计区可读，避免摘要被压扁。
    - 中等窗口适度按比例增长。
    - 大窗口/最大化时给统计区设置封顶，让列表继续占主导。
    """
    total_height = max(1, int(total_height))
    min_stats_height = _scale_px(widget, 220)
    preferred_stats_height = _scale_px(widget, 280)
    min_list_height = _scale_px(widget, 300)

    medium_threshold = _scale_px(widget, 760)
    large_threshold = _scale_px(widget, 980)

    if total_height >= large_threshold:
        target_ratio = 0.31
        max_stats_cap = _scale_px(widget, 360)
    elif total_height >= medium_threshold:
        target_ratio = 0.34
        max_stats_cap = _scale_px(widget, 330)
    else:
        target_ratio = 0.38
        max_stats_cap = _scale_px(widget, 320)

    desired_height = max(preferred_stats_height, int(round(total_height * target_ratio)))
    available_cap = max(min_stats_height, total_height - min_list_height)
    return max(min_stats_height, min(desired_height, max_stats_cap, available_cap))


class ControlPanel(ttk.Frame):
    """控制面板 - 包含文件操作、比例尺设置、ROI管理、预处理参数"""

    def __init__(self, parent: tk.Widget, colors: dict, callbacks: dict, variables: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.callbacks = callbacks
        self.variables = variables
        # 平台自适应字体
        self._ui_font = get_platform_font(self.winfo_toplevel())
        self.display_mode_buttons: List[ttk.Radiobutton] = []
        self.select_scale_button: Optional[ttk.Button] = None
        self.apply_scale_button: Optional[ttk.Button] = None
        self.select_roi_button: Optional[ttk.Button] = None
        self.remove_roi_button: Optional[ttk.Button] = None
        self.clear_rois_button: Optional[ttk.Button] = None
        self.auto_suggest_button: Optional[ttk.Button] = None
        self.detect_button: Optional[ttk.Button] = None
        self.live_preview_checkbutton: Optional[ttk.Checkbutton] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置UI"""
        # 创建可滚动框架
        scrollable = ScrollableFrame(self, bg_color=self.colors['bg_secondary'])
        scrollable.pack(fill=tk.BOTH, expand=True)
        control_frame = scrollable.get_inner_frame()

        # 比例尺设置
        self._create_scale_frame(control_frame)

        # ROI管理
        self._create_roi_frame(control_frame)

        # 显示模式
        self._create_display_frame(control_frame)

        # 预处理参数
        self._create_preprocess_frame(control_frame)

        # 分析按钮
        self._create_analysis_frame(control_frame)

    def _create_scale_frame(self, parent: tk.Widget) -> None:
        """创建比例尺设置框架"""
        scale_frame = ttk.LabelFrame(parent, text="比例尺设置")
        scale_frame.pack(fill=tk.X, padx=10, pady=8)

        self.select_scale_button = ttk.Button(
            scale_frame,
            text="🖱️ 图上选择比例尺",
            style='Accent.TButton',
            command=self.callbacks.get('select_scale'),
        )
        self.select_scale_button.pack(fill=tk.X, padx=8, pady=5)

        ttk.Label(scale_frame, text="或手动输入:").pack(anchor=tk.W, padx=8, pady=2)

        ttk.Label(scale_frame, text="像素数:").pack(anchor=tk.W, padx=8)
        ttk.Entry(scale_frame, textvariable=self.variables.get('scale_pixels'),
                  width=15).pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(scale_frame, text="对应微米数:").pack(anchor=tk.W, padx=8)
        ttk.Entry(scale_frame, textvariable=self.variables.get('scale_um'),
                  width=15).pack(fill=tk.X, padx=8, pady=2)

        self.apply_scale_button = ttk.Button(
            scale_frame,
            text="应用比例尺",
            command=self.callbacks.get('apply_scale'),
        )
        self.apply_scale_button.pack(fill=tk.X, padx=8, pady=8)

        self.scale_label = ttk.Label(scale_frame, text=f"当前比例尺: 默认 {SCALE_BAR_DEFAULT_UM:g}μm（待应用）",
                                     foreground=self.colors['accent_primary'],
                                     font=(self._ui_font, 9, 'italic'))
        self.scale_label.pack(anchor=tk.W, padx=8, pady=5)

        self.scale_status_label = ttk.Label(
            scale_frame,
            text="比例尺状态: 待检测",
            foreground=self.colors['text_secondary'],
            font=(self._ui_font, 9),
            wraplength=_scale_px(self, 230),
            justify=tk.LEFT,
        )
        self.scale_status_label.pack(anchor=tk.W, padx=8, pady=(0, 6))

    def _create_roi_frame(self, parent: tk.Widget) -> None:
        """创建ROI管理框架"""
        roi_frame = ttk.LabelFrame(parent, text="ROI管理")
        roi_frame.pack(fill=tk.X, padx=10, pady=8)

        self.select_roi_button = ttk.Button(
            roi_frame,
            text="➕ 选择新ROI",
            style='Accent.TButton',
            command=self.callbacks.get('select_roi'),
        )
        self.select_roi_button.pack(fill=tk.X, padx=8, pady=5)

        ttk.Label(roi_frame, text="已选择的ROI:").pack(anchor=tk.W, padx=8, pady=2)

        self.roi_listbox = tk.Listbox(roi_frame, height=6,
                                        bg=self.variables.get('listbox_bg', '#FFFFFF'),
                                        fg=self.variables.get('listbox_fg', '#2D3748'),
                                        selectbackground=self.variables.get('listbox_select_bg', '#E0E7FF'),
                                        selectforeground=self.variables.get('listbox_select_fg', '#2D3748'),
                                        relief='flat',
                                        borderwidth=1,
                                        highlightthickness=1,
                                        highlightcolor=self.colors['accent_primary'],
                                        highlightbackground=self.colors['border'],
                                        font=(self._ui_font, 9))
        self.roi_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self.roi_listbox.bind('<<ListboxSelect>>', self.callbacks.get('on_select_roi'))

        btn_frame = ttk.Frame(roi_frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=5)
        
        self.remove_roi_button = ttk.Button(
            btn_frame,
            text="❌ 删除",
            command=self.callbacks.get('remove_roi'),
        )
        self.remove_roi_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.clear_rois_button = ttk.Button(
            btn_frame,
            text="🗑️ 清空",
            command=self.callbacks.get('clear_rois'),
        )
        self.clear_rois_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    def _create_display_frame(self, parent: tk.Widget) -> None:
        """创建显示模式框架"""
        display_frame = ttk.LabelFrame(parent, text="显示模式")
        display_frame.pack(fill=tk.X, padx=10, pady=8)

        modes = [
            ("原图", "original"),
            ("二值图", "binary"),
            ("检测结果", "result"),
            ("检测+骨架", "skeleton"),
            ("实时骨架预览", "skeleton_preview")
        ]

        self.display_mode_buttons.clear()
        for text, value in modes:
            radio = ttk.Radiobutton(
                display_frame,
                text=text,
                variable=self.variables.get('display_mode'),
                value=value,
                command=self.callbacks.get('on_display_mode_change'),
            )
            radio.pack(anchor=tk.W, padx=12, pady=4)
            self.display_mode_buttons.append(radio)

    def _create_preprocess_frame(self, parent: tk.Widget) -> None:
        """创建预处理参数框架"""
        preprocess_frame = ttk.LabelFrame(parent, text="预处理参数")
        preprocess_frame.pack(fill=tk.X, padx=10, pady=8)

        # 高斯模糊
        blur_frame = ttk.Frame(preprocess_frame)
        blur_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(blur_frame, text="高斯模糊核:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.blur_label = ttk.Label(
            blur_frame,
            text=str(int(self.variables.get('blur_kernel').get()) if self.variables.get('blur_kernel') else 9),
            font=(self._ui_font, 9, 'bold'),
        )
        self.blur_label.pack(side=tk.RIGHT)

        self.blur_scale = ttk.Scale(preprocess_frame, from_=1, to=15,
                                    variable=self.variables.get('blur_kernel'), orient=tk.HORIZONTAL,
                                    command=self.callbacks.get('on_blur_change'))
        self.blur_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        # 自适应块大小
        block_frame = ttk.Frame(preprocess_frame)
        block_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(block_frame, text="自适应块大小:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.block_label = ttk.Label(
            block_frame,
            text=str(int(self.variables.get('adaptive_block').get()) if self.variables.get('adaptive_block') else 15),
            font=(self._ui_font, 9, 'bold'),
        )
        self.block_label.pack(side=tk.RIGHT)

        self.block_scale = ttk.Scale(preprocess_frame, from_=3, to=51,
                                     variable=self.variables.get('adaptive_block'), orient=tk.HORIZONTAL,
                                     command=self.callbacks.get('on_block_change'))
        self.block_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        # 自适应常数C
        c_frame = ttk.Frame(preprocess_frame)
        c_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(c_frame, text="自适应常数C:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.c_label = ttk.Label(
            c_frame,
            text=str(int(self.variables.get('adaptive_c').get()) if self.variables.get('adaptive_c') else 2),
            font=(self._ui_font, 9, 'bold'),
        )
        self.c_label.pack(side=tk.RIGHT)

        self.c_scale = ttk.Scale(preprocess_frame, from_=0, to=10,
                                 variable=self.variables.get('adaptive_c'), orient=tk.HORIZONTAL,
                                 command=self.callbacks.get('on_c_change'))
        self.c_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        bridge_frame = ttk.Frame(preprocess_frame)
        bridge_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(bridge_frame, text="桥接强度:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.bridge_label = ttk.Label(
            bridge_frame,
            text=str(int(self.variables.get('bridge_strength').get()) if self.variables.get('bridge_strength') else CNT_BRIDGE_STRENGTH_DEFAULT),
            font=(self._ui_font, 9, 'bold'),
        )
        self.bridge_label.pack(side=tk.RIGHT)

        self.bridge_scale = ttk.Scale(
            preprocess_frame,
            from_=0,
            to=CNT_BRIDGE_STRENGTH_MAX,
            variable=self.variables.get('bridge_strength'),
            orient=tk.HORIZONTAL,
            command=self.callbacks.get('on_bridge_change'))
        self.bridge_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.live_preview_checkbutton = ttk.Checkbutton(
            preprocess_frame,
            text="启用实时预览（调整参数时自动刷新）",
            variable=self.variables.get('live_preview'),
            command=self.callbacks.get('on_live_preview_toggle'),
        )
        self.live_preview_checkbutton.pack(anchor=tk.W, padx=12, pady=(0, 8))

        self.auto_suggest_button = ttk.Button(
            preprocess_frame,
            text="♻ 重新自动推荐参数",
            command=self.callbacks.get('auto_suggest_params'),
        )
        self.auto_suggest_button.pack(fill=tk.X, padx=12, pady=(0, 10))

    def _create_analysis_frame(self, parent: tk.Widget) -> None:
        """创建分析按钮框架"""
        analysis_frame = ttk.LabelFrame(parent, text="分析操作")
        analysis_frame.pack(fill=tk.X, padx=10, pady=8)

        # 过滤参数（放在按钮上方，让用户先设置再检测）
        filter_frame = ttk.Frame(analysis_frame)
        filter_frame.pack(fill=tk.X, padx=8, pady=5)

        ttk.Label(filter_frame, text="最小长度(μm):", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        ttk.Entry(filter_frame, textvariable=self.variables.get('min_length'),
                  width=10).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="最大长度(μm):", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        ttk.Entry(filter_frame, textvariable=self.variables.get('max_length'),
                  width=10).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="最小长宽比:", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        ttk.Entry(filter_frame, textvariable=self.variables.get('min_slenderness'),
                  width=10).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="识别策略:", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        profile_box = ttk.Combobox(
            filter_frame,
            textvariable=self.variables.get('detect_profile'),
            values=('严格（少误检）', '标准（推荐）', '敏感（少漏检）'),
            state='readonly'
        )
        profile_box.pack(fill=tk.X, pady=(0, 8))
        profile_box.bind('<<ComboboxSelected>>', self.callbacks.get('on_profile_change'))

        ttk.Label(filter_frame, text="粘连拆分:", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        split_mode_box = ttk.Combobox(
            filter_frame,
            textvariable=self.variables.get('split_mode'),
            values=('不拆分', '标准拆分', '强力拆分'),
            state='readonly'
        )
        split_mode_box.pack(fill=tk.X, pady=(0, 8))
        split_mode_box.bind('<<ComboboxSelected>>', self.callbacks.get('on_split_mode_change'))

        merge_frame = ttk.Frame(filter_frame)
        merge_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(merge_frame, text="近邻合并距离(px):", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.merge_distance_label = ttk.Label(
            merge_frame,
            text=str(int(self.variables.get('merge_distance_px').get()) if self.variables.get('merge_distance_px') else CNT_MERGE_DISTANCE_DEFAULT_PX),
            font=(self._ui_font, 9, 'bold'),
        )
        self.merge_distance_label.pack(side=tk.RIGHT)

        self.merge_distance_scale = ttk.Scale(
            filter_frame,
            from_=0,
            to=CNT_MERGE_DISTANCE_MAX_PX,
            variable=self.variables.get('merge_distance_px'),
            orient=tk.HORIZONTAL,
            command=self.callbacks.get('on_merge_distance_change'))
        self.merge_distance_scale.pack(fill=tk.X, pady=(0, 8))

        self.detect_button = ttk.Button(
            analysis_frame,
            text="🔍 开始检测CNT",
            style='Danger.TButton',
            command=self.callbacks.get('detect_cnt'),
        )
        self.detect_button.pack(fill=tk.X, padx=8, pady=10)

        self.analysis_status_label = ttk.Label(
            analysis_frame,
            text="检测输入状态: 待加载图像",
            foreground=self.colors['text_secondary'],
            font=(self._ui_font, 9),
            wraplength=_scale_px(self, 260),
            justify=tk.LEFT,
        )
        self.analysis_status_label.pack(anchor=tk.W, padx=8, pady=(0, 8))

    def update_scale_label(self, text: str) -> None:
        """更新比例尺标签"""
        self.scale_label.config(text=text)

    def update_scale_status(self, text: str, color: Optional[str] = None) -> None:
        """更新比例尺状态标签"""
        self.scale_status_label.config(text=text)
        if color:
            self.scale_status_label.config(foreground=color)

    def update_blur_label(self, value: str) -> None:
        """更新模糊核标签"""
        self.blur_label.config(text=value)

    def update_block_label(self, value: str) -> None:
        """更新块大小标签"""
        self.block_label.config(text=value)

    def update_c_label(self, value: str) -> None:
        """更新常数C标签"""
        self.c_label.config(text=value)

    def update_bridge_label(self, value: str) -> None:
        """更新桥接强度标签"""
        self.bridge_label.config(text=value)

    def update_merge_distance_label(self, value: str) -> None:
        """更新近邻合并距离标签"""
        self.merge_distance_label.config(text=value)

    def update_analysis_status(self, text: str, color: Optional[str] = None) -> None:
        """更新检测输入状态"""
        self.analysis_status_label.config(text=text)
        if color:
            self.analysis_status_label.config(foreground=color)

    def clear_roi_list(self) -> None:
        """清空ROI列表"""
        self.roi_listbox.delete(0, tk.END)

    def add_roi_to_list(self, name: str) -> None:
        """添加ROI到列表"""
        self.roi_listbox.insert(tk.END, name)

    def get_selected_roi_index(self) -> int:
        """获取选中的ROI索引"""
        selection = self.roi_listbox.curselection()
        return selection[0] if selection else -1

    @staticmethod
    def _set_widget_enabled(widget: Optional[ttk.Widget], enabled: bool) -> None:
        """统一设置 ttk 控件启用状态。"""
        if widget is None:
            return
        if enabled:
            widget.state(['!disabled'])
        else:
            widget.state(['disabled'])

    def set_interaction_state(self, *, has_image: bool, has_rois: bool) -> None:
        """根据当前上下文启用或禁用关键交互入口。"""
        self._set_widget_enabled(self.select_scale_button, has_image)
        self._set_widget_enabled(self.apply_scale_button, has_image)
        self._set_widget_enabled(self.select_roi_button, has_image)
        self._set_widget_enabled(self.remove_roi_button, has_rois)
        self._set_widget_enabled(self.clear_rois_button, has_rois)
        self._set_widget_enabled(self.auto_suggest_button, has_image)
        self._set_widget_enabled(self.detect_button, has_image)

        for radio in self.display_mode_buttons:
            self._set_widget_enabled(radio, has_image)


class ImagePanel(ttk.Frame):
    """图像显示面板 - 支持ROI和比例尺选择"""

    def __init__(self, parent: tk.Widget, colors: dict, callbacks: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.callbacks = callbacks
        self._ui_font = get_platform_font(self.winfo_toplevel())
        self.canvas: Optional[tk.Canvas] = None
        self._image_origin = (0.0, 0.0)  # 图像在画布坐标系中的左上角
        self._image_size = (0.0, 0.0)    # 当前显示图像尺寸（缩放后）
        self._is_panning = False

        # 选择模式
        self.select_mode = None  # 'roi' 或 'scale'
        self.select_start = None
        self.select_end = None
        self.select_rect_id = None
        self.select_line_id = None
        self.on_select_complete = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置UI"""
        footer_bg = self.colors.get('bg_tertiary', '#F1F5F9')
        self.zoom_var = tk.StringVar(value="缩放: 100%")
        self.status_var = tk.StringVar(value="")

        footer_frame = tk.Frame(self, bg=footer_bg, height=30)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        footer_frame.pack_propagate(False)

        left_frame = tk.Frame(footer_frame, bg=footer_bg)
        left_frame.pack(side=tk.LEFT, padx=8)

        self.zoom_label = tk.Label(
            left_frame,
            textvariable=self.zoom_var,
            bg=footer_bg,
            fg=self.colors.get('accent_primary', '#6366F1'),
            font=(self._ui_font, 9, 'bold'),
            padx=2,
        )
        self.zoom_label.pack(side=tk.LEFT, pady=4)

        self.fit_button = ttk.Button(
            left_frame,
            text="适应窗口",
            command=self.callbacks.get('fit_to_window'),
        )
        self.fit_button.pack(side=tk.LEFT, padx=(10, 0), pady=2)

        self.status_bar = tk.Label(
            footer_frame,
            textvariable=self.status_var,
            bg=footer_bg,
            fg=self.colors.get('text_secondary', '#64748B'),
            font=(self._ui_font, 9),
            anchor='w',
            padx=10,
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        canvas_frame = ttk.Frame(self, style='Card.TFrame')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.canvas = tk.Canvas(canvas_frame, bg=self.colors.get('bg_tertiary', '#EDF2F7'),
                                highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                    command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL,
                                    command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=h_scrollbar.set,
                              yscrollcommand=v_scrollbar.set)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

    # ... (中间代码省略，主要是事件处理逻辑，不需要改动) ...

    def _on_mouse_down(self, event) -> None:
        """鼠标按下"""
        if self.select_mode is None:
            return

        # 获取画布坐标
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.select_start = (x, y)
        self.select_end = None

    def _on_mouse_drag(self, event) -> None:
        """鼠标拖拽"""
        if self.select_mode is None or self.select_start is None:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.select_end = (x, y)

        if self.select_mode == 'roi':
            self._draw_roi_rect()
        elif self.select_mode == 'scale':
            self._draw_scale_line()

    def _on_mouse_up(self, event) -> None:
        """鼠标释放"""
        if self.select_mode is None or self.select_start is None:
            return

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.select_end = (x, y)

        # 完成选择
        if self.on_select_complete:
            if self.select_mode == 'roi':
                x1, y1 = self.select_start
                x2, y2 = self.select_end
                # 将画布坐标映射到图像局部坐标，并裁剪到图像范围内
                ox, oy = self._image_origin
                iw, ih = self._image_size
                if iw <= 0 or ih <= 0:
                    self.cancel_selection()
                    return

                ix1 = max(0.0, min(float(iw), x1 - ox))
                iy1 = max(0.0, min(float(ih), y1 - oy))
                ix2 = max(0.0, min(float(iw), x2 - ox))
                iy2 = max(0.0, min(float(ih), y2 - oy))

                x = min(ix1, ix2)
                y = min(iy1, iy2)
                w = abs(ix2 - ix1)
                h = abs(iy2 - iy1)
                if w > MIN_ROI_SIZE and h > MIN_ROI_SIZE:  # 最小尺寸限制
                    self.on_select_complete((int(x), int(y), int(w), int(h)))
            elif self.select_mode == 'scale':
                x1, y1 = self.select_start
                x2, y2 = self.select_end
                ox, oy = self._image_origin
                iw, ih = self._image_size
                if iw <= 0 or ih <= 0:
                    self.cancel_selection()
                    return

                ix1 = max(0.0, min(float(iw), x1 - ox))
                iy1 = max(0.0, min(float(ih), y1 - oy))
                ix2 = max(0.0, min(float(iw), x2 - ox))
                iy2 = max(0.0, min(float(ih), y2 - oy))
                length = ((ix2 - ix1) ** 2 + (iy2 - iy1) ** 2) ** 0.5
                if length > MIN_SCALE_LENGTH:  # 最小长度限制
                    self.on_select_complete({
                        'length': float(length),
                        'start': (float(ix1), float(iy1)),
                        'end': (float(ix2), float(iy2)),
                    })

        # 清除选择图形
        if self.select_rect_id:
            self.canvas.delete(self.select_rect_id)
            self.select_rect_id = None
        if self.select_line_id:
            self.canvas.delete(self.select_line_id)
            self.select_line_id = None

        self.select_start = None
        self.select_end = None
        self.select_mode = None
        self.canvas.config(cursor='')
        self.hide_status()

    def _draw_roi_rect(self) -> None:
        """绘制ROI矩形"""
        if self.select_rect_id:
            self.canvas.delete(self.select_rect_id)

        if self.select_start and self.select_end:
            x1, y1 = self.select_start
            x2, y2 = self.select_end
            self.select_rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='#00FF00',
                width=2,
                dash=(5, 5)
            )

    def _draw_scale_line(self) -> None:
        """绘制比例尺线段"""
        if self.select_line_id:
            self.canvas.delete(self.select_line_id)
        self.canvas.delete('scale_text')

        if self.select_start and self.select_end:
            x1, y1 = self.select_start
            x2, y2 = self.select_end
            self.select_line_id = self.canvas.create_line(
                x1, y1, x2, y2,
                fill='#00FF00',
                width=2
            )
            # 显示长度（同时显示原图像素，消除缩放误导）
            canvas_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            zoom = self._get_zoom_level()
            real_length = canvas_length / zoom if zoom > 0 else canvas_length
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.canvas.create_text(
                mid_x + 10, mid_y - 10,
                text=f"原图: {real_length:.1f}px",
                fill='#00FF00',
                font=(self._ui_font, 10),
                tags='scale_text'
            )

    def _get_zoom_level(self) -> float:
        """获取当前缩放级别（由外部设置）"""
        return getattr(self, '_zoom_level', 1.0)

    def set_zoom_level(self, zoom: float) -> None:
        """设置当前缩放级别（供外部同步）"""
        self._zoom_level = zoom
        if hasattr(self, 'zoom_var'):
            self.zoom_var.set(f"缩放: {zoom:.0%}")

    def _on_mousewheel(self, event) -> str:
        """鼠标滚轮缩放"""
        if self.select_mode is None:
            callback = self.callbacks.get('on_mousewheel')
            if callback:
                callback(event)
        return "break"

    def _on_pan_start(self, event) -> str:
        """中键按下开始平移画布"""
        if self.select_mode is not None:
            return "break"
        self._is_panning = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor='fleur')
        return "break"

    def _on_pan_drag(self, event) -> str:
        """中键拖动平移画布"""
        if not self._is_panning:
            return "break"
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def _on_pan_end(self, event) -> str:
        """中键释放结束平移"""
        if self._is_panning:
            self._is_panning = False
            self.canvas.config(cursor='crosshair' if self.select_mode else '')
        return "break"

    def start_roi_selection(self, on_complete: Callable) -> None:
        """开始ROI选择"""
        self.select_mode = 'roi'
        self.on_select_complete = on_complete
        self.canvas.config(cursor='crosshair')

    def start_scale_selection(self, on_complete: Callable) -> None:
        """开始比例尺选择"""
        self.select_mode = 'scale'
        self.on_select_complete = on_complete
        self.canvas.config(cursor='crosshair')

    def show_status(self, text: str) -> None:
        """显示状态栏消息"""
        self.status_var.set(text)

    def hide_status(self) -> None:
        """隐藏状态栏"""
        self.status_var.set("")

    def cancel_selection(self) -> None:
        """取消选择"""
        self.select_mode = None
        self.select_start = None
        self.select_end = None
        if self.select_rect_id:
            self.canvas.delete(self.select_rect_id)
            self.select_rect_id = None
        if self.select_line_id:
            self.canvas.delete(self.select_line_id)
            self.select_line_id = None
        self.canvas.config(cursor='')
        self.hide_status()

    def set_scroll_region(self, width: int, height: int) -> None:
        """设置滚动区域，图像小于画布时居中显示"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        # 滚动区域至少与画布一样大，保证图像可以居中
        region_w = max(width, canvas_w)
        region_h = max(height, canvas_h)
        self.canvas.configure(scrollregion=(0, 0, region_w, region_h))

    def clear_canvas(self) -> None:
        """清空画布"""
        self.canvas.delete("all")

    def set_image_actions_enabled(self, has_image: bool) -> None:
        """同步图像区交互按钮状态。"""
        if hasattr(self, 'fit_button') and self.fit_button is not None:
            if has_image:
                self.fit_button.state(['!disabled'])
            else:
                self.fit_button.state(['disabled'])

    def create_image(self, photo, center: bool = True) -> int:
        """创建图像，默认居中显示"""
        if center:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w = photo.width()
            img_h = photo.height()
            x = max(0, (canvas_w - img_w) // 2) if img_w < canvas_w else 0
            y = max(0, (canvas_h - img_h) // 2) if img_h < canvas_h else 0
            self._image_origin = (float(x), float(y))
            self._image_size = (float(img_w), float(img_h))
            return self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self._image_origin = (0.0, 0.0)
        self._image_size = (float(photo.width()), float(photo.height()))
        return self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)


class ResultPanel(ttk.Frame):
    """结果面板 - 显示统计信息和测量列表"""

    def __init__(self, parent: tk.Widget, colors: dict, callbacks: dict, variables: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.callbacks = callbacks
        self.variables = variables
        self.tree: Optional[SortableTreeview] = None
        self.stats_text: Optional[tk.Text] = None
        self._result_paned: Optional[tk.PanedWindow] = None
        self._result_layout_job: Optional[str] = None
        self._last_result_panel_height: Optional[int] = None
        self._tree_columns = ('ID', '长度(μm)', '分散CNT', '团聚CNT')

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置UI"""
        result_paned = tk.PanedWindow(
            self,
            orient=tk.VERTICAL,
            sashwidth=_scale_px(self, 6),
            bd=0,
            bg=self.colors.get('bg_primary', '#FAFBFC'),
        )
        result_paned.pack(fill=tk.BOTH, expand=True, padx=_scale_px(self, 5), pady=_scale_px(self, 5))
        self._result_paned = result_paned

        stats_frame = ttk.LabelFrame(result_paned, text="统计信息")
        result_paned.add(
            stats_frame,
            minsize=_scale_px(self, 200),
            height=_scale_px(self, 280),
        )

        stats_text_frame = ttk.Frame(stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True, padx=_scale_px(self, 8), pady=_scale_px(self, 5))

        _ui_font = get_platform_font(self.winfo_toplevel())
        self.stats_text = tk.Text(stats_text_frame,
                                   height=10,
                                   wrap=tk.WORD,
                                   bg=self.variables.get('text_bg', '#FFFFFF'),
                                   fg=self.variables.get('text_fg', '#2D3748'),
                                   relief='flat',
                                   borderwidth=1,
                                   highlightthickness=1,
                                   highlightcolor=self.colors['accent_primary'],
                                   highlightbackground=self.colors['border'],
                                   font=(_ui_font, 9))
        stats_scrollbar = ttk.Scrollbar(stats_text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text.tag_configure('header', foreground=self.colors['accent_primary'], font=(_ui_font, 9, 'bold'))
        self.stats_text.tag_configure('value', foreground=self.colors['accent_secondary'], font=(_ui_font, 9, 'bold'))
        self.stats_text.tag_configure('success', foreground=self.colors['success'])
        self.stats_text.tag_configure('warning', foreground=self.colors['warning'])
        self.stats_text.tag_configure('error', foreground=self.colors['error'])

        list_frame = ttk.LabelFrame(result_paned, text="测量列表 (点击列标题排序)")
        result_paned.add(list_frame, minsize=_scale_px(self, 300))

        self.tree = SortableTreeview(list_frame, columns=self._tree_columns, show='headings')

        # 配置列标题和列属性，统一居中对齐
        for col in self._tree_columns:
            self.tree.heading(col, text=col)
            default_width = _scale_px(self, 80 if col == 'ID' else 110)
            self.tree.column(col, width=default_width, anchor='center')

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 配置统一字体：使用平台自适应中文字体
        cjk = get_cjk_font()
        style = ttk.Style()
        style.configure('Treeview', font=(cjk, 9))
        style.configure('Treeview.Heading', font=(cjk, 9, 'bold'))

        self.tree.bind('<<TreeviewSelect>>', self.callbacks.get('on_select_cnt'))
        self.tree.bind('<Configure>', self._on_tree_resize, add='+')
        self.bind('<Configure>', self._on_result_panel_resize, add='+')
        self.after_idle(self._apply_balanced_result_split)

    def _on_result_panel_resize(self, event=None) -> None:
        """窗口高度变化后延迟重排结果区上下比例。"""
        if event is not None and event.widget is not self:
            return
        current_height = max(1, self.winfo_height())
        last_height = self._last_result_panel_height
        if last_height is not None and abs(current_height - last_height) < _scale_px(self, 24):
            return
        self._last_result_panel_height = current_height
        if self._result_layout_job is not None:
            self.after_cancel(self._result_layout_job)
        self._result_layout_job = self.after(80, self._apply_balanced_result_split)

    def _apply_balanced_result_split(self) -> None:
        """按照当前结果面板高度重设统计区默认占比。"""
        self._result_layout_job = None
        if self._result_paned is None or not self._result_paned.winfo_exists():
            return
        if len(self._result_paned.panes()) < 2:
            return

        total_height = max(1, self._result_paned.winfo_height())
        self._last_result_panel_height = total_height
        stats_height = _calculate_result_split_height(self, total_height)

        try:
            self._result_paned.sash_place(0, 0, stats_height)
        except tk.TclError:
            return

    def _on_tree_resize(self, event=None) -> None:
        """根据可用宽度自适应结果列表列宽。"""
        if self.tree is None:
            return

        total_width = int(event.width) if event is not None else self.tree.winfo_width()
        total_width = max(_scale_px(self, 360), total_width)

        id_min = _scale_px(self, 56)
        length_min = _scale_px(self, 104)
        status_min = _scale_px(self, 88)

        id_width = max(id_min, int(total_width * 0.16))
        length_width = max(length_min, int(total_width * 0.34))
        dispersed_width = max(status_min, int(total_width * 0.25))
        agglomerated_width = max(status_min, total_width - id_width - length_width - dispersed_width - _scale_px(self, 8))

        self.tree.column(self._tree_columns[0], width=id_width, minwidth=id_min, stretch=False, anchor='center')
        self.tree.column(self._tree_columns[1], width=length_width, minwidth=length_min, stretch=True, anchor='center')
        self.tree.column(self._tree_columns[2], width=dispersed_width, minwidth=status_min, stretch=True, anchor='center')
        self.tree.column(self._tree_columns[3], width=agglomerated_width, minwidth=status_min, stretch=True, anchor='center')

    def clear_stats(self) -> None:
        """清空统计信息"""
        self.stats_text.delete(1.0, tk.END)

    def set_stats(self, text: str) -> None:
        """设置统计信息"""
        self.stats_text.insert(tk.END, text)

    def clear_tree(self) -> None:
        """清空树形列表"""
        self.tree.clear_data()

    def add_measurement(self, values: tuple) -> None:
        """添加测量数据"""
        self.tree.insert_data(values)


class ScrollableDashboardPanel(ttk.Frame):
    """可滚动的通用分析面板基类"""

    def __init__(self, parent: tk.Widget, colors: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.chart_frames = {}
        self.chart_placeholders = {}
        self.text_widgets = {}
        self.section_containers = {}
        self.scrollable_frame: Optional[ScrollableFrame] = None
        self.inner_frame: Optional[ttk.Frame] = None

        self._setup_base_ui()
        self._setup_sections()

    def _setup_base_ui(self) -> None:
        """初始化滚动区域"""
        scrollable = ScrollableFrame(self, bg_color=self.colors['bg_secondary'])
        scrollable.pack(fill=tk.BOTH, expand=True)
        self.scrollable_frame = scrollable
        self.inner_frame = scrollable.get_inner_frame()

    def _setup_sections(self) -> None:
        """由子类实现具体内容区域"""
        raise NotImplementedError

    def _create_chart_container(self,
                                key: str,
                                title: str,
                                height: int = 420,
                                placeholder: Optional[str] = None,
                                title_color: Optional[str] = None) -> None:
        """创建图表容器"""
        if self.inner_frame is None:
            return

        container = ttk.Frame(self.inner_frame, style='Card.TFrame')
        container.pack(fill=tk.X, expand=False, padx=_scale_px(self, 10), pady=_scale_px(self, 10))
        container.configure(height=_scale_px(self, height))
        container.pack_propagate(False)

        _ui_font = get_platform_font(self.winfo_toplevel())
        ttk.Label(
            container,
            text=title,
            font=(_ui_font, 10, 'bold'),
            foreground=title_color or self.colors['text_primary'],
        ).pack(anchor=tk.W, padx=_scale_px(self, 5), pady=_scale_px(self, 5))

        chart_area = ttk.Frame(container, style='Card.TFrame')
        chart_area.pack(fill=tk.BOTH, expand=True)

        if placeholder:
            ttk.Label(
                chart_area,
                text=placeholder,
                style='Card.TLabel',
                foreground=self.colors['text_secondary'],
                justify=tk.CENTER,
            ).pack(expand=True)

        self.section_containers[key] = container
        self.chart_frames[key] = chart_area
        self.chart_placeholders[key] = placeholder or ""

        if placeholder:
            self.clear_chart_content(key)

    def _create_text_container(self,
                               key: str,
                               title: str,
                               placeholder: str,
                               height: int = 240,
                               title_color: Optional[str] = None) -> None:
        """创建文本摘要容器"""
        if self.inner_frame is None:
            return

        container = ttk.Frame(self.inner_frame, style='Card.TFrame')
        container.pack(fill=tk.X, expand=False, padx=_scale_px(self, 10), pady=_scale_px(self, 10))
        container.configure(height=_scale_px(self, height))
        container.pack_propagate(False)

        _ui_font = get_platform_font(self.winfo_toplevel())
        ttk.Label(
            container,
            text=title,
            font=(_ui_font, 10, 'bold'),
            foreground=title_color or self.colors['text_primary'],
        ).pack(anchor=tk.W, padx=_scale_px(self, 5), pady=_scale_px(self, 5))

        text_frame = ttk.Frame(container, style='Card.TFrame')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=_scale_px(self, 4), pady=(0, _scale_px(self, 4)))

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            relief='flat',
            font=(get_cjk_font(), 10),
            padx=_scale_px(self, 8),
            pady=_scale_px(self, 8),
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.section_containers[key] = container
        self.text_widgets[key] = text_widget
        self.set_text_content(key, placeholder)

    def get_chart_frame(self, key: str) -> Optional[ttk.Frame]:
        """获取指定图表容器"""
        return self.chart_frames.get(key)

    def clear_chart_content(self, key: str, placeholder: Optional[str] = None) -> None:
        """清空图表容器并恢复占位态。"""
        frame = self.chart_frames.get(key)
        if frame is None:
            return

        for child in frame.winfo_children():
            child.destroy()

        placeholder_text = self.chart_placeholders.get(key, "") if placeholder is None else placeholder
        self.chart_placeholders[key] = placeholder_text or ""
        if placeholder_text:
            ttk.Label(
                frame,
                text=placeholder_text,
                style='Card.TLabel',
                foreground=self.colors['text_secondary'],
                justify=tk.CENTER,
                wraplength=_scale_px(self, 560),
            ).pack(expand=True, padx=_scale_px(self, 14), pady=_scale_px(self, 14))

    def set_section_height(self, key: str, height: int) -> None:
        """调整指定区域的高度"""
        container = self.section_containers.get(key)
        if container is None:
            return

        container.configure(height=max(120, int(height)))

    def set_text_content(self, key: str, text: str) -> None:
        """设置指定文本容器内容"""
        widget = self.text_widgets.get(key)
        if widget is None:
            return

        widget.configure(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def scroll_to_top(self) -> None:
        """滚动到面板顶部"""
        self.refresh_layout()
        if self.scrollable_frame is not None:
            self.scrollable_frame.canvas.yview_moveto(0.0)

    def scroll_to_bottom(self) -> None:
        """滚动到面板底部"""
        self.refresh_layout()
        if self.scrollable_frame is not None:
            self.scrollable_frame.canvas.yview_moveto(1.0)

    def refresh_layout(self) -> None:
        """刷新布局，确保滚动区域正确"""
        self.update_idletasks()
        if self.inner_frame is not None:
            canvas = self.inner_frame.master
            if isinstance(canvas, tk.Canvas):
                canvas.configure(scrollregion=canvas.bbox("all"))


class AdvancedAnalysisPanel(ScrollableDashboardPanel):
    """高级分析面板 - 显示单图统计详情和分布图表"""

    def _setup_sections(self) -> None:
        """设置高级分析布局"""
        self._create_chart_container(
            "score",
            "核心五指标总览",
            placeholder="完成检测后，这里会显示总CNT数量、分散比例、网格CV、团聚面积占比和P90宽度的统一总览。",
            title_color=self.colors['accent_primary'],
        )
        self._create_chart_container(
            "histogram",
            "CNT 长度分布",
            placeholder="完成检测后，这里会显示当前图像或当前 ROI 的 CNT 长度分布。",
            title_color=self.colors['accent_secondary'],
        )
        self._create_chart_container(
            "pie",
            "分散 / 团聚占比",
            placeholder="完成检测后，这里会显示分散 CNT 与团聚 CNT 的数量占比。",
            title_color=self.colors['accent_amber'],
        )
        self._create_chart_container(
            "cluster",
            "长度-宽度散点 / 聚类",
            placeholder="完成检测后，这里会显示 CNT 的长度-宽度散点与简单聚类结果。",
            title_color=self.colors['accent_primary'],
        )
        self._create_chart_container(
            "heatmap",
            "阴影团聚热图",
            placeholder="完成检测后，这里会显示阴影团聚的空间热点热图。",
            title_color=self.colors['accent_teal'],
        )


class ComparisonAnalysisPanel(ScrollableDashboardPanel):
    """对比分析面板 - 显示双图/组别对比摘要与图表"""

    def __init__(self, parent: tk.Widget, colors: dict, **kwargs):
        self.progress_container: Optional[ttk.Frame] = None
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.progress_label: Optional[tk.Label] = None
        self.progress_percent_label: Optional[tk.Label] = None
        super().__init__(parent, colors, **kwargs)

    def _setup_sections(self) -> None:
        """设置对比分析布局"""
        # 进度条容器（初始隐藏）
        self._create_progress_container()
        
        self._create_text_container(
            "comparison_summary",
            "对比分析摘要",
            '尚未执行对比分析。使用顶部"对比分析"按钮后，结果会显示在这里。',
            height=160,
            title_color=self.colors['accent_amber'],
        )
        self._create_chart_container(
            "comparison",
            "对比分析图表",
            height=1120,
            placeholder="尚未生成对比图表。",
            title_color=self.colors['accent_amber'],
        )

    def _create_progress_container(self) -> None:
        """创建进度条容器"""
        if self.inner_frame is None:
            return

        container = ttk.Frame(self.inner_frame, style='Card.TFrame')
        container.pack(fill=tk.X, expand=False, padx=_scale_px(self, 10), pady=_scale_px(self, 10))
        container.pack_forget()  # 初始隐藏

        _ui_font = get_platform_font(self.winfo_toplevel())
        ttk.Label(
            container,
            text="正在分析图像...",
            font=(_ui_font, 10, 'bold'),
            foreground=self.colors['accent_amber'],
        ).pack(anchor=tk.W, padx=_scale_px(self, 5), pady=_scale_px(self, 5))

        progress_frame = ttk.Frame(container, style='Card.TFrame')
        progress_frame.pack(fill=tk.X, padx=_scale_px(self, 12), pady=_scale_px(self, 12))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=_scale_px(self, 400),
            maximum=100,
            style='Comparison.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, ipady=4, pady=(0, 8))

        self.progress_percent_label = tk.Label(
            progress_frame,
            text="0%",
            bg=self.colors['bg_secondary'],
            fg=self.colors['accent_amber'],
            font=(_ui_font, 9, 'bold'),
        )
        self.progress_percent_label.pack(anchor=tk.E, pady=(0, 4))

        self.progress_label = tk.Label(
            progress_frame,
            text="准备开始...",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_secondary'],
            font=(_ui_font, 9, 'bold'),
        )
        self.progress_label.pack(anchor=tk.W)

        self.progress_container = container

    def show_progress(self) -> None:
        """显示进度条"""
        if self.progress_container is not None:
            summary_container = self.section_containers.get('comparison_summary')
            if summary_container is not None:
                self.progress_container.pack(fill=tk.X, expand=False, padx=10, pady=10, before=summary_container)
            else:
                self.progress_container.pack(fill=tk.X, expand=False, padx=10, pady=10)
            if self.progress_bar is not None:
                self.progress_bar['value'] = 0
            if self.progress_percent_label is not None:
                self.progress_percent_label.config(text="0%")
            if self.progress_label is not None:
                self.progress_label.config(text="准备开始...")
            self.refresh_layout()

    def hide_progress(self) -> None:
        """隐藏进度条"""
        if self.progress_container is not None:
            self.progress_container.pack_forget()
            self.refresh_layout()

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """更新进度条
        
        Args:
            current: 当前完成数量
            total: 总数量
            message: 进度消息
        """
        if self.progress_bar is not None and total > 0:
            progress = min(100, max(0, (current / total) * 100))
            self.progress_bar['value'] = progress
            if self.progress_percent_label is not None:
                self.progress_percent_label.config(text=f"{progress:.0f}%")
        
        if self.progress_label is not None:
            if message:
                self.progress_label.config(text=f"{message} ({current}/{total})")
            else:
                self.progress_label.config(text=f"已完成 {current}/{total} 张图像")
        
        self.update_idletasks()
