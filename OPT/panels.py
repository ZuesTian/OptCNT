"""
面板模块 - 包含各个功能面板类
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, List
from datetime import datetime

from widgets import SortableTreeview, ScrollableFrame
from utils import SCALE_BAR_DEFAULT_UM

# 常量定义
MIN_ROI_SIZE = 10  # ROI最小尺寸 (像素)
MIN_SCALE_LENGTH = 5  # 比例尺最小长度 (像素)


class ControlPanel(ttk.Frame):
    """控制面板 - 包含文件操作、比例尺设置、ROI管理、预处理参数"""

    def __init__(self, parent: tk.Widget, colors: dict, callbacks: dict, variables: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.callbacks = callbacks
        self.variables = variables

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

        ttk.Button(scale_frame, text="🖱️ 图上选择比例尺",
                   style='Accent.TButton',
                   command=self.callbacks.get('select_scale')).pack(fill=tk.X, padx=8, pady=5)

        ttk.Label(scale_frame, text="或手动输入:").pack(anchor=tk.W, padx=8, pady=2)

        ttk.Label(scale_frame, text="像素数:").pack(anchor=tk.W, padx=8)
        ttk.Entry(scale_frame, textvariable=self.variables.get('scale_pixels'),
                  width=15).pack(fill=tk.X, padx=8, pady=2)

        ttk.Label(scale_frame, text="对应微米数:").pack(anchor=tk.W, padx=8)
        ttk.Entry(scale_frame, textvariable=self.variables.get('scale_um'),
                  width=15).pack(fill=tk.X, padx=8, pady=2)

        ttk.Button(scale_frame, text="应用比例尺",
                   command=self.callbacks.get('apply_scale')).pack(fill=tk.X, padx=8, pady=8)

        self.scale_label = ttk.Label(scale_frame, text=f"当前比例尺: 默认 {SCALE_BAR_DEFAULT_UM:g}μm（待应用）",
                                     foreground=self.colors['accent_primary'],
                                     font=('Segoe UI', 9, 'italic'))
        self.scale_label.pack(anchor=tk.W, padx=8, pady=5)

    def _create_roi_frame(self, parent: tk.Widget) -> None:
        """创建ROI管理框架"""
        roi_frame = ttk.LabelFrame(parent, text="ROI管理")
        roi_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(roi_frame, text="➕ 选择新ROI",
                   style='Accent.TButton',
                   command=self.callbacks.get('select_roi')).pack(fill=tk.X, padx=8, pady=5)

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
                                        font=('Segoe UI', 9))
        self.roi_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self.roi_listbox.bind('<<ListboxSelect>>', self.callbacks.get('on_select_roi'))

        btn_frame = ttk.Frame(roi_frame)
        btn_frame.pack(fill=tk.X, padx=8, pady=5)
        
        ttk.Button(btn_frame, text="❌ 删除",
                   command=self.callbacks.get('remove_roi')).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="🗑️ 清空",
                   command=self.callbacks.get('clear_rois')).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

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

        for text, value in modes:
            ttk.Radiobutton(display_frame, text=text,
                            variable=self.variables.get('display_mode'),
                            value=value,
                            command=self.callbacks.get('on_display_mode_change')).pack(anchor=tk.W, padx=12, pady=4)

    def _create_preprocess_frame(self, parent: tk.Widget) -> None:
        """创建预处理参数框架"""
        preprocess_frame = ttk.LabelFrame(parent, text="预处理参数")
        preprocess_frame.pack(fill=tk.X, padx=10, pady=8)

        # 高斯模糊
        blur_frame = ttk.Frame(preprocess_frame)
        blur_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(blur_frame, text="高斯模糊核:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.blur_label = ttk.Label(blur_frame, text="9", font=('Segoe UI', 9, 'bold'))
        self.blur_label.pack(side=tk.RIGHT)

        self.blur_scale = ttk.Scale(preprocess_frame, from_=1, to=15,
                                    variable=self.variables.get('blur_kernel'), orient=tk.HORIZONTAL,
                                    command=self.callbacks.get('on_blur_change'))
        self.blur_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        # 自适应块大小
        block_frame = ttk.Frame(preprocess_frame)
        block_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(block_frame, text="自适应块大小:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.block_label = ttk.Label(block_frame, text="15", font=('Segoe UI', 9, 'bold'))
        self.block_label.pack(side=tk.RIGHT)

        self.block_scale = ttk.Scale(preprocess_frame, from_=3, to=51,
                                     variable=self.variables.get('adaptive_block'), orient=tk.HORIZONTAL,
                                     command=self.callbacks.get('on_block_change'))
        self.block_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

        # 自适应常数C
        c_frame = ttk.Frame(preprocess_frame)
        c_frame.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(c_frame, text="自适应常数C:", foreground=self.colors['text_secondary']).pack(side=tk.LEFT)
        self.c_label = ttk.Label(c_frame, text="2", font=('Segoe UI', 9, 'bold'))
        self.c_label.pack(side=tk.RIGHT)

        self.c_scale = ttk.Scale(preprocess_frame, from_=0, to=10,
                                 variable=self.variables.get('adaptive_c'), orient=tk.HORIZONTAL,
                                 command=self.callbacks.get('on_c_change'))
        self.c_scale.pack(fill=tk.X, padx=12, pady=(0, 8))

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

        ttk.Label(filter_frame, text="粘连分离强度:", foreground=self.colors['text_secondary']).pack(anchor=tk.W)
        split_mode_box = ttk.Combobox(
            filter_frame,
            textvariable=self.variables.get('split_mode'),
            values=('关闭', '保守', '激进'),
            state='readonly'
        )
        split_mode_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(analysis_frame, text="🔍 开始检测CNT",
                   style='Danger.TButton',
                   command=self.callbacks.get('detect_cnt')).pack(fill=tk.X, padx=8, pady=10)

    def update_scale_label(self, text: str) -> None:
        """更新比例尺标签"""
        self.scale_label.config(text=text)

    def update_blur_label(self, value: str) -> None:
        """更新模糊核标签"""
        self.blur_label.config(text=value)

    def update_block_label(self, value: str) -> None:
        """更新块大小标签"""
        self.block_label.config(text=value)

    def update_c_label(self, value: str) -> None:
        """更新常数C标签"""
        self.c_label.config(text=value)

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


class ImagePanel(ttk.Frame):
    """图像显示面板 - 支持ROI和比例尺选择"""

    def __init__(self, parent: tk.Widget, colors: dict, callbacks: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        self.callbacks = callbacks
        self.canvas: Optional[tk.Canvas] = None
        self._image_origin = (0.0, 0.0)  # 图像在画布坐标系中的左上角
        self._image_size = (0.0, 0.0)    # 当前显示图像尺寸（缩放后）

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
        # 状态栏（位于底部）
        self.status_var = tk.StringVar(value="")
        self.status_bar = ttk.Label(self, textvariable=self.status_var,
                                    foreground=self.colors.get('accent_primary', '#6366F1'),
                                    background=self.colors.get('bg_tertiary', '#F1F5F9'),
                                    font=('Segoe UI', 9),
                                    padding=(8, 4))
        # 状态栏默认隐藏，有消息时才显示
        
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
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length > MIN_SCALE_LENGTH:  # 最小长度限制
                    self.on_select_complete(length)

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
                font=('Segoe UI', 10),
                tags='scale_text'
            )

    def _get_zoom_level(self) -> float:
        """获取当前缩放级别（由外部设置）"""
        return getattr(self, '_zoom_level', 1.0)

    def set_zoom_level(self, zoom: float) -> None:
        """设置当前缩放级别（供外部同步）"""
        self._zoom_level = zoom

    def _on_mousewheel(self, event) -> str:
        """鼠标滚轮缩放"""
        if self.select_mode is None:
            callback = self.callbacks.get('on_mousewheel')
            if callback:
                callback(event)
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
        children = self.winfo_children()
        if len(children) > 1:
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, before=children[1])
        else:
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def hide_status(self) -> None:
        """隐藏状态栏"""
        self.status_var.set("")
        self.status_bar.pack_forget()

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

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置UI"""
        # 统计信息（固定高度，不随窗口拉伸）
        stats_frame = ttk.LabelFrame(self, text="统计信息")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_text = tk.Text(stats_frame,
                                   height=10,
                                   bg=self.variables.get('text_bg', '#FFFFFF'),
                                   fg=self.variables.get('text_fg', '#2D3748'),
                                   relief='flat',
                                   borderwidth=1,
                                   highlightthickness=1,
                                   highlightcolor=self.colors['accent_primary'],
                                   highlightbackground=self.colors['border'],
                                   font=('Segoe UI', 9))
        self.stats_text.pack(fill=tk.X, padx=8, pady=5)
        
        self.stats_text.tag_configure('header', foreground=self.colors['accent_primary'], font=('Segoe UI', 9, 'bold'))
        self.stats_text.tag_configure('value', foreground=self.colors['accent_secondary'], font=('Segoe UI', 9, 'bold'))
        self.stats_text.tag_configure('success', foreground=self.colors['success'])
        self.stats_text.tag_configure('warning', foreground=self.colors['warning'])
        self.stats_text.tag_configure('error', foreground=self.colors['error'])

        # 测量列表（占据剩余空间）
        list_frame = ttk.LabelFrame(self, text="测量列表 (点击列标题排序)")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ('ID', '长度(μm)')
        self.tree = SortableTreeview(list_frame, columns=columns, show='headings')

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80 if col == 'ID' else 120)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind('<<TreeviewSelect>>', self.callbacks.get('on_select_cnt'))

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


class AdvancedAnalysisPanel(ttk.Frame):
    """高级分析面板 - 显示统计详情和分布图表"""

    def __init__(self, parent: tk.Widget, colors: dict, **kwargs):
        super().__init__(parent, **kwargs)
        self.colors = colors
        # 图表容器
        self.chart_frames = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """设置UI - 直接显示分布图表（无需Tab切换）"""
        # 使用 ScrollableFrame 容纳多个图表
        scrollable_dist = ScrollableFrame(self, bg_color=self.colors['bg_secondary'])
        scrollable_dist.pack(fill=tk.BOTH, expand=True)
        self.adv_dist_inner = scrollable_dist.get_inner_frame()

        # 初始化图表容器
        self._create_chart_container("histogram", "长度分布直方图")
        self._create_chart_container("pie", "长度占比饼状图")
        self._create_chart_container("cluster", "聚类分析 (长度 vs 宽度)")

    def _create_chart_container(self, key: str, title: str) -> None:
        """创建单个图表的容器"""
        container = ttk.Frame(self.adv_dist_inner, style='Card.TFrame')
        container.pack(fill=tk.X, expand=False, padx=10, pady=10)
        container.configure(height=420)
        container.pack_propagate(False)
        
        # 标题带颜色
        title_colors = {
            "histogram": self.colors['accent_primary'],
            "pie": self.colors['accent_secondary'],
            "cluster": self.colors['accent_tertiary']
        }
        title_color = title_colors.get(key, self.colors['text_primary'])
        ttk.Label(container, text=title, font=('Segoe UI', 10, 'bold'), 
                  foreground=title_color).pack(anchor=tk.W, padx=5, pady=5)
        
        # 图表区域
        chart_area = ttk.Frame(container, style='Card.TFrame')
        chart_area.pack(fill=tk.BOTH, expand=True)
        
        self.chart_frames[key] = chart_area

    def get_chart_frame(self, key: str) -> ttk.Frame:
        """获取指定图表的容器框架"""
        return self.chart_frames.get(key)

    def refresh_layout(self):
        """刷新布局，确保滚动区域正确"""
        self.update_idletasks()
        if hasattr(self, 'adv_dist_inner'):
            # 强制更新滚动区域
            canvas = self.adv_dist_inner.master
            if isinstance(canvas, tk.Canvas):
                canvas.configure(scrollregion=canvas.bbox("all"))
