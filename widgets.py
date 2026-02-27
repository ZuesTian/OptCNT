"""
自定义控件模块 - 包含可复用的UI控件类
"""
import tkinter as tk
from tkinter import ttk


class SortableTreeview(ttk.Treeview):
    """可排序的Treeview类，支持现代化样式"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.sort_column = None
        self.sort_reverse = False
        self.data = []
        self.bind('<Button-1>', self._on_heading_click)
        
        # 设置斑马纹效果
        self.tag_configure('oddrow', background='#F7F9FC')
        self.tag_configure('evenrow', background='#FFFFFF')

    def _on_heading_click(self, event):
        """处理列标题点击事件"""
        region = self.identify_region(event.x, event.y)
        if region == 'heading':
            column = self.identify_column(event.x)
            if column == "#0":
                return
            col_index = int(column[1:]) - 1
            columns = self['columns']

            if col_index < len(columns):
                clicked_column = columns[col_index]

                if self.sort_column == clicked_column:
                    self.sort_reverse = not self.sort_reverse
                else:
                    self.sort_column = clicked_column
                    self.sort_reverse = False

                self._sort_data(clicked_column, self.sort_reverse)
                
                # 更新标题显示（显示排序箭头）
                for col in columns:
                    text = self.heading(col, 'text')
                    text = text.replace(' ▲', '').replace(' ▼', '')
                    self.heading(col, text=text)
                
                current_text = self.heading(clicked_column, 'text')
                arrow = ' ▼' if self.sort_reverse else ' ▲'
                self.heading(clicked_column, text=current_text + arrow)

    @staticmethod
    def _try_float(value) -> float:
        """安全地将值转换为浮点数，失败返回0"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _sort_data(self, column, reverse=False):
        """根据列排序数据"""
        col_index = self['columns'].index(column)
        # 判断是否为数值列
        is_numeric = any(k in column for k in ['长度', '宽度', 'ID', '像素', '长宽比'])

        if is_numeric:
            sorted_data = sorted(
                self.data,
                key=lambda x: self._try_float(x[col_index]),
                reverse=reverse
            )
        else:
            sorted_data = sorted(
                self.data,
                key=lambda x: str(x[col_index]),
                reverse=reverse
            )

        self.data = sorted_data
        self._refresh_display()

    def _refresh_display(self):
        """刷新显示"""
        for item in self.get_children():
            self.delete(item)
        
        for i, row_data in enumerate(self.data):
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            self.insert('', tk.END, values=row_data, tags=(tag,))

    def insert_data(self, values):
        """插入新数据"""
        self.data.append(values)
        tag = 'evenrow' if len(self.data) % 2 == 0 else 'oddrow'
        super().insert('', tk.END, values=values, tags=(tag,))

    def clear_data(self):
        """清空所有数据"""
        self.data = []
        for item in self.get_children():
            self.delete(item)


class ScrollableFrame(ttk.Frame):
    """带滚动条的框架，现代化样式"""

    def __init__(self, parent, bg_color='#FFFFFF', *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        # 样式配置
        self.configure(style='Card.TFrame')

        # 创建画布和滚动条
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=bg_color)
        
        # 自定义滚动条样式（需要在外部Style中定义）
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        # 创建内部框架
        self.inner_frame = ttk.Frame(self.canvas, style='Card.TFrame')

        # 配置画布滚动
        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.inner_frame.bind("<Configure>", on_frame_configure)

        # 画布大小变化时也更新滚动区域
        def on_canvas_resize(event):
            # 确保内部框架至少和画布一样高，防止内容消失
            bbox = self.canvas.bbox("all")
            if bbox:
                canvas_height = event.height
                content_height = bbox[3] - bbox[1]
                if content_height < canvas_height:
                    self.canvas.configure(scrollregion=(0, 0, event.width, canvas_height))

        # 将框架放入画布
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # 绑定画布大小变化，强制内部框架宽度一致
        def on_canvas_configure(event):
            if self.canvas.winfo_exists():
                self.canvas.itemconfig(self.canvas_window, width=event.width)
                on_canvas_resize(event)

        self.canvas.bind('<Configure>', on_canvas_configure)

        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            else:
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.canvas.bind("<Button-4>", _on_mousewheel)
        self.canvas.bind("<Button-5>", _on_mousewheel)
        self.inner_frame.bind("<MouseWheel>", _on_mousewheel)
        self.inner_frame.bind("<Button-4>", _on_mousewheel)
        self.inner_frame.bind("<Button-5>", _on_mousewheel)
        self.canvas.bind("<Enter>", lambda event: self.canvas.focus_set())
        self.inner_frame.bind("<Enter>", lambda event: self.canvas.focus_set())

        # 递归绑定所有子控件的滚轮事件，解决子控件上滚动不生效的问题
        self._mousewheel_handler = _on_mousewheel
        self._bind_scheduled = False

        def _schedule_bind(event):
            """延迟绑定，避免频繁触发"""
            if not self._bind_scheduled:
                self._bind_scheduled = True
                self.after(100, self._do_bind_children)

        self.inner_frame.bind("<Map>", _schedule_bind)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _do_bind_children(self):
        """执行子控件滚轮绑定"""
        self._bind_scheduled = False
        self._bind_children_mousewheel(self.inner_frame)

    def _bind_children_mousewheel(self, widget):
        """递归绑定所有子控件的鼠标滚轮事件"""
        for child in widget.winfo_children():
            child.bind("<MouseWheel>", self._mousewheel_handler, add='+')
            child.bind("<Button-4>", self._mousewheel_handler, add='+')
            child.bind("<Button-5>", self._mousewheel_handler, add='+')
            self._bind_children_mousewheel(child)

    def get_inner_frame(self):
        """获取内部框架"""
        return self.inner_frame
