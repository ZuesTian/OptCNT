"""
GUI样式模块 - 负责定义颜色方案和应用ttk样式

跨平台注意事项:
- 字体名称通过 get_platform_font() 在运行时检测，而非硬编码
- 字号使用相对值，配合 Tkinter DPI scaling 自动适应不同缩放比例
"""
import logging
import sys
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

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


def get_platform_font(root: tk.Tk = None) -> str:
    """获取当前平台最佳 UI 字体名称。

    优先从 root._platform_font（由 main.py 设置）读取，
    否则根据操作系统返回合理默认值。
    """
    if root is not None and hasattr(root, '_platform_font'):
        return root._platform_font
    if sys.platform == 'win32':
        return 'Segoe UI'
    elif sys.platform == 'darwin':
        return 'Helvetica Neue'
    return 'sans-serif'


def get_cjk_font(root: tk.Tk = None) -> str:
    """获取用于中文内容的字体名称（如 Treeview 数据列）。"""
    if sys.platform == 'win32':
        return 'Microsoft YaHei'
    elif sys.platform == 'darwin':
        return 'PingFang SC'
    return get_platform_font(root)


def get_ui_scale(root: tk.Tk = None) -> float:
    """获取当前界面的像素缩放因子。"""
    if root is not None:
        for attr in ('_ui_scale', '_dpi_scale'):
            value = getattr(root, attr, None)
            try:
                scale = float(value)
            except (TypeError, ValueError):
                continue
            if scale > 0:
                return max(1.0, scale)
    return 1.0


def scale_ui_value(root: tk.Tk, value: float, minimum: int = 1) -> int:
    """按 UI scale 将逻辑像素换算为当前窗口使用的像素值。"""
    return max(int(minimum), int(round(float(value) * get_ui_scale(root))))


def apply_modern_style(root: tk.Tk, colors: dict = None) -> None:
    """
    应用Modern风格样式到tkinter根窗口
    
    Args:
        root: tkinter根窗口
        colors: 可选的自定义颜色字典，默认使用MODERN_COLORS
    """
    if colors is None:
        colors = MODERN_COLORS
    
    c = colors
    root.configure(bg=c['bg_primary'])
    style = ttk.Style()

    try:
        style.theme_use('clam')
    except tk.TclError:
        logger.debug("clam主题不可用，使用默认主题")

    # 使用平台自适应字体
    ui_font = get_platform_font(root)
    cjk_font = get_cjk_font(root)
    px = lambda value, minimum=1: scale_ui_value(root, value, minimum)
    default_font = (ui_font, 9)
    heading_font = (ui_font, 10, 'bold')

    style.configure('.',
                    background=c['bg_primary'],
                    foreground=c['text_primary'],
                    font=default_font)

    style.configure('TFrame', background=c['bg_primary'])
    style.configure('Card.TFrame', background=c['bg_secondary'])

    style.configure('TLabel', background=c['bg_primary'], foreground=c['text_primary'], font=default_font)
    style.configure('Card.TLabel', background=c['bg_secondary'], foreground=c['text_primary'], font=default_font)
    style.configure('Header.TLabel', background=c['bg_secondary'], foreground=c['accent_primary'], font=(ui_font, 11, 'bold'))
    style.configure('Secondary.TLabel', background=c['bg_primary'], foreground=c['text_secondary'], font=(ui_font, 9, 'italic'))

    style.configure('TButton',
                    background=c['button_bg'],
                    foreground=c['accent_primary'],
                    borderwidth=1,
                    relief='flat',
                    font=(ui_font, 9, 'bold'),
                    padding=px(5))

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
                    font=(ui_font, 9, 'bold'),
                    padding=px(6))

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
                    font=(ui_font, 9, 'bold'),
                    padding=px(6))

    style.map('Success.TButton',
              background=[('active', '#059669'),
                          ('pressed', '#047857')])

    style.configure('Warning.TButton',
                    background=c['warning'],
                    foreground='#FFFFFF',
                    borderwidth=0,
                    relief='flat',
                    font=(ui_font, 9, 'bold'),
                    padding=px(6))

    style.map('Warning.TButton',
              background=[('active', '#D97706'),
                          ('pressed', '#B45309')])

    style.configure('Danger.TButton',
                    background=c['error'],
                    foreground='#FFFFFF',
                    borderwidth=0,
                    relief='flat',
                    font=(ui_font, 9, 'bold'),
                    padding=px(6))

    style.map('Danger.TButton',
              background=[('active', '#DC2626'),
                          ('pressed', '#B91C1C')])

    style.configure('TEntry',
                    fieldbackground=c['input_bg'],
                    foreground=c['text_primary'],
                    borderwidth=1,
                    relief='solid',
                    padding=px(5))

    style.configure('TLabelframe',
                    background=c['bg_primary'],
                    borderwidth=1,
                    relief='solid',
                    bordercolor=c['border'])

    style.configure('TLabelframe.Label',
                    background=c['bg_primary'],
                    foreground=c['accent_secondary'],
                    font=heading_font)

    style.configure('TNotebook', background=c['bg_primary'], tabmargins=[px(2), px(5), px(2), 0], borderwidth=0)
    style.configure('TNotebook.Tab',
                    background=c['bg_tertiary'],
                    foreground=c['text_secondary'],
                    padding=[px(15), px(8)],
                    font=(ui_font, 9),
                    borderwidth=0)

    style.map('TNotebook.Tab',
              background=[('selected', c['bg_secondary']),
                          ('active', c['hover_bg'])],
              foreground=[('selected', c['accent_primary']),
                          ('active', c['text_primary'])],
              expand=[('selected', [1, 1, 1, 0])])

    style.configure('TScale', background=c['bg_primary'], troughcolor=c['border'], sliderlength=px(20))

    style.configure('TScrollbar',
                    background=c['bg_tertiary'],
                    troughcolor=c['bg_primary'],
                    borderwidth=0,
                    arrowsize=px(12))
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
                    font=(cjk_font, 9),
                    rowheight=px(28))

    style.configure('Treeview.Heading',
                    background=c['bg_tertiary'],
                    foreground=c['text_secondary'],
                    font=(cjk_font, 9, 'bold'),
                    borderwidth=0,
                    relief='flat')

    style.map('Treeview.Heading',
              background=[('active', c['hover_bg'])],
              foreground=[('active', c['accent_primary'])])

    style.configure('Horizontal.TProgressbar',
                    background=c['accent_primary'],
                    troughcolor=c['bg_tertiary'],
                    thickness=px(12))
    style.configure('Comparison.Horizontal.TProgressbar',
                    background=c['accent_amber'],
                    troughcolor='#FEF3C7',
                    lightcolor='#FBBF24',
                    darkcolor='#D97706',
                    bordercolor=c['accent_amber'],
                    thickness=px(20))
