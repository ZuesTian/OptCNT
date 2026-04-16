"""
CNT图像分析系统 - 程序入口
"""
import os
import ctypes
import logging
import sys
import tkinter as tk
from multiprocessing import freeze_support
from pathlib import Path

# 解决 NumExpr 和 Joblib 核心数检测警告
# 设置最大线程数，避免自动检测失败或过高
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['LOKY_MAX_CPU_COUNT'] = '16'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from src.gui import CNTAnalyzerGUI
else:
    from .gui import CNTAnalyzerGUI

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _get_system_dpi_scale() -> float:
    """获取 Windows 系统 DPI 缩放因子（相对于 96 DPI 基准）。

    返回值 1.0 = 100%, 1.25 = 125%, 1.5 = 150%, 2.0 = 200%。
    非 Windows 或获取失败时返回 1.0。
    """
    if sys.platform != 'win32':
        return 1.0
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return max(1.0, dpi / 96.0)
    except Exception:
        return 1.0


def _find_best_cjk_font() -> str:
    """在系统已安装字体中查找最佳中文字体名称。"""
    candidates = [
        'Microsoft YaHei',   # Windows 中文
        'SimHei',            # Windows 备选
        'PingFang SC',       # macOS
        'Noto Sans CJK SC',  # Linux
        'WenQuanYi Micro Hei',  # Linux 备选
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return 'sans-serif'


def _configure_matplotlib_fonts():
    """配置 Matplotlib 中文字体，使用运行时检测而非硬编码。"""
    best_cjk = _find_best_cjk_font()
    fallback_chain = [best_cjk] if best_cjk != 'sans-serif' else []
    fallback_chain.extend([
        'Microsoft YaHei',
        'SimHei',
        'PingFang SC',
        'Noto Sans CJK SC',
        'DejaVu Sans',
    ])
    # 去重保留顺序
    seen = set()
    unique_chain = []
    for f in fallback_chain:
        if f not in seen:
            seen.add(f)
            unique_chain.append(f)

    plt.rcParams['font.sans-serif'] = unique_chain
    plt.rcParams['axes.unicode_minus'] = False
    logger.info("Matplotlib 字体回退链: %s", unique_chain)


def _detect_platform_font() -> str:
    """检测当前平台最佳 UI 字体名称，供 Tkinter 使用。"""
    if sys.platform == 'win32':
        # Segoe UI 是 Windows Vista+ 的标准 UI 字体
        return 'Segoe UI'
    elif sys.platform == 'darwin':
        return 'Helvetica Neue'
    else:
        return 'sans-serif'


# 统一 DPI 常量：所有 Figure 使用此值创建
CHART_DPI = 100


def main():
    """主函数"""
    # Windows DPI 感知，确保高分辨率屏幕下 UI 清晰
    dpi_scale = 1.0
    if sys.platform == 'win32':
        try:
            # Per-Monitor DPI Aware V1 - 兼容 Windows 8.1+
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        dpi_scale = _get_system_dpi_scale()

    # 配置 Matplotlib 字体（在创建 Figure 之前）
    _configure_matplotlib_fonts()

    root = tk.Tk()

    # 同步 Tkinter DPI 缩放
    if dpi_scale > 1.0:
        # Tkinter 默认 scaling = 1.0 对应 72 DPI
        # Windows DPI 感知模式下需要补偿系统缩放
        current_scaling = root.tk.call('tk', 'scaling')
        try:
            current_scaling = float(current_scaling)
        except (TypeError, ValueError):
            current_scaling = 1.0
        # 只在 scaling 明显偏低时修正，避免重复叠加
        if current_scaling < dpi_scale * 1.2:
            root.tk.call('tk', 'scaling', dpi_scale * 1.333)
            logger.info("Tkinter scaling 设置为 %.3f (系统 DPI 缩放 %.0f%%)",
                         dpi_scale * 1.333, dpi_scale * 100)

    # 将 DPI 信息存储在 root 上，供 GUI 模块读取
    root._dpi_scale = dpi_scale
    root._chart_dpi = CHART_DPI
    root._platform_font = _detect_platform_font()

    # 创建应用（主题在 CNTAnalyzerGUI._apply_modern_style 中设置）
    app = CNTAnalyzerGUI(root)

    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    freeze_support()
    main()
