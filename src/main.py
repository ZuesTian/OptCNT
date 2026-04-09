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

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数"""
    # Windows DPI 感知，确保高分辨率屏幕下 UI 清晰
    if sys.platform == 'win32':
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    root = tk.Tk()

    # 创建应用（主题在 CNTAnalyzerGUI._apply_modern_style 中设置）
    app = CNTAnalyzerGUI(root)

    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    freeze_support()
    main()
