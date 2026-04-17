from __future__ import annotations

import logging
import tkinter as tk
from typing import Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..core.utils import CHART_REBUILD_DRAW_LIMIT

logger = logging.getLogger(__name__)


class ChartManager:
    """Manage GUI chart state and Tk/Matplotlib lifecycle."""

    def __init__(self, gui):
        self.gui = gui
        self.charts = self._ensure_chart_registry()

    def _build_default_registry(self) -> dict:
        return {
            'score': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'histogram': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'pie': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'cluster': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'heatmap': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
            'comparison': {'fig': None, 'ax': None, 'canvas': None, 'colorbar': None, 'draw_count': 0},
        }

    def _ensure_chart_registry(self) -> dict:
        charts = getattr(self.gui, '_charts', None)
        if charts is None:
            charts = self._build_default_registry()
            self.gui._charts = charts
            return charts

        defaults = self._build_default_registry()
        for key, default_chart in defaults.items():
            chart = charts.setdefault(key, {})
            for field, value in default_chart.items():
                chart.setdefault(field, value)
        return charts

    def get_chart(self, key: str) -> dict:
        return self._ensure_chart_registry()[key]

    def dispose_chart(self, key: str) -> None:
        chart = self.get_chart(key)
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

    def init_chart(self, key: str, figsize=(6, 4)) -> dict:
        chart = self.get_chart(key)
        analysis_panel = getattr(self.gui, 'analysis_panel', None)
        frame = analysis_panel.get_chart_frame(key) if analysis_panel is not None else None
        if not frame:
            return chart

        should_rebuild = (
            chart['fig'] is None
            or chart.get('draw_count', 0) >= CHART_REBUILD_DRAW_LIMIT
        )
        if should_rebuild:
            self.dispose_chart(key)
            for child in frame.winfo_children():
                child.destroy()
            chart['fig'] = Figure(figsize=figsize, dpi=getattr(self.gui, '_chart_dpi', 100))
            chart['fig'].patch.set_facecolor(self.gui.MODERN_COLORS['bg_secondary'])
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

    def mount_comparison_figure(self, figure: Figure, chart_frame, *, padx: int = 8, pady: int = 8):
        if chart_frame is None:
            return None

        for child in chart_frame.winfo_children():
            child.destroy()
        self.dispose_chart('comparison')
        chart = self.get_chart('comparison')
        canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
        chart['fig'] = figure
        chart['ax'] = None
        chart['canvas'] = canvas
        chart['colorbar'] = None
        chart['draw_count'] = 1
        return chart


def ensure_chart_manager(gui):
    manager = getattr(gui, 'chart_manager', None)
    if manager is None:
        manager = ChartManager(gui)
        gui.chart_manager = manager
    else:
        manager.charts = manager._ensure_chart_registry()
    return manager
