"""Extracted layout helpers for CNTAnalyzerGUI."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple
import tkinter as tk


def _get_expected_paned_width(self, root_width: int) -> int:
    """根据根窗口宽度估算主三栏可用宽度。"""
    return max(1, int(root_width) - self._scale_px(24))


def _get_paned_sash_width(self, paned: Optional[tk.PanedWindow] = None) -> int:
    """Return the sash width used by the PanedWindow."""
    paned = paned or self.main_paned
    if paned is None:
        return self._scale_px(6)
    try:
        return max(0, int(paned.cget('sashwidth')))
    except (tk.TclError, TypeError, ValueError, AttributeError):
        return self._scale_px(6)


def _get_paned_layout_width(self, paned: Optional[tk.PanedWindow] = None) -> int:
    """Return the usable pane width after subtracting sash pixels."""
    paned = paned or self.main_paned
    if paned is None or not paned.winfo_exists():
        return 1
    total_w = max(1, int(paned.winfo_width()))
    pane_count = len(paned.panes())
    sash_total = self._get_paned_sash_width(paned) * max(0, pane_count - 1)
    return max(1, total_w - sash_total)


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
    """返回统一的三栏布局配置（逻辑像素 + 稳定比例）。"""
    _ = tab_key or self._get_active_center_tab_key()
    return {
        'left_floor': 240,
        'right_floor': 320,
        'center_min': 620,
        'left_ratio': 0.17,
        'right_ratio': 0.30,
        'left_ratio_cap': 0.24,
        'right_ratio_cap': 0.30,
    }


def _calculate_pane_widths(self, total_w: int, tab_key: Optional[str] = None) -> Tuple[int, int, int]:
    """按统一比例和最小宽度计算左/中/右三栏宽度。"""
    profile = self._get_pane_layout_profile(tab_key)
    total_w = max(1, int(total_w))
    left_floor = self._scale_px(profile['left_floor'])
    right_floor = self._scale_px(profile['right_floor'])
    center_min = self._scale_px(profile['center_min'])
    left_ratio = float(profile.get('left_ratio', 0.0) or 0.0)
    right_ratio = float(profile.get('right_ratio', 0.0) or 0.0)
    left_ratio_cap = float(profile.get('left_ratio_cap', 0.24) or 0.24)
    right_ratio_cap = float(profile.get('right_ratio_cap', 0.28) or 0.28)

    if left_ratio <= 0.0 or right_ratio <= 0.0 or (left_ratio + right_ratio) >= 0.85:
        left_ratio = 0.17
        right_ratio = 0.30

    left_ratio_cap = max(left_ratio, min(left_ratio_cap, 0.45))
    right_ratio_cap = max(right_ratio, min(right_ratio_cap, 0.30))

    # Relax hard side-pane floors on narrower windows so the result pane does
    # not crowd out the center workspace before the window reaches its minimum.
    left_floor = min(left_floor, max(self._scale_px(180), int(total_w * left_ratio_cap)))
    right_floor = min(right_floor, max(self._scale_px(220), int(total_w * right_ratio_cap)))

    left_w = max(left_floor, int(round(total_w * left_ratio)))
    right_w = max(right_floor, int(round(total_w * right_ratio)))
    left_cap_width = max(left_floor, int(round(total_w * left_ratio_cap)))
    right_cap_width = max(right_floor, int(round(total_w * right_ratio_cap)))
    left_w = min(left_w, left_cap_width)
    right_w = min(right_w, right_cap_width)
    center_w = total_w - left_w - right_w

    if center_w < center_min:
        shortage = center_min - center_w
        reducible_left = max(0, left_w - left_floor)
        reducible_right = max(0, right_w - right_floor)
        total_reducible = reducible_left + reducible_right

        if total_reducible > 0:
            reclaim_left = min(reducible_left, int(round(shortage * (reducible_left / total_reducible))))
            reclaim_right = min(reducible_right, shortage - reclaim_left)
            reclaimed = reclaim_left + reclaim_right
            remaining = shortage - reclaimed

            if remaining > 0:
                extra_right = min(remaining, reducible_right - reclaim_right)
                reclaim_right += extra_right
                remaining -= extra_right
            if remaining > 0:
                extra_left = min(remaining, reducible_left - reclaim_left)
                reclaim_left += extra_left

            left_w -= reclaim_left
            right_w -= reclaim_right
        center_w = total_w - left_w - right_w

    if center_w < center_min:
        left_w = left_floor
        right_w = right_floor
        center_w = total_w - left_w - right_w

    return left_w, center_w, right_w


def _apply_pane_widths(self, paned: Optional[tk.PanedWindow] = None, tab_key: Optional[str] = None) -> bool:
    """Apply calculated widths and sash positions to the three-pane layout."""
    paned = paned or self.main_paned
    if paned is None or not paned.winfo_exists() or len(paned.panes()) < 3:
        return False

    total_w = max(1, int(paned.winfo_width()))
    sash_width = self._get_paned_sash_width(paned)
    layout_w = self._get_paned_layout_width(paned)
    left_w, center_w, right_w = self._calculate_pane_widths(layout_w, tab_key)
    left_sash = left_w
    right_sash = max(left_sash + sash_width, total_w - right_w - sash_width)

    try:
        paned.sash_place(0, left_sash, 0)
        paned.sash_place(1, min(right_sash, total_w - sash_width), 0)
    except tk.TclError:
        return False

    return True


def _optimize_window_distribution(self):
    """使用目标宽度约束优化窗口分布。"""
    self._layout_job = None
    paned = self.main_paned
    if paned is None or not paned.winfo_exists() or len(paned.panes()) < 3:
        return

    total_w = max(1, paned.winfo_width())
    root_w = max(1, self.root.winfo_width())
    expected_paned_w = self._get_expected_paned_width(root_w)
    width_gap = abs(total_w - expected_paned_w)
    stabilization_tolerance = max(self._scale_px(48), int(expected_paned_w * 0.04))
    current_snapshot = (root_w, total_w)
    previous_snapshot = getattr(self, '_layout_stable_snapshot', None)
    is_stable_pass = (
        width_gap <= stabilization_tolerance
        and previous_snapshot is not None
        and abs(previous_snapshot[0] - root_w) <= self._scale_px(4)
        and abs(previous_snapshot[1] - total_w) <= self._scale_px(4)
    )
    forced_apply = False
    if not is_stable_pass:
        self._layout_stable_snapshot = current_snapshot
        if self._layout_retry_count < 12:
            self._layout_retry_count += 1
            self._schedule_window_distribution(delay_ms=90)
            return
        forced_apply = True

    self._layout_retry_count = 0
    self._layout_stable_snapshot = current_snapshot
    if not self._apply_pane_widths(paned):
        return

    self._schedule_comparison_layout_refresh(delay_ms=180 if forced_apply else 120)


_LAYOUT_METHODS = {
    '_get_expected_paned_width': _get_expected_paned_width,
    '_get_paned_sash_width': _get_paned_sash_width,
    '_get_paned_layout_width': _get_paned_layout_width,
    '_on_root_resize': _on_root_resize,
    '_schedule_window_distribution': _schedule_window_distribution,
    '_get_active_center_tab_key': _get_active_center_tab_key,
    '_select_center_tab': _select_center_tab,
    '_get_pane_layout_profile': _get_pane_layout_profile,
    '_calculate_pane_widths': _calculate_pane_widths,
    '_apply_pane_widths': _apply_pane_widths,
    '_optimize_window_distribution': _optimize_window_distribution,
}


def bind_layout_helpers(gui) -> None:
    """Attach extracted layout helpers to an existing GUI controller instance."""
    for name, func in _LAYOUT_METHODS.items():
        setattr(gui, name, MethodType(func, gui))
