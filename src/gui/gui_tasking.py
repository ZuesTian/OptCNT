"""Extracted task/executor helpers for CNTAnalyzerGUI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import MethodType
from typing import Optional
import sys


def _get_thread_pool_executor_class(self):
    gui_module = sys.modules.get(self.__class__.__module__)
    if gui_module is not None and hasattr(gui_module, 'ThreadPoolExecutor'):
        return getattr(gui_module, 'ThreadPoolExecutor')
    return ThreadPoolExecutor


def _create_preprocess_executor(self) -> ThreadPoolExecutor:
    """Create the dedicated executor used for background preprocess previews."""
    return self._get_thread_pool_executor_class()(max_workers=1, thread_name_prefix="cnt-preprocess")



def _reset_preprocess_executor(self) -> None:
    """Swap in a fresh preprocess executor so stale jobs cannot block newer previews."""
    executor = getattr(self, '_preprocess_executor', None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            self._log_helper_debug("Unable to reset the preprocess executor cleanly.")
    self._preprocess_executor = self._create_preprocess_executor()



def _discard_preprocess_state(self,
                              *,
                              include_completed: bool = False,
                              notify: bool = False,
                              image_reason: Optional[str] = None) -> bool:
    """Invalidate stale preprocess preview state after context changes."""
    preprocess_job = getattr(self, '_preprocess_job', None)
    if preprocess_job is not None and getattr(self, 'root', None) is not None:
        try:
            self.root.after_cancel(preprocess_job)
        except Exception:
            self._log_helper_debug("Unable to cancel the pending preprocess debounce callback cleanly.")
        self._preprocess_job = None

    future = getattr(self, '_preprocess_future', None)
    snapshot = getattr(self, '_preprocess_snapshot', None)
    if future is None and snapshot is None:
        return preprocess_job is not None

    is_running = future is not None and not future.done()
    if not include_completed and not is_running:
        return preprocess_job is not None

    if is_running:
        try:
            future.cancel()
        except Exception:
            self._log_helper_debug("Unable to cancel the in-flight preprocess future; it will finish in the background.")
        self._reset_preprocess_executor()

    self._preprocess_future = None
    self._preprocess_snapshot = None
    self._preprocess_token += 1

    if notify and image_reason and getattr(self, 'image_panel', None) is not None:
        self.image_panel.show_status(image_reason)
    return True



def _create_single_detect_executor(self) -> ThreadPoolExecutor:
    """Create the dedicated executor used for single-image CNT detection."""
    return self._get_thread_pool_executor_class()(max_workers=1, thread_name_prefix="cnt-single")



def _create_compare_executor(self) -> ThreadPoolExecutor:
    """Create the dedicated executor used for compare-analysis requests."""
    return self._get_thread_pool_executor_class()(max_workers=1, thread_name_prefix="cnt-compare")



def _reset_single_detect_executor(self) -> None:
    """Swap in a fresh executor so a stale running task cannot block new image analysis."""
    executor = getattr(self, '_single_detect_executor', None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            self._log_helper_debug("Unable to reset the single-image detection executor cleanly.")
    self._single_detect_executor = self._create_single_detect_executor()



def _reset_compare_executor(self) -> None:
    """Swap in a fresh executor so a stale compare task cannot block the next request."""
    executor = getattr(self, '_compare_executor', None)
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            self._log_helper_debug("Unable to reset the compare-analysis executor cleanly.")
    self._compare_executor = self._create_compare_executor()



def _discard_single_detection_state(self,
                                    *,
                                    reason: Optional[str] = None,
                                    image_reason: Optional[str] = None,
                                    include_completed: bool = False,
                                    notify: bool = True) -> bool:
    """Invalidate stale single-image detection state after context changes."""
    future = getattr(self, '_single_detect_future', None)
    snapshot = getattr(self, '_single_detect_snapshot', None)
    if future is None and snapshot is None:
        return False

    is_running = future is not None and not future.done()
    if not include_completed and not is_running:
        return False

    if is_running:
        try:
            future.cancel()
        except Exception:
            self._log_helper_debug("Unable to cancel the in-flight detection future; it will finish in the background.")
        self._reset_single_detect_executor()

    self._single_detect_future = None
    self._single_detect_snapshot = None
    self._single_detect_token += 1
    self._set_single_detection_busy_state(False)

    if notify and reason:
        if getattr(self, 'control_panel', None) is not None:
            self.control_panel.update_analysis_status(reason, color=self.MODERN_COLORS['warning'])
        if getattr(self, 'image_panel', None) is not None and image_reason:
            self.image_panel.show_status(image_reason)
    return True



def _abandon_single_detection_if_running(self,
                                         reason: str = "检测参数已更新，当前后台结果将忽略，可重新开始分析。") -> bool:
    """Drop the current single-image detection result when settings change mid-run."""
    return self._discard_single_detection_state(
        reason=reason,
        image_reason="检测参数已更新，可重新开始CNT检测",
        include_completed=False,
        notify=True,
    )



def _set_single_detection_busy_state(self, busy: bool) -> None:
    """Toggle the single-detection button and related entry points while keeping the UI responsive."""
    if getattr(self, 'control_panel', None) is not None:
        self._set_ttk_widget_enabled(getattr(self.control_panel, 'detect_button', None), not busy)
    self._set_ttk_widget_enabled(self.compare_analysis_button, not busy)
    if not busy:
        self._refresh_interaction_state()



def _set_compare_analysis_busy_state(self, busy: bool) -> None:
    """Toggle compare-analysis entry points while a background compare task is running."""
    if getattr(self, 'control_panel', None) is not None:
        self._set_ttk_widget_enabled(getattr(self.control_panel, 'detect_button', None), not busy)
    self._set_ttk_widget_enabled(self.compare_analysis_button, not busy)
    if not busy:
        self._refresh_interaction_state()



def _discard_compare_analysis_state(self,
                                    *,
                                    include_completed: bool = False,
                                    notify: bool = False,
                                    reason: Optional[str] = None) -> bool:
    """Invalidate stale compare-analysis state and optionally cancel an in-flight request."""
    future = getattr(self, '_compare_future', None)
    snapshot = getattr(self, '_compare_snapshot', None)
    if future is None and snapshot is None:
        return False

    is_running = future is not None and not future.done()
    if not include_completed and not is_running:
        return False

    if is_running:
        try:
            future.cancel()
        except Exception:
            self._log_helper_debug("Unable to cancel the in-flight compare future; it will finish in the background.")
        self._reset_compare_executor()

    self._compare_future = None
    self._compare_snapshot = None
    self._compare_token += 1
    if getattr(self, 'comparison_panel', None) is not None:
        self.comparison_panel.hide_progress()
    self._set_compare_analysis_busy_state(False)

    if notify and reason and getattr(self, 'image_panel', None) is not None:
        self.image_panel.show_status(reason)
    return True



def _log_helper_debug(self, message: str) -> None:
    self._task_logger.debug(message)


_TASK_METHODS = {
    '_get_thread_pool_executor_class': _get_thread_pool_executor_class,
    '_create_preprocess_executor': _create_preprocess_executor,
    '_reset_preprocess_executor': _reset_preprocess_executor,
    '_discard_preprocess_state': _discard_preprocess_state,
    '_create_single_detect_executor': _create_single_detect_executor,
    '_create_compare_executor': _create_compare_executor,
    '_reset_single_detect_executor': _reset_single_detect_executor,
    '_reset_compare_executor': _reset_compare_executor,
    '_discard_single_detection_state': _discard_single_detection_state,
    '_abandon_single_detection_if_running': _abandon_single_detection_if_running,
    '_set_single_detection_busy_state': _set_single_detection_busy_state,
    '_set_compare_analysis_busy_state': _set_compare_analysis_busy_state,
    '_discard_compare_analysis_state': _discard_compare_analysis_state,
    '_log_helper_debug': _log_helper_debug,
}


def bind_task_helpers(gui) -> None:
    """Attach extracted task/executor helpers to an existing GUI controller instance."""
    gui._task_logger = __import__('logging').getLogger(__name__)
    for name, func in _TASK_METHODS.items():
        setattr(gui, name, MethodType(func, gui))
