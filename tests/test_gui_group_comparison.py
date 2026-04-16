import matplotlib
import pytest
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

matplotlib.use("Agg")

import src.gui.gui as gui_module
from src.gui.gui import CNTAnalyzerGUI


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyWidget:
    def __init__(self, width: int):
        self._width = width

    def update_idletasks(self):
        return None

    def winfo_width(self):
        return self._width


class _DummyNotebook:
    def __init__(self, selected="image_tab"):
        self.selected = selected
        self.select_calls = []
        self.updated = 0

    def select(self, tab=None):
        if tab is None:
            return self.selected
        self.selected = str(tab)
        self.select_calls.append(str(tab))
        return self.selected

    def update_idletasks(self):
        self.updated += 1


class _DummyStateWidget:
    def __init__(self):
        self.disabled = False

    def state(self, flags):
        if 'disabled' in flags:
            self.disabled = True
        if '!disabled' in flags:
            self.disabled = False


class _RunningFuture:
    def __init__(self):
        self.cancel_calls = 0

    def done(self):
        return False

    def cancel(self):
        self.cancel_calls += 1
        return False


class _DoneFuture:
    def done(self):
        return True


class _ResultFuture(_DoneFuture):
    def __init__(self, payload):
        self.payload = payload

    def result(self):
        return self.payload


class _DummyExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, wait=False, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


class _DummySubmitExecutor(_DummyExecutor):
    def __init__(self, future):
        super().__init__()
        self.future = future
        self.submissions = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        return self.future


class _DummyRoot:
    def __init__(self):
        self.after_calls = []
        self.after_cancel_calls = []

    def after(self, delay, callback, *args):
        token = f"after-{len(self.after_calls) + 1}"
        self.after_calls.append((delay, callback, args, token))
        return token

    def after_cancel(self, token):
        self.after_cancel_calls.append(token)


class _DummyControlPanelState:
    def __init__(self):
        self.calls = []

    def set_interaction_state(self, **kwargs):
        self.calls.append(kwargs)


class _DummyImagePanelState:
    def __init__(self):
        self.enabled_calls = []
        self.cleared = 0
        self.hidden = 0

    def set_image_actions_enabled(self, has_image: bool):
        self.enabled_calls.append(has_image)

    def clear_canvas(self):
        self.cleared += 1

    def hide_status(self):
        self.hidden += 1


class _DummyAnalysisPanel:
    def __init__(self):
        self.cleared_keys = []
        self.refresh_count = 0
        self.scroll_count = 0
        self.frames = {}

    def clear_chart_content(self, key: str):
        self.cleared_keys.append(key)

    def refresh_layout(self):
        self.refresh_count += 1

    def scroll_to_top(self):
        self.scroll_count += 1

    def get_chart_frame(self, key: str):
        return self.frames.get(key)


class _DummyComparisonPanel(_DummyAnalysisPanel):
    def __init__(self):
        super().__init__()
        self.progress_visible = False
        self.progress_updates = []
        self.section_heights = {}
        self.text_content = {}

    def show_progress(self):
        self.progress_visible = True

    def hide_progress(self):
        self.progress_visible = False

    def update_progress(self, current: int, total: int, message: str = ""):
        self.progress_updates.append((current, total, message))

    def set_section_height(self, key: str, value: int):
        self.section_heights[key] = value

    def set_text_content(self, key: str, value: str):
        self.text_content[key] = value


class _DummyTextWidget:
    def __init__(self):
        self.parts = []

    def insert(self, index, text, tag=None):
        self.parts.append(str(text))

    def delete(self, start, end):
        self.parts.clear()

    def getvalue(self):
        return "".join(self.parts)


class _DummyResultPanelState:
    def __init__(self):
        self.stats_text = _DummyTextWidget()
        self.rows = []

    def clear_stats(self):
        self.stats_text.delete(None, None)

    def clear_tree(self):
        self.rows.clear()

    def add_measurement(self, values):
        self.rows.append(values)


def _make_gui_stub() -> CNTAnalyzerGUI:
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui._preprocess_preview_fast = False
    gui._preprocess_result_exact = False
    gui._preprocess_future = None
    gui._preprocess_snapshot = None
    gui._preprocess_token = 0
    gui._preprocess_job = None
    gui._single_detect_future = None
    gui._single_detect_snapshot = None
    gui._single_detect_token = 0
    gui._compare_future = None
    gui._compare_snapshot = None
    gui._compare_token = 0
    gui._comparison_layout_job = None
    gui.blur_kernel_var = _DummyVar(9)
    gui.adaptive_block_var = _DummyVar(11)
    gui.adaptive_c_var = _DummyVar(3)
    gui.bridge_strength_var = _DummyVar(2)
    gui.min_length_um_var = _DummyVar(4.0)
    gui.max_length_um_var = _DummyVar(200.0)
    gui.min_slenderness_var = _DummyVar(3.0)
    gui.detect_profile_var = _DummyVar("标准（推荐）")
    gui.split_mode_var = _DummyVar("不拆分")
    gui.merge_distance_px_var = _DummyVar(6)
    gui.scale_um_var = _DummyVar(10.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    return gui


def _make_result(name: str,
                 count: float,
                 uniformity_overall: float,
                 uniformity_nn: float,
                 uniformity_grid: float,
                 uniformity_moran: float,
                 nn_cv: float,
                 grid_cv: float,
                 morans_i: float) -> dict:
    aggregation_scores = {
        "nearest_neighbor": 100.0 - uniformity_nn,
        "grid_density": 100.0 - uniformity_grid,
        "moran": 100.0 - uniformity_moran,
        "overall": 100.0 - uniformity_overall,
    }
    dispersed_count = max(0.0, round(count * 0.7))
    agglomerated_count = max(0.0, count - dispersed_count)
    framework = {
        "uniformity": {"score": float(uniformity_grid)},
        "thick_bundle": {"score": float(max(20.0, 88.0 - count * 0.6))},
        "long_cnt": {"score": float(min(100.0, 35.0 + count))},
        "agglomeration": {"score": float(uniformity_overall)},
    }
    framework["hybrid_score"] = (
        framework["uniformity"]["score"] * 0.35 +
        framework["thick_bundle"]["score"] * 0.20 +
        framework["long_cnt"]["score"] * 0.25 +
        framework["agglomeration"]["score"] * 0.20
    )
    return {
        "name": name,
        "path": f"/tmp/{name}.png",
        "stats": {
            "count": count,
            "length_mean": 12.5,
            "spatial_distribution": {
                "nearest_neighbor_cv": nn_cv,
                "nearest_neighbor_index": 0.95,
                "grid_density_cv": grid_cv,
                "morans_i": morans_i,
                "grid_entropy": 0.9,
                "occupancy_ratio": 0.55,
                "density_grid": [[1.0, 0.0], [0.0, 1.0]],
                "uniformity_scores": {
                    "nearest_neighbor": uniformity_nn,
                    "grid_density": uniformity_grid,
                    "moran": uniformity_moran,
                    "overall": uniformity_overall,
                },
                "aggregation_scores": aggregation_scores,
            },
        },
        "dispersed_stats": {
            "dispersed_count": dispersed_count,
            "agglomerated_count": agglomerated_count,
            "dispersed_ratio": dispersed_count / count if count else 0.0,
            "dispersed_length_stats": {
                "length_mean": 11.0,
                "length_std": 1.5,
            },
        },
        "evaluation_framework": framework,
    }


def _make_group_results(prefix: str, specs: list[tuple[float, float, float, float, float, float]]) -> list[dict]:
    results = []
    for index, (count, uniformity_overall, uniformity_nn, uniformity_grid, uniformity_moran, morans_i) in enumerate(specs):
        nn_cv = max(0.05, 1.0 - uniformity_nn / 100.0)
        grid_cv = max(0.05, 1.0 - uniformity_grid / 100.0)
        results.append(
            _make_result(
                f"{prefix}_{index + 1}",
                count=count,
                uniformity_overall=uniformity_overall,
                uniformity_nn=uniformity_nn,
                uniformity_grid=uniformity_grid,
                uniformity_moran=uniformity_moran,
                nn_cv=nn_cv,
                grid_cv=grid_cv,
                morans_i=morans_i,
            )
        )
    return results


def _make_batch_analysis_result(image_path: str, count: float) -> dict:
    name = image_path.replace("\\", "/").split("/")[-1]
    return {
        "path": image_path,
        "name": name,
        "stats": {
            "count": count,
            "spatial_distribution": {},
        },
        "evaluation_framework": {
            "uniformity": {"score": 60.0},
            "thick_bundle": {"score": 70.0},
            "long_cnt": {"score": 65.0},
            "agglomeration": {"score": 62.0},
            "hybrid_score": 63.9,
        },
        "scale_info": None,
        "scale_status": {},
        "visualization": None,
    }


def test_summarize_group_results_includes_aggregation_statistics():
    gui = _make_gui_stub()
    base_results = _make_group_results(
        "base",
        [
            (32.0, 55.0, 54.0, 56.0, 55.0, 0.32),
            (30.0, 60.0, 58.0, 61.0, 61.0, 0.28),
        ],
    )

    summary = gui._summarize_group_results("base组", base_results)

    assert summary["file_details"][0]["aggregation_score"] == pytest.approx(45.0)
    assert summary["file_details"][0]["shadow_aggregation_score"] == pytest.approx(45.0)
    assert summary["file_details"][0]["dispersed_count"] > 0
    assert summary["dispersed_count_stats"]["mean"] > 0
    assert summary["agglomerated_count_stats"]["mean"] >= 0
    assert summary["dispersed_ratio_stats"]["mean"] > 0
    assert summary["spatial_stats"]["aggregation_score"]["mean"] == pytest.approx(42.5)
    assert summary["spatial_stats"]["shadow_aggregation_score"]["mean"] == pytest.approx(42.5)
    assert summary["spatial_stats"]["aggregation_moran_score"]["max"] == pytest.approx(45.0)


def test_get_preprocess_signature_changes_when_scale_exclusion_rect_changes():
    gui = _make_gui_stub()
    gui.current_roi = None
    gui.analyzer = SimpleNamespace(scale_exclusion_rect=None, rois=[])

    signature_without_exclusion = gui._get_preprocess_signature()

    gui.analyzer.scale_exclusion_rect = (180, 192, 302, 226)
    signature_with_exclusion = gui._get_preprocess_signature()

    assert signature_without_exclusion != signature_with_exclusion


def test_apply_preprocessing_skips_skeleton_for_live_binary_preview():
    gui = _make_gui_stub()
    gui.current_roi = None
    gui.display_var = _DummyVar("binary")
    gui.live_preview_var = _DummyVar(True)
    gui._preprocess_job = object()
    gui._last_preprocess_signature = None
    gui._update_display = lambda: None
    gui.bridge_strength_var = _DummyVar(9)

    captured = {}

    def fake_preprocess(**kwargs):
        captured.update(kwargs)

    gui.analyzer = SimpleNamespace(
        image=np.zeros((24, 24, 3), dtype=np.uint8),
        binary_image=None,
        scale_exclusion_rect=None,
        rois=[],
        preprocess=fake_preprocess,
    )

    gui._apply_preprocessing(force=False)

    assert captured["generate_skeleton"] is False
    assert captured["bridge_strength"] == 5
    assert gui._preprocess_preview_fast is True
    assert gui._preprocess_result_exact is False


def test_apply_preprocessing_submits_background_preview_when_root_available():
    gui = _make_gui_stub()
    gui.root = _DummyRoot()
    gui.current_roi = None
    gui.display_var = _DummyVar("binary")
    gui.live_preview_var = _DummyVar(True)
    gui._last_preprocess_signature = None
    gui.image_panel = SimpleNamespace(show_status=lambda text: None)
    gui._update_display = lambda: None
    gui.bridge_strength_var = _DummyVar(8)
    gui._preprocess_executor = _DummySubmitExecutor(_RunningFuture())

    analyzer = gui_module.CNTAnalyzer()
    analyzer.image = np.zeros((24, 24, 3), dtype=np.uint8)
    analyzer.original_image = analyzer.image.copy()
    analyzer.analysis_gray_image = np.zeros((24, 24), dtype=np.uint8)
    gui.analyzer = analyzer

    gui._apply_preprocessing(force=False)

    assert gui._preprocess_future is gui._preprocess_executor.future
    assert gui._preprocess_snapshot["fast_preview"] is True
    assert gui._preprocess_snapshot["result_exact"] is False
    assert len(gui._preprocess_executor.submissions) == 1
    _, args, _ = gui._preprocess_executor.submissions[0]
    assert args[1]["bridge_strength"] == 5
    assert args[1]["generate_skeleton"] is False
    assert len(gui.root.after_calls) == 1


def test_needs_preprocessing_requires_exact_result_for_skeleton_preview():
    gui = _make_gui_stub()
    gui.current_roi = None
    gui.display_var = _DummyVar("skeleton_preview")
    gui.live_preview_var = _DummyVar(True)
    gui.analyzer = SimpleNamespace(
        binary_image=np.ones((8, 8), dtype=np.uint8),
        skeleton_image=np.ones((8, 8), dtype=np.uint8),
        scale_exclusion_rect=None,
        rois=[],
    )
    gui._last_preprocess_signature = gui._get_preprocess_signature()
    gui._preprocess_result_exact = False

    assert gui._needs_preprocessing() is True


def test_load_image_common_discards_stale_preprocess_state(monkeypatch):
    gui = _make_gui_stub()
    gui.root = _DummyRoot()
    gui.MODERN_COLORS = {"warning": "#f59e0b"}
    old_executor = _DummyExecutor()
    replacement_executors = []
    reset_calls = []
    statuses = []

    class _ReplacementExecutor(_DummyExecutor):
        pass

    def _fake_thread_pool(*args, **kwargs):
        executor = _ReplacementExecutor()
        replacement_executors.append((executor, args, kwargs))
        return executor

    monkeypatch.setattr(gui_module, "ThreadPoolExecutor", _fake_thread_pool)

    gui._preprocess_executor = old_executor
    gui._preprocess_future = _RunningFuture()
    gui._preprocess_snapshot = {"image_id": 1}
    gui._preprocess_token = 4
    gui._preprocess_job = "after-1"
    gui._discard_single_detection_state = lambda *args, **kwargs: None
    gui._reset_display = lambda: reset_calls.append(True)
    gui._refresh_scale_status_ui = lambda: None
    gui._auto_suggest_params = lambda: None
    gui._refresh_analysis_status_ui = lambda: None
    gui.live_preview_var = _DummyVar(False)
    gui.scale_um_var = _DummyVar(0.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    gui.image_panel = SimpleNamespace(show_status=lambda text: statuses.append(text))
    gui.analyzer = SimpleNamespace(
        apply_detected_scale=lambda default_micrometers: {"applied": False, "scale_info": None},
    )

    gui._load_image_common()

    assert gui._preprocess_future is None
    assert gui._preprocess_snapshot is None
    assert gui._preprocess_token == 5
    assert old_executor.shutdown_calls == [(False, True)]
    assert len(replacement_executors) == 1
    assert isinstance(gui._preprocess_executor, _ReplacementExecutor)
    assert gui.root.after_cancel_calls == ["after-1"]
    assert reset_calls == [True]
    assert len(statuses) == 1


def test_display_mode_change_forces_full_preprocess_when_skeleton_preview_needs_skeleton():
    gui = _make_gui_stub()
    gui.current_roi = None
    gui.display_var = _DummyVar("skeleton_preview")
    gui.live_preview_var = _DummyVar(True)

    calls = []
    gui._apply_preprocessing = lambda force=False: calls.append(force)
    gui._update_display = lambda: calls.append("display")
    gui.analyzer = SimpleNamespace(
        image=np.zeros((24, 24, 3), dtype=np.uint8),
        binary_image=np.ones((24, 24), dtype=np.uint8),
        skeleton_image=None,
        scale_exclusion_rect=None,
        rois=[],
    )
    gui._last_preprocess_signature = gui._get_preprocess_signature()

    gui._on_display_mode_change()

    assert calls == [True]


def test_init_chart_removes_cached_colorbar_before_redraw():
    gui = _make_gui_stub()
    gui.analysis_panel = _DummyAnalysisPanel()
    gui.analysis_panel.frames["heatmap"] = object()

    class _DummyColorbar:
        def __init__(self):
            self.remove_calls = 0

        def remove(self):
            self.remove_calls += 1

    class _DummyAxes:
        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

    colorbar = _DummyColorbar()
    axes = _DummyAxes()
    gui._charts = {
        "heatmap": {
            "fig": object(),
            "ax": axes,
            "canvas": object(),
            "colorbar": colorbar,
            "draw_count": 1,
        }
    }

    chart = gui._init_chart("heatmap")

    assert chart["colorbar"] is None
    assert colorbar.remove_calls == 1
    assert axes.clear_calls == 1


def test_handle_single_detection_result_defers_advanced_analysis_refresh(monkeypatch):
    gui = _make_gui_stub()
    gui.root = _DummyRoot()
    gui.current_roi = None
    gui._single_detect_token = 3
    gui._single_detect_future = None
    gui._single_detect_snapshot = None
    gui._last_preprocess_signature = ("sig",)
    gui._roi_signature = lambda roi: None
    gui.MODERN_COLORS = {"info": "#0ea5e9"}
    gui.analyzer = SimpleNamespace(image=np.zeros((12, 12, 3), dtype=np.uint8), measurements=[])

    status_calls = []
    sync_calls = []
    busy_calls = []
    refresh_calls = []
    image_statuses = []

    gui.control_panel = SimpleNamespace(update_analysis_status=lambda *args, **kwargs: status_calls.append((args, kwargs)))
    gui.image_panel = SimpleNamespace(show_status=lambda text: image_statuses.append(text))
    gui._sync_views = lambda **kwargs: sync_calls.append(kwargs)
    gui._set_single_detection_busy_state = lambda busy: busy_calls.append(busy)
    gui._refresh_analysis_status_ui = lambda: refresh_calls.append(True)
    gui._update_advanced_analysis = lambda: None

    class _DoneFutureWithResult:
        def result(self):
            return [SimpleNamespace(id=0)]

    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    future = _DoneFutureWithResult()
    snapshot = {"image_id": id(gui.analyzer.image), "preprocess_signature": ("sig",), "roi_signature": None}
    gui._handle_single_detection_result(3, snapshot, future)

    assert sync_calls == [{"refresh_analysis": False}]
    assert busy_calls == [False]
    assert refresh_calls == [True]
    assert gui.root.after_calls[0][0] == 10
    assert gui.root.after_calls[0][1] == gui._update_advanced_analysis
    assert len(gui.analyzer.measurements) == 1
    assert image_statuses


def test_apply_scale_forces_preprocess_refresh_in_preprocess_mode(monkeypatch):
    gui = _make_gui_stub()
    gui.scale_pixels_var = _DummyVar(100.0)
    gui.scale_um_var = _DummyVar(10.0)
    gui.display_var = _DummyVar("binary")
    gui._pending_scale_selection = None
    gui._last_preprocess_signature = ("stale",)

    preprocess_calls = []
    sync_calls = []

    def fake_record_manual_scale(*args, **kwargs):
        gui.analyzer.scale_um_per_pixel = 0.2

    gui.analyzer = SimpleNamespace(
        scale_um_per_pixel=0.1,
        measurements=[],
        rois=[],
        record_manual_scale=fake_record_manual_scale,
    )
    gui._refresh_scale_status_ui = lambda: None
    gui._refresh_analysis_status_ui = lambda: None
    gui._apply_preprocessing = lambda force=False: preprocess_calls.append(force)
    gui._sync_views = lambda: sync_calls.append(True)

    monkeypatch.setattr(gui_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(gui_module.messagebox, "showerror", lambda *args, **kwargs: None)

    gui._apply_scale()

    assert gui._last_preprocess_signature is None
    assert preprocess_calls == [True]
    assert sync_calls == [True]


def test_auto_suggest_params_returns_none_for_expected_errors():
    gui = _make_gui_stub()
    gui.current_roi = None
    gui._last_auto_suggest_result = {"blur_kernel": 7}
    gui.analyzer = SimpleNamespace(
        suggest_preprocess_params=lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid image")),
    )
    gui.control_panel = SimpleNamespace(
        update_blur_label=lambda value: None,
        update_block_label=lambda value: None,
        update_c_label=lambda value: None,
    )
    gui._refresh_analysis_status_ui = lambda: None

    result = gui._auto_suggest_params()

    assert result is None
    assert gui._last_auto_suggest_result is None


def test_format_compact_params_uses_short_layout():
    gui = _make_gui_stub()

    result = gui._format_compact_params()

    assert result.startswith("识别参数: ")
    assert "模糊9/块11/C3" in result
    assert "长度≥4.0μm" in result
    assert "长宽比≥3.0" in result
    assert "标准（推荐）" in result


def test_format_group_comparison_summary_uses_strong_clustering_conclusion():
    gui = _make_gui_stub()
    base_group = gui._summarize_group_results(
        "base组",
        _make_group_results(
            "base",
            [
                (30.0, 34.0, 33.0, 36.0, 33.0, 0.62),
                (31.0, 36.0, 35.0, 37.0, 36.0, 0.58),
                (29.0, 35.0, 34.0, 36.0, 35.0, 0.60),
                (30.0, 37.0, 36.0, 38.0, 37.0, 0.57),
                (32.0, 35.0, 34.0, 35.0, 36.0, 0.61),
            ],
        ),
    )
    exp_group = gui._summarize_group_results(
        "实验组",
        _make_group_results(
            "exp",
            [
                (33.0, 58.0, 57.0, 59.0, 58.0, 0.22),
                (34.0, 60.0, 59.0, 61.0, 60.0, 0.20),
                (32.0, 57.0, 56.0, 58.0, 57.0, 0.24),
                (35.0, 61.0, 60.0, 62.0, 61.0, 0.18),
                (34.0, 59.0, 58.0, 60.0, 59.0, 0.21),
            ],
        ),
    )

    summary = gui._format_group_comparison_summary(base_group, exp_group)

    assert "CNT数量" in summary
    assert "分散CNT数量" in summary
    assert "总CNT数量" in summary
    assert "团聚区域CNT数量" in summary
    assert "分散比例" in summary
    assert "分散CNT长度统计" in summary
    assert "空间分布均匀性" in summary
    assert "base: 均值" in summary
    assert "核心指标" in summary
    assert "结论: 实验组更优" in summary
    assert "阴影团聚(0-100，越低越好)" in summary
    assert "均匀度(0-100，越高越好)" in summary
    assert "base组逐图明细" in summary
    assert "总CNT=" in summary
    assert "Mann-Whitney" not in summary
    return

    assert "CNT???" in summary
    assert "???CNT???" in summary
    assert "??NT???" in summary
    assert "??????CNT???" in summary
    assert "??????" in summary
    assert "???CNT??????" in summary
    assert "???????????" in summary
    assert "CNT数量" in summary
    assert "base: 均值" in summary
    assert "核心指标" in summary
    assert "结论: 实验组更优" in summary
    assert "阴影团聚(0-100，越低越好)" in summary
    assert "均匀度(0-100，越高越好)" in summary
    assert "base组逐图明细" in summary
    assert "CNT=" in summary
    assert "Mann-Whitney" not in summary


def test_format_group_comparison_summary_uses_cautious_conclusion_when_gap_is_small():
    gui = _make_gui_stub()
    base_group = gui._summarize_group_results(
        "base组",
        _make_group_results(
            "base",
            [
                (30.0, 46.0, 45.0, 47.0, 46.0, 0.38),
                (31.0, 47.0, 46.0, 48.0, 47.0, 0.37),
                (29.0, 45.0, 44.0, 46.0, 45.0, 0.39),
            ],
        ),
    )
    exp_group = gui._summarize_group_results(
        "实验组",
        _make_group_results(
            "exp",
            [
                (32.0, 49.0, 48.0, 50.0, 49.0, 0.34),
                (31.0, 50.0, 49.0, 51.0, 50.0, 0.33),
                (30.0, 48.0, 47.0, 49.0, 48.0, 0.35),
            ],
        ),
    )

    summary = gui._format_group_comparison_summary(base_group, exp_group)

    assert "结论: 实验组趋势更优" in summary
    assert "统计证据仍偏弱" in summary
    assert "结论: 实验组更优" not in summary


def test_legacy_group_comparison_summary_delegates_to_current_formatter():
    gui = _make_gui_stub()
    base_group = gui._summarize_group_results(
        "base组",
        _make_group_results(
            "base",
            [
                (30.0, 46.0, 45.0, 47.0, 46.0, 0.38),
                (31.0, 47.0, 46.0, 48.0, 47.0, 0.37),
                (29.0, 45.0, 44.0, 46.0, 45.0, 0.39),
            ],
        ),
    )
    exp_group = gui._summarize_group_results(
        "实验组",
        _make_group_results(
            "exp",
            [
                (32.0, 49.0, 48.0, 50.0, 49.0, 0.34),
                (31.0, 50.0, 49.0, 51.0, 50.0, 0.33),
                (30.0, 48.0, 47.0, 49.0, 48.0, 0.35),
            ],
        ),
    )

    legacy = gui._legacy_format_group_comparison_summary(
        base_group,
        exp_group,
        note="note",
        failures=["bad.png"],
    )
    current = gui._format_group_comparison_summary(
        base_group,
        exp_group,
        note="note",
        failures=["bad.png"],
    )

    assert legacy == current


def test_format_comparison_summary_uses_compact_text():
    gui = _make_gui_stub()
    left_result = _make_result("left", 48.0, 60.0, 61.0, 59.0, 60.0, 0.39, 0.41, 0.18)
    right_result = _make_result("right", 41.0, 47.0, 46.0, 48.0, 47.0, 0.51, 0.53, 0.31)

    summary = gui._format_comparison_summary(left_result, right_result, "样品A", "样品B")

    assert summary.startswith("对比预处理: 模糊9/块11/C3")
    assert "CNT数量" in summary
    assert "样品A: left | CNT=48" in summary
    assert "样品B: right | CNT=41" in summary
    assert "核心指标" in summary
    assert "阴影团聚(0-100，越低越好)" in summary
    assert "均匀度(0-100，越高越好)" in summary
    assert "结论: 样品A更优" in summary
    assert "Mann-Whitney" not in summary


def test_format_comparison_summary_prefers_stored_compare_context_over_live_widgets():
    gui = _make_gui_stub()
    gui.min_length_um_var = _DummyVar(40.0)
    gui.min_slenderness_var = _DummyVar(9.0)
    gui.bridge_strength_var = _DummyVar(1)
    gui.detect_profile_var = _DummyVar("严格（少误检）")
    gui.split_mode_var = _DummyVar("不拆分")
    gui.merge_distance_px_var = _DummyVar(0)

    compare_context = {
        "preprocess_settings": {
            "blur_kernel": 15,
            "adaptive_block": 21,
            "adaptive_c": 5,
            "bridge_strength": 6,
            "threshold_invert": True,
        },
        "detect_settings": {
            "min_length_um": 12.0,
            "max_length_um": 200.0,
            "min_slenderness": 4.5,
            "detection_profile": "recall",
            "merge_distance_px": 8.0,
            "split_mode": "conservative",
        },
        "analysis_roi": {
            "mode": "center_fraction",
            "fraction": 0.75,
            "label": "中部75%",
        },
    }

    left_result = {
        **_make_batch_analysis_result("left.png", 48.0),
        "analysis_context": compare_context,
        "stats": {"count": 48, "spatial_distribution": {}},
        "dispersed_stats": {"dispersed_count": 30, "agglomerated_count": 18, "dispersed_ratio": 30 / 48},
    }
    right_result = {
        **_make_batch_analysis_result("right.png", 41.0),
        "analysis_context": compare_context,
        "stats": {"count": 41, "spatial_distribution": {}},
        "dispersed_stats": {"dispersed_count": 20, "agglomerated_count": 21, "dispersed_ratio": 20 / 41},
    }

    summary = gui._format_comparison_summary(left_result, right_result, "样品A", "样品B")

    assert summary.startswith("对比预处理: 模糊15/块21/C5")
    assert "长度≥12.0μm" in summary
    assert "长宽比≥4.5" in summary
    assert "敏感（少漏检）" in summary
    assert "桥接6" in summary
    assert "标准（推荐）" not in summary
    assert "长度≥40.0μm" not in summary


def test_plot_group_aggregation_risk_chart_uses_fixed_scale_and_group_colors():
    gui = _make_gui_stub()
    base_group = gui._summarize_group_results(
        "base组",
        _make_group_results(
            "base",
            [
                (30.0, 35.0, 34.0, 36.0, 35.0, 0.60),
                (31.0, 36.0, 35.0, 37.0, 36.0, 0.59),
                (29.0, 34.0, 33.0, 35.0, 34.0, 0.61),
            ],
        ),
    )
    exp_group = gui._summarize_group_results(
        "实验组",
        _make_group_results(
            "exp",
            [
                (33.0, 58.0, 57.0, 59.0, 58.0, 0.22),
                (34.0, 60.0, 59.0, 61.0, 60.0, 0.20),
                (32.0, 57.0, 56.0, 58.0, 57.0, 0.24),
            ],
        ),
    )
    tests = gui._compute_group_comparison_tests(base_group, exp_group)
    figure = Figure()
    ax = figure.add_subplot(111)

    gui._plot_group_aggregation_risk_chart(ax, base_group, exp_group, tests)

    assert ax.get_title() == "阴影团聚 / 均匀度组间对比"
    assert ax.get_ylim() == (0.0, 100.0)
    assert ax.get_legend() is not None
    assert ax.patches[0].get_facecolor() == pytest.approx(to_rgba(CNTAnalyzerGUI.MODERN_COLORS["accent_rose"], alpha=0.95))
    assert ax.patches[2].get_facecolor() == pytest.approx(to_rgba(CNTAnalyzerGUI.MODERN_COLORS["accent_teal"], alpha=0.82))


def test_render_group_count_distribution_views_switches_to_single_point_mode():
    gui = _make_gui_stub()
    base_group = gui._summarize_group_results(
        "base组",
        _make_group_results("base", [(632.0, 62.0, 61.0, 63.0, 62.0, 0.06)]),
    )
    exp_group = gui._summarize_group_results(
        "实验组",
        _make_group_results("exp", [(811.0, 64.0, 63.0, 65.0, 64.0, 0.11)]),
    )
    count_tests = gui._compute_group_comparison_tests(base_group, exp_group)["count"]
    figure = Figure()
    ax_box = figure.add_subplot(121)
    ax_detail = figure.add_subplot(122)

    gui._render_group_count_distribution_views(ax_box, ax_detail, base_group, exp_group, count_tests)

    assert "样本量较少" in ax_box.get_title()
    assert ax_detail.get_title() == "逐图CNT概览"
    assert len(ax_detail.patches) == 2


def test_fit_comparison_figure_to_frame_shrinks_figure_to_available_width():
    gui = _make_gui_stub()
    gui.center_notebook = _DummyWidget(760)
    figure = Figure(figsize=(14.0, 9.0), dpi=100)

    gui._fit_comparison_figure_to_frame(figure, _DummyWidget(720))

    assert figure.get_figwidth() * figure.dpi <= 736


def test_fit_comparison_figure_to_frame_can_expand_after_narrow_first_render():
    gui = _make_gui_stub()
    gui.center_notebook = _DummyWidget(1160)
    figure = Figure(figsize=(5.1, 7.5), dpi=90)

    gui._fit_comparison_figure_to_frame(figure, _DummyWidget(1113), allow_expand=True)

    width_px = figure.get_figwidth() * figure.dpi
    assert 680 <= width_px <= 690


def test_calculate_pane_widths_prefers_center_column():
    gui = _make_gui_stub()

    left_w, center_w, right_w = gui._calculate_pane_widths(2156)

    assert left_w == 431
    assert center_w == 1294
    assert right_w == 431
    assert center_w > left_w
    assert center_w > right_w


def test_calculate_pane_widths_gives_comparison_tab_more_center_space():
    gui = _make_gui_stub()

    image_widths = gui._calculate_pane_widths(2156, 'image')
    comparison_widths = gui._calculate_pane_widths(2156, 'comparison')

    assert comparison_widths[1] > image_widths[1]
    assert comparison_widths[0] < image_widths[0]
    assert comparison_widths[2] < image_widths[2]


def test_build_comparison_layout_uses_stable_presets():
    gui = _make_gui_stub()

    pair_stacked = gui._build_comparison_layout(stacked=True, variant='pair')
    group_side_by_side = gui._build_comparison_layout(stacked=False, variant='group')

    assert pair_stacked['figsize'] == (13.4, 11.5)
    assert pair_stacked['height_ratios'] == [0.75, 1.6, 1.6]
    assert group_side_by_side['figsize'] == (14.5, 11.8)
    assert group_side_by_side['height_ratios'] == [0.90, 0.80, 1.5]


def test_show_comparison_window_renders_only_dispersion_charts():
    gui = _make_gui_stub()
    captured = {}

    left_result = _make_result("left", 28.0, 61.0, 60.0, 62.0, 61.0, 0.4, 0.38, 0.22)
    right_result = _make_result("right", 34.0, 55.0, 53.0, 56.0, 55.0, 0.47, 0.44, 0.35)
    left_result["visualization"] = np.zeros((120, 180, 3), dtype=np.uint8)
    right_result["visualization"] = np.zeros((120, 180, 3), dtype=np.uint8)

    gui._render_comparison_figure = lambda summary_text, figure: captured.update({"summary": summary_text, "figure": figure})

    gui._show_comparison_window(left_result, right_result, "图像A", "图像B")

    titles = [ax.get_title() for ax in captured["figure"].axes]
    assert titles[:3] == ["分散CNT数量对比", "分散比例对比", "混合评分对比"]
    assert len(captured["figure"].axes) == 5


def test_show_group_comparison_window_reuses_dispersion_dashboard():
    gui = _make_gui_stub()
    captured = {}

    base_results = _make_group_results(
        "base",
        [
            (30.0, 58.0, 57.0, 59.0, 58.0, 0.26),
            (32.0, 56.0, 55.0, 57.0, 56.0, 0.24),
        ],
    )
    exp_results = _make_group_results(
        "exp",
        [
            (36.0, 62.0, 61.0, 63.0, 62.0, 0.18),
            (35.0, 64.0, 63.0, 65.0, 64.0, 0.16),
        ],
    )
    for index, result in enumerate(base_results):
        result["path"] = rf"C:\tmp\base_{index}.png"
        result["visualization"] = np.zeros((120, 180, 3), dtype=np.uint8)
    for index, result in enumerate(exp_results):
        result["path"] = rf"C:\tmp\exp_{index}.png"
        result["visualization"] = np.zeros((120, 180, 3), dtype=np.uint8)

    base_group = gui._summarize_group_results("base组", base_results)
    exp_group = gui._summarize_group_results("实验组", exp_results)
    gui._render_comparison_figure = lambda summary_text, figure: captured.update({"summary": summary_text, "figure": figure})

    gui._show_group_comparison_window(base_group, exp_group)

    titles = [ax.get_title() for ax in captured["figure"].axes]
    assert titles[:3] == ["分散CNT数量对比", "分散比例对比", "混合评分对比"]
    assert len(captured["figure"].axes) == 5


def test_plot_dispersion_bar_chart_ylim_includes_error_bars_and_pvalue_annotation():
    gui = _make_gui_stub()
    figure = Figure()
    ax = figure.add_subplot(111)
    left_group = {
        "label": "base缁?",
        "dispersed_count_stats": {"mean": 8.0, "std": 28.0},
    }
    right_group = {
        "label": "瀹為獙缁?",
        "dispersed_count_stats": {"mean": 12.0, "std": 18.0},
    }

    gui._plot_dispersion_bar_chart(
        ax,
        left_group,
        right_group,
        "dispersed_count_stats",
        "鍒嗘暎CNT鏁伴噺瀵规瘮",
        "鏁伴噺",
        "{:.1f}",
        test_result={"t_pvalue": 0.012},
    )

    lower, upper = ax.get_ylim()
    assert lower == 0.0
    assert upper > 36.0
    assert any("p=" in text.get_text() for text in ax.texts)


def test_prepare_comparison_display_image_skips_near_identity_resize():
    gui = _make_gui_stub()
    image = np.zeros((620, 1000, 3), dtype=np.uint8)

    prepared = gui._prepare_comparison_display_image(image, max_width=1100, max_height=650)

    assert prepared is image


def test_estimate_comparison_summary_height_stays_compact():
    gui = _make_gui_stub()
    summary_text = "\n".join([
        "识别参数: 模糊9/块11/C3, 长度>=4.0μm",
        "",
        "图像A: sample_a.jpg",
        "图像B: sample_b.jpg",
        "",
        "核心指标",
        "阴影团聚(0-100，越低越好): 图像A 41.3 vs 图像B 40.7",
        "均匀度(0-100，越高越好): 图像A 58.7 vs 图像B 59.3",
        "",
        "结论: 两组整体接近。",
    ])

    height = gui._estimate_comparison_summary_height(summary_text)

    assert 160 <= height <= 220


def test_configure_comparison_image_axis_uses_overlay_caption():
    gui = _make_gui_stub()
    figure = Figure()
    ax = figure.add_subplot(111)

    gui._configure_comparison_image_axis(
        ax,
        image=np.zeros((8, 12, 3), dtype=np.uint8),
        title="图像A典型分布\nsample_a.jpg\nCNT=451",
    )

    assert ax.get_title() == ""
    assert len(ax.texts) == 1
    assert "sample_a.jpg" in ax.texts[0].get_text()
    assert "CNT=451" in ax.texts[0].get_text()


def test_build_spatial_hotspot_grid_highlights_obvious_local_cluster():
    gui = _make_gui_stub()
    spatial = {
        "point_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 9.0, 7.0],
            [0.0, 1.0, 0.0],
        ],
        "coverage_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 0.82, 0.63],
            [0.0, 0.10, 0.0],
        ],
    }

    hotspot_grid, hotspot_info = gui._build_spatial_hotspot_grid(spatial)

    assert hotspot_grid.shape == (3, 3)
    assert hotspot_info["hotspot_regions"] >= 1
    assert hotspot_info["hotspot_mass_ratio"] > 0.5
    assert hotspot_info["peak_share"] > 0.5


def test_build_spatial_hotspot_grid_uses_shadow_support_for_shadow_cluster():
    gui = _make_gui_stub()
    spatial = {
        "point_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        "coverage_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 0.18, 0.06],
            [0.0, 0.0, 0.0],
        ],
        "shadow_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 0.92, 0.78],
            [0.0, 0.08, 0.0],
        ],
    }

    _, hotspot_info = gui._build_spatial_hotspot_grid(spatial)

    assert hotspot_info["hotspot_regions"] >= 1
    assert hotspot_info["shadow_support_ratio"] > 0.5


def test_configure_comparison_image_axis_appends_hotspot_summary_when_spatial_given():
    gui = _make_gui_stub()
    figure = Figure()
    ax = figure.add_subplot(111)
    spatial = {
        "point_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 8.0, 7.0],
            [0.0, 1.0, 0.0],
        ],
        "coverage_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 0.78, 0.61],
            [0.0, 0.08, 0.0],
        ],
        "shadow_density_grid": [
            [0.0, 0.0, 0.0],
            [0.0, 0.88, 0.72],
            [0.0, 0.11, 0.0],
        ],
    }

    gui._configure_comparison_image_axis(
        ax,
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        title="图像A典型分布",
        spatial=spatial,
    )

    assert ax.get_title() == ""
    assert len(ax.texts) == 1
    assert "阴影团聚" in ax.texts[0].get_text()
    assert "均匀度" in ax.texts[0].get_text()


def test_update_advanced_analysis_clears_stale_charts_when_active_context_has_no_measurements():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.analysis_panel = _DummyAnalysisPanel()
    gui.current_roi = SimpleNamespace(measurements=[])
    gui.analyzer = SimpleNamespace(measurements=[object()])
    gui._charts = {
        'score': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'histogram': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'pie': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'cluster': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'heatmap': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'comparison': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
    }

    gui._update_advanced_analysis()

    assert gui.analysis_panel.cleared_keys == ['score', 'histogram', 'pie', 'cluster', 'heatmap']
    assert gui.analysis_panel.refresh_count == 1


def test_update_advanced_analysis_populates_expanded_chart_set():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    measurements = [
        SimpleNamespace(id=1, length_um=10.0, width_mean_um=0.3),
        SimpleNamespace(id=2, length_um=12.0, width_mean_um=0.5),
        SimpleNamespace(id=3, length_um=16.0, width_mean_um=0.7),
    ]
    gui.current_roi = None
    gui._get_active_measurements = lambda: measurements
    gui.analysis_panel = _DummyAnalysisPanel()
    gui.analyzer = SimpleNamespace(
        get_statistics=lambda roi=None: {
            "count": 3,
            "spatial_distribution": {"uniformity_scores": {}, "aggregation_scores": {}},
        },
        get_dispersed_statistics=lambda roi=None: {
            "dispersed_count": 2,
            "agglomerated_count": 1,
            "dispersed_measurements": measurements[:2],
        },
    )

    calls = []
    gui._draw_spatial_score_chart = lambda spatial, cnt_count=None: calls.append(("score", cnt_count))
    gui._draw_distribution_chart = lambda values: calls.append(("histogram", [m.id for m in values]))
    gui._draw_pie_chart = lambda distribution: calls.append(("pie", distribution))
    gui._draw_cluster_analysis = lambda values: calls.append(("cluster", [m.id for m in values]))
    gui._draw_spatial_heatmap = lambda spatial, cnt_count=None: calls.append(("heatmap", cnt_count))

    gui._update_advanced_analysis()

    assert calls == [
        ("score", 3),
        ("histogram", [1, 2]),
        ("pie", {"分散CNT": 2, "团聚CNT": 1}),
        ("cluster", [1, 2]),
        ("heatmap", 3),
    ]
    assert gui.analysis_panel.refresh_count == 1


def test_refresh_interaction_state_uses_active_context_measurements_for_toolbar_actions():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.analyzer = SimpleNamespace(image=object(), rois=[object()], measurements=[object()])
    gui.current_roi = SimpleNamespace(measurements=[])
    gui.control_panel = _DummyControlPanelState()
    gui.image_panel = _DummyImagePanelState()
    gui.save_results_button = _DummyStateWidget()
    gui.export_report_button = _DummyStateWidget()
    gui.compare_analysis_button = _DummyStateWidget()

    gui._refresh_interaction_state()

    assert gui.control_panel.calls[-1] == {'has_image': True, 'has_rois': True}
    assert gui.image_panel.enabled_calls[-1] is True
    assert gui.save_results_button.disabled is True
    assert gui.export_report_button.disabled is True
    assert gui.compare_analysis_button.disabled is False

    gui.current_roi = None
    gui._refresh_interaction_state()

    assert gui.save_results_button.disabled is False
    assert gui.export_report_button.disabled is False


def test_sync_views_runs_unified_refresh_flow_and_clears_canvas_without_image():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.analyzer = SimpleNamespace(image=None)
    gui.image_panel = _DummyImagePanelState()
    gui.current_image = object()
    gui.photo = object()
    calls = []

    gui._update_display = lambda: calls.append("display")
    gui._update_results = lambda: calls.append("results")
    gui._update_advanced_analysis = lambda: calls.append("analysis")
    gui._refresh_interaction_state = lambda: calls.append("controls")
    gui.comparison_panel = None

    gui._sync_views()

    assert calls == ["results", "analysis", "controls"]
    assert gui.image_panel.cleared == 1
    assert gui.image_panel.hidden == 1
    assert gui.current_image is None
    assert gui.photo is None


def test_select_center_tab_switches_notebook_tab_when_target_exists():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.center_notebook = _DummyNotebook(selected="image_tab")
    gui._center_tabs = {
        "image": "image_tab",
        "comparison": "comparison_tab",
    }

    changed = gui._select_center_tab("comparison")

    assert changed is True
    assert gui.center_notebook.selected == "comparison_tab"
    assert gui.center_notebook.select_calls == ["comparison_tab"]
    assert gui.center_notebook.updated == 1


def test_update_results_prioritizes_dispersed_statistics_for_primary_summary():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.current_roi = None
    gui.result_panel = _DummyResultPanelState()
    measurements = [
        SimpleNamespace(id=1, length_um=10.0, width_median_um=0.4, width_iqr_um=0.1),
        SimpleNamespace(id=2, length_um=12.0, width_median_um=0.5, width_iqr_um=0.2),
        SimpleNamespace(id=3, length_um=18.0, width_median_um=0.6, width_iqr_um=0.3),
    ]
    gui._get_active_measurements = lambda: measurements
    gui.analyzer = SimpleNamespace(
        get_statistics=lambda roi=None: {
            "count": 9,
            "length_mean": 99.0,
            "length_std": 10.0,
            "length_min": 80.0,
            "length_max": 120.0,
            "length_distribution": {"A": 9},
            "spatial_distribution": {},
        },
        get_dispersed_statistics=lambda roi=None: {
            "total_count": 9,
            "dispersed_count": 3,
            "agglomerated_count": 6,
            "dispersed_ratio": 1 / 3,
            "dispersed_measurements": measurements[:2],
            "agglomerated_measurements": measurements[2:],
            "dispersed_length_stats": {
                "count": 2,
                "length_mean": 11.0,
                "length_std": 1.0,
                "length_min": 10.0,
                "length_max": 12.0,
                "length_distribution": {"A": 2},
            },
        },
    )

    gui._update_results()

    text = gui.result_panel.stats_text.getvalue()
    assert "3\n" in text
    assert "9\n" in text
    assert "11.00" in text
    assert "99.00" not in text
    assert len(gui.result_panel.rows) == 3
    assert gui.result_panel.rows[0] == (1, "10.00", "是", "")
    assert gui.result_panel.rows[2] == (3, "18.00", "", "是")


def test_update_results_falls_back_to_base_statistics_when_dispersed_stats_fail():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.current_roi = None
    gui.result_panel = _DummyResultPanelState()
    measurements = [
        SimpleNamespace(id=1, length_um=10.0, width_median_um=0.4, width_iqr_um=0.1),
        SimpleNamespace(id=2, length_um=12.0, width_median_um=0.5, width_iqr_um=0.2),
    ]
    gui._get_active_measurements = lambda: measurements
    gui.analyzer = SimpleNamespace(
        get_statistics=lambda roi=None: {
            "count": 2,
            "length_mean": 11.0,
            "length_std": 1.0,
            "length_min": 10.0,
            "length_max": 12.0,
            "length_distribution": {"A": 2},
            "spatial_distribution": {},
        },
        get_dispersed_statistics=lambda roi=None: (_ for _ in ()).throw(ValueError("bad dispersed stats")),
    )

    gui._update_results()

    text = gui.result_panel.stats_text.getvalue()
    assert "2\n\n" in text
    assert "11.00" in text
    assert len(gui.result_panel.rows) == 2
    assert gui.result_panel.rows[0] == (1, "10.00", "", "")


def test_update_results_appends_evaluation_framework_when_available():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui.current_roi = None
    gui.result_panel = _DummyResultPanelState()
    measurements = [
        SimpleNamespace(id=1, length_um=10.0, width_median_um=0.4, width_iqr_um=0.1),
        SimpleNamespace(id=2, length_um=35.0, width_median_um=0.8, width_iqr_um=0.2),
    ]
    gui._get_active_measurements = lambda: measurements
    gui.analyzer = SimpleNamespace(
        get_statistics=lambda roi=None: {
            "count": 2,
            "length_mean": 22.5,
            "length_std": 12.5,
            "length_min": 10.0,
            "length_max": 35.0,
            "length_distribution": {"A": 2},
            "spatial_distribution": {"grid_size": 4, "grid_density_cv": 0.31, "nearest_neighbor_cv": 0.2, "nearest_neighbor_index": 1.1, "grid_entropy": 0.8, "morans_i": 0.12, "occupancy_ratio": 0.5, "uniformity_scores": {"overall": 78.0}},
        },
        get_dispersed_statistics=lambda roi=None: {
            "total_count": 2,
            "dispersed_count": 2,
            "agglomerated_count": 0,
            "dispersed_ratio": 1.0,
            "dispersed_measurements": measurements,
            "agglomerated_measurements": [],
            "dispersed_length_stats": {
                "count": 2,
                "length_mean": 22.5,
                "length_std": 12.5,
                "length_min": 10.0,
                "length_max": 35.0,
                "length_distribution": {"A": 2},
            },
        },
        get_evaluation_framework=lambda roi=None, stats=None, dispersed_stats=None: {
            "uniformity": {"grid_density_cv": 0.31},
            "thick_bundle": {"apparent_width_mean_um": 0.72, "width_p90_um": 0.95},
            "long_cnt": {"skeleton_length_mean_um": 22.5, "ultra_long_threshold_um": 30.0, "ultra_long_ratio": 0.5},
            "agglomeration": {"agglomerated_area_ratio": 0.18, "largest_agglomerate_area_um2": 14.2},
        },
    )

    gui._update_results()

    text = gui.result_panel.stats_text.getvalue()
    assert "评判框架" in text
    assert "A. 均匀性主指标" in text
    assert "P90 宽度" in text
    assert "最大团聚体面积" in text


def test_analyze_image_files_reuses_cache_and_dispatches_only_uncached_tasks():
    gui = _make_gui_stub()
    gui._analysis_cache = OrderedDict()
    gui._analysis_cache_limit = 48

    context = {
        "preprocess_settings": {},
        "detect_settings": {},
        "scale_um": 10.0,
        "manual_scale_pixels": 0.0,
    }
    gui._build_analysis_context = lambda: context

    cached_path = r"C:\tmp\a.png"
    pending_path = r"C:\tmp\b.png"
    failed_path = r"C:\tmp\c.png"
    final_path = r"C:\tmp\d.png"

    cached_result = _make_batch_analysis_result(cached_path, 11.0)
    gui._analysis_cache[gui._make_analysis_cache_key(cached_path, context, False)] = cached_result

    dispatched = []

    def _fake_parallel(tasks, task_context, include_visualization=False, preview_visualization=False):
        dispatched.append((list(tasks), task_context, include_visualization, preview_visualization))
        return [
            (1, _make_batch_analysis_result(pending_path, 12.0), None),
            (2, None, "c.png: bad image"),
            (3, _make_batch_analysis_result(final_path, 14.0), None),
        ]

    gui._run_group_analysis_tasks = _fake_parallel

    results, failures = gui._analyze_image_files(
        [cached_path, pending_path, failed_path, final_path],
        "base组",
    )

    assert [result["path"] for result in results] == [cached_path, pending_path, final_path]
    assert failures == ["c.png: bad image"]
    assert dispatched == [(
        [(1, pending_path), (2, failed_path), (3, final_path)],
        context,
        True,
        True,
    )]
    cached_pending = gui._get_cached_analysis_result(gui._make_analysis_cache_key(pending_path, context, False))
    assert cached_pending is not None
    assert cached_pending["path"] == pending_path


def test_run_group_analysis_tasks_falls_back_to_sequential_when_thread_pool_fails(monkeypatch):
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui._get_group_analysis_worker_count = lambda image_count: 2
    gui._run_image_analysis = lambda image_path, context, include_visualization=False, preview_visualization=False: _make_batch_analysis_result(
        image_path,
        10.0 if image_path.endswith("a.png") else 20.0,
    )

    class _BrokenExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("thread pool unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(gui_module, "ThreadPoolExecutor", _BrokenExecutor)

    results = gui._run_group_analysis_tasks(
        [(0, r"C:\tmp\a.png"), (1, r"C:\tmp\b.png")],
        {"preprocess_settings": {}, "detect_settings": {}, "scale_um": 10.0, "manual_scale_pixels": 0.0},
    )

    assert [item[0] for item in results] == [0, 1]
    assert [item[1]["path"] for item in results] == [r"C:\tmp\a.png", r"C:\tmp\b.png"]
    assert [item[2] for item in results] == [None, None]


def test_run_group_analysis_tasks_skips_process_pool_for_frozen_windows(monkeypatch):
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui._get_group_analysis_worker_count = lambda image_count: 2
    gui._run_image_analysis = lambda image_path, context, include_visualization=False, preview_visualization=False: _make_batch_analysis_result(
        image_path,
        10.0 if image_path.endswith("a.png") else 20.0,
    )
    gui._should_use_process_pool_for_group_analysis = lambda: False

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _ThreadExecutor:
        def __init__(self, *args, **kwargs):
            self.submissions = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submissions.append((fn, args, kwargs))
            return _ImmediateFuture(fn(*args, **kwargs))

    class _ProcessExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("process pool should be skipped for frozen Windows builds")

    monkeypatch.setattr(gui_module, "ThreadPoolExecutor", _ThreadExecutor)
    monkeypatch.setattr(gui_module, "ProcessPoolExecutor", _ProcessExecutor)

    results = gui._run_group_analysis_tasks(
        [(0, r"C:\tmp\a.png"), (1, r"C:\tmp\b.png")],
        {"preprocess_settings": {}, "detect_settings": {}, "scale_um": 10.0, "manual_scale_pixels": 0.0},
    )

    assert [item[0] for item in results] == [0, 1]
    assert [item[1]["path"] for item in results] == [r"C:\tmp\a.png", r"C:\tmp\b.png"]
    assert [item[2] for item in results] == [None, None]


def test_abandon_single_detection_if_running_reenables_detect_button():
    gui = _make_gui_stub()
    gui.MODERN_COLORS = {"warning": "#f59e0b"}
    gui.compare_analysis_button = _DummyStateWidget()
    detect_button = _DummyStateWidget()
    old_executor = _DummyExecutor()
    status_messages = []
    image_statuses = []
    refresh_calls = []

    gui.control_panel = SimpleNamespace(
        detect_button=detect_button,
        update_analysis_status=lambda text, color=None: status_messages.append((text, color)),
    )
    gui.image_panel = SimpleNamespace(show_status=lambda text: image_statuses.append(text))
    gui._refresh_interaction_state = lambda: refresh_calls.append(True)
    gui._single_detect_executor = old_executor
    gui._single_detect_future = _RunningFuture()
    gui._single_detect_snapshot = {"image_id": 1}
    gui._single_detect_token = 7

    abandoned = gui._abandon_single_detection_if_running()

    assert abandoned is True
    assert gui._single_detect_future is None
    assert gui._single_detect_snapshot is None
    assert gui._single_detect_token == 8
    assert old_executor.shutdown_calls == [(False, True)]
    assert detect_button.disabled is False
    assert gui.compare_analysis_button.disabled is False
    assert refresh_calls == [True]
    assert status_messages[-1][0].startswith("检测参数已更新")
    assert image_statuses[-1] == "检测参数已更新，可重新开始CNT检测"

def test_load_image_common_discards_stale_single_detection_state(monkeypatch):
    gui = _make_gui_stub()
    gui.MODERN_COLORS = {"warning": "#f59e0b"}
    gui.compare_analysis_button = _DummyStateWidget()
    detect_button = _DummyStateWidget()
    old_executor = _DummyExecutor()
    refresh_calls = []
    reset_calls = []
    statuses = []
    replacement_executors = []

    class _ReplacementExecutor(_DummyExecutor):
        pass

    def _fake_thread_pool(*args, **kwargs):
        executor = _ReplacementExecutor()
        replacement_executors.append((executor, args, kwargs))
        return executor

    monkeypatch.setattr(gui_module, "ThreadPoolExecutor", _fake_thread_pool)

    gui.control_panel = SimpleNamespace(
        detect_button=detect_button,
        update_analysis_status=lambda *args, **kwargs: None,
    )
    gui.image_panel = SimpleNamespace(show_status=lambda text: statuses.append(text))
    gui._refresh_interaction_state = lambda: refresh_calls.append(True)
    gui._single_detect_executor = old_executor
    gui._single_detect_future = _RunningFuture()
    gui._single_detect_snapshot = {"image_id": 1}
    gui._single_detect_token = 11
    gui._reset_display = lambda: reset_calls.append(True)
    gui._refresh_scale_status_ui = lambda: None
    gui._auto_suggest_params = lambda: None
    gui._refresh_analysis_status_ui = lambda: None
    gui.live_preview_var = _DummyVar(False)
    gui.scale_um_var = _DummyVar(0.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    gui.analyzer = SimpleNamespace(
        apply_detected_scale=lambda default_micrometers: {"applied": False, "scale_info": None},
    )

    gui._load_image_common()

    assert gui._single_detect_future is None
    assert gui._single_detect_snapshot is None
    assert gui._single_detect_token == 12
    assert old_executor.shutdown_calls == [(False, True)]
    assert len(replacement_executors) == 1
    assert isinstance(gui._single_detect_executor, _ReplacementExecutor)
    assert detect_button.disabled is False
    assert gui.compare_analysis_button.disabled is False
    assert refresh_calls == [True]
    assert reset_calls == [True]
    assert len(statuses) == 1


def test_discard_single_detection_state_can_drop_completed_future_without_notifying():
    gui = _make_gui_stub()
    gui.MODERN_COLORS = {"warning": "#f59e0b"}
    gui.compare_analysis_button = _DummyStateWidget()
    detect_button = _DummyStateWidget()
    refresh_calls = []
    image_statuses = []
    status_messages = []

    gui.control_panel = SimpleNamespace(
        detect_button=detect_button,
        update_analysis_status=lambda text, color=None: status_messages.append((text, color)),
    )
    gui.image_panel = SimpleNamespace(show_status=lambda text: image_statuses.append(text))
    gui._refresh_interaction_state = lambda: refresh_calls.append(True)
    gui._single_detect_executor = _DummyExecutor()
    gui._single_detect_future = _DoneFuture()
    gui._single_detect_snapshot = {"image_id": 2}
    gui._single_detect_token = 3

    discarded = gui._discard_single_detection_state(include_completed=True, notify=False)

    assert discarded is True
    assert gui._single_detect_future is None
    assert gui._single_detect_snapshot is None
    assert gui._single_detect_token == 4
    assert refresh_calls == [True]
    assert status_messages == []
    assert image_statuses == []


def test_get_detection_filter_settings_reports_invalid_range():
    gui = _make_gui_stub()
    gui.min_length_um_var = _DummyVar(20.0)
    gui.max_length_um_var = _DummyVar(10.0)
    gui.min_slenderness_var = _DummyVar(4.0)
    gui.merge_distance_px_var = _DummyVar(6.0)

    result = gui._get_detection_filter_settings(strict=False)

    assert result["valid"] is False
    assert "最大长度不能小于最小长度" in result["message"]


def test_group_analysis_process_pool_is_disabled_for_gui_runtime():
    assert CNTAnalyzerGUI._should_use_process_pool_for_group_analysis() is False
    assert CNTAnalyzerGUI._should_use_process_pool_for_group_analysis(platform="win32", frozen=False) is False
    assert CNTAnalyzerGUI._should_use_process_pool_for_group_analysis(platform="linux", frozen=False) is False


def test_run_group_analysis_task_collects_expected_analysis_errors():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    gui._run_image_analysis = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad image"))

    result = gui._run_group_analysis_task(
        (3, r"C:\tmp\broken.png"),
        {"preprocess_settings": {}, "detect_settings": {}, "scale_um": 10.0, "manual_scale_pixels": 0.0},
    )

    assert result == (3, None, "broken.png: bad image")


def test_compare_two_images_uses_batch_analysis_pipeline(monkeypatch):
    gui = _make_gui_stub()
    gui._get_compare_initial_dir = lambda: r"C:\tmp"
    gui._get_supported_image_filetypes = lambda: [("图像文件", "*.png")]

    selected_paths = iter([r"C:\tmp\a.png", r"C:\tmp\b.png"])
    monkeypatch.setattr(gui_module.filedialog, "askopenfilename", lambda **kwargs: next(selected_paths))

    requests = []
    gui._start_compare_analysis = lambda request: requests.append(dict(request))

    gui._compare_two_images()

    assert requests == [{
        "mode": "pair",
        "left_path": r"C:\tmp\a.png",
        "right_path": r"C:\tmp\b.png",
        "total_images": 2,
    }]


def test_start_compare_analysis_freezes_context_and_submits_background_task():
    gui = _make_gui_stub()
    gui.root = _DummyRoot()
    gui.MODERN_COLORS = {"warning": "#f59e0b"}
    gui.comparison_panel = _DummyComparisonPanel()
    gui.compare_analysis_button = _DummyStateWidget()
    detect_button = _DummyStateWidget()
    selected_tabs = []

    gui.control_panel = SimpleNamespace(detect_button=detect_button)
    gui._refresh_interaction_state = lambda: None
    gui._select_center_tab = lambda key: selected_tabs.append(key)
    gui._build_compare_analysis_context = lambda: {"token": "frozen-context"}
    gui._compare_executor = _DummySubmitExecutor(_RunningFuture())

    gui._start_compare_analysis({
        "mode": "group",
        "base_paths": [r"C:\tmp\base_a.png"],
        "exp_paths": [r"C:\tmp\exp_a.png", r"C:\tmp\exp_b.png"],
        "total_images": 3,
    })

    submission = gui._compare_executor.submissions[0]
    submitted_request = submission[1][0]
    assert submitted_request["context"] == {"token": "frozen-context"}
    assert gui._compare_snapshot["request"]["context"] == {"token": "frozen-context"}
    assert gui._compare_future is gui._compare_executor.future
    assert gui.comparison_panel.progress_visible is True
    assert gui.comparison_panel.progress_updates[0] == (0, 3, "准备开始...")
    assert gui.compare_analysis_button.disabled is True
    assert detect_button.disabled is True
    assert selected_tabs == ["comparison"]
    assert gui.root.after_calls[0][0] == 80


def test_poll_compare_analysis_result_renders_pair_result_and_resets_busy_state():
    gui = _make_gui_stub()
    gui.root = _DummyRoot()
    gui.comparison_panel = _DummyComparisonPanel()
    gui.compare_analysis_button = _DummyStateWidget()
    detect_button = _DummyStateWidget()
    refresh_calls = []
    rendered = []

    gui.control_panel = SimpleNamespace(detect_button=detect_button)
    gui._refresh_interaction_state = lambda: refresh_calls.append(True)
    gui._show_comparison_window = lambda left, right, left_label, right_label, note=None: rendered.append(
        (left["path"], right["path"], left_label, right_label, note)
    )
    gui._compare_token = 5
    gui._compare_future = _ResultFuture({
        "mode": "pair",
        "left_result": {"path": r"C:\tmp\a.png"},
        "right_result": {"path": r"C:\tmp\b.png"},
        "note": "done",
    })
    gui._compare_snapshot = {
        "progress_state": {"current": 2, "total": 2, "message": "分析完成"},
    }

    gui._poll_compare_analysis_result(5)

    assert rendered == [(r"C:\tmp\a.png", r"C:\tmp\b.png", "图像A", "图像B", "done")]
    assert gui._compare_future is None
    assert gui._compare_snapshot is None
    assert gui.comparison_panel.progress_visible is False
    assert gui.comparison_panel.progress_updates[-1] == (2, 2, "分析完成")
    assert gui.compare_analysis_button.disabled is False
    assert detect_button.disabled is False
    assert refresh_calls == [True]


def test_build_compare_analysis_context_uses_current_preprocess_triplet():
    gui = _make_gui_stub()
    gui.max_length_um_var = _DummyVar(200.0)
    gui.scale_um_var = _DummyVar(10.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    gui._get_detection_profile_key = lambda: "balanced"

    gui.blur_kernel_var.set(15)
    gui.adaptive_block_var.set(21)
    gui.adaptive_c_var.set(5)
    gui.bridge_strength_var.set(4)

    context = gui._build_compare_analysis_context()

    assert context["preprocess_settings"] == {
        "blur_kernel": 15,
        "adaptive_block": 21,
        "adaptive_c": 5,
        "bridge_strength": 4,
        "threshold_invert": True,
    }
    assert context["detect_settings"]["min_length_um"] == pytest.approx(4.0)
    assert context["detect_settings"]["max_length_um"] == pytest.approx(200.0)
    assert context["detect_settings"]["min_slenderness"] == pytest.approx(3.0)
    assert context["analysis_roi"] == {
        "mode": "center_fraction",
        "fraction": 0.75,
        "label": "中部75%区域",
    }
    assert context["scale_detection"] == {"recognize_text": False}


def test_invoke_compare_batch_analysis_passes_fixed_context_when_supported():
    gui = _make_gui_stub()
    gui.max_length_um_var = _DummyVar(200.0)
    gui.scale_um_var = _DummyVar(10.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    gui._get_detection_profile_key = lambda: "balanced"

    captured = {}

    def _fake_analyze(paths, group_label, context=None):
        captured["paths"] = list(paths)
        captured["group_label"] = group_label
        captured["context"] = context
        return [], []

    gui._analyze_image_files = _fake_analyze

    gui._invoke_compare_batch_analysis([r"C:\tmp\a.png"], "双图对比")

    assert captured["paths"] == [r"C:\tmp\a.png"]
    assert captured["group_label"] == "双图对比"
    assert captured["context"]["preprocess_settings"]["blur_kernel"] == 9
    assert captured["context"]["preprocess_settings"]["adaptive_block"] == 11
    assert captured["context"]["preprocess_settings"]["adaptive_c"] == 3
    assert captured["context"]["analysis_roi"]["fraction"] == pytest.approx(0.75)
    assert captured["context"]["scale_detection"]["recognize_text"] is False


def test_invoke_compare_batch_analysis_can_skip_visualization_for_group_mode():
    gui = _make_gui_stub()
    gui.max_length_um_var = _DummyVar(200.0)
    gui.scale_um_var = _DummyVar(10.0)
    gui.scale_pixels_var = _DummyVar(0.0)
    gui._get_detection_profile_key = lambda: "balanced"

    captured = {}

    def _fake_analyze(paths, group_label, context=None, include_visualization=True, preview_visualization=True):
        captured["paths"] = list(paths)
        captured["group_label"] = group_label
        captured["context"] = context
        captured["include_visualization"] = include_visualization
        captured["preview_visualization"] = preview_visualization
        return [], []

    gui._analyze_image_files = _fake_analyze

    gui._invoke_compare_batch_analysis([r"C:\tmp\a.png"], "base组", include_visualization=False)

    assert captured["paths"] == [r"C:\tmp\a.png"]
    assert captured["group_label"] == "base组"
    assert captured["include_visualization"] is False
    assert captured["preview_visualization"] is False


def test_get_group_analysis_worker_count_uses_conservative_cap(monkeypatch):
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)

    monkeypatch.setattr(gui_module.os, "cpu_count", lambda: 32)
    assert gui._get_group_analysis_worker_count(20) == 8
    assert gui._get_group_analysis_worker_count(3) == 3

    monkeypatch.setattr(gui_module.os, "cpu_count", lambda: 2)
    assert gui._get_group_analysis_worker_count(10) == 1


def test_run_image_analysis_uses_center_roi_and_crops_visualization_for_compare_context(monkeypatch):
    captured = {}

    class _DummyAnalyzer:
        def __init__(self):
            self.image = np.zeros((200, 400, 3), dtype=np.uint8)

        def load_image(self, path):
            captured["path"] = path

        def apply_detected_scale(self, default_micrometers, recognize_text=True):
            captured["scale_um"] = default_micrometers
            captured["recognize_text"] = recognize_text
            return {"applied": True, "scale_info": {"pixels": 100}}

        def preprocess(self, roi=None, **kwargs):
            captured["preprocess_roi"] = roi

        def detect_cnts_hybrid(self, roi=None, **kwargs):
            captured["detect_roi"] = roi
            return []

        def get_statistics(self, roi=None):
            captured["stats_roi"] = roi
            return {"count": 0}

        def get_dispersed_statistics(self, roi=None):
            captured["dispersed_roi"] = roi
            return {"dispersed_count": 0, "agglomerated_count": 0, "dispersed_ratio": 0.0}

        def get_visualization(self, roi=None):
            captured["visualization_roi"] = roi
            return np.zeros((200, 400, 3), dtype=np.uint8)

        def get_scale_status(self):
            return {"source": "auto_detected"}

    monkeypatch.setattr(gui_module, "CNTAnalyzer", _DummyAnalyzer)

    gui = _make_gui_stub()
    gui._prepare_comparison_display_image = lambda image, max_width=1000, max_height=560: image

    context = {
        "preprocess_settings": {"blur_kernel": 9, "adaptive_block": 11, "adaptive_c": 3, "bridge_strength": 0, "threshold_invert": True},
        "detect_settings": {"min_length_um": 4.0, "max_length_um": 200.0, "min_slenderness": 3.0, "detection_profile": "balanced", "merge_distance_px": 0.0, "split_mode": "off", "roi": None},
        "scale_um": 10.0,
        "manual_scale_pixels": 0.0,
        "analysis_roi": {"mode": "center_fraction", "fraction": 0.75, "label": "中部75%区域"},
        "scale_detection": {"recognize_text": False},
    }

    result = gui._run_image_analysis(r"C:\tmp\sample.png", context, include_visualization=True, preview_visualization=True)

    roi = captured["detect_roi"]
    assert roi is not None
    assert (roi.x, roi.y, roi.width, roi.height) == (50, 25, 300, 150)
    assert captured["preprocess_roi"] is roi
    assert captured["stats_roi"] is roi
    assert captured["dispersed_roi"] is roi
    assert captured["visualization_roi"] is roi
    assert captured["recognize_text"] is False
    assert result["visualization"].shape == (150, 300, 3)
    assert result["analysis_roi"] == {
        "name": "中部75%区域",
        "x": 50,
        "y": 25,
        "width": 300,
        "height": 150,
    }


def test_analyze_image_files_stores_visualization_preview_for_group_results():
    gui = _make_gui_stub()
    gui._analysis_cache = OrderedDict()
    gui._analysis_cache_limit = 48

    context = {
        "preprocess_settings": {},
        "detect_settings": {},
        "scale_um": 10.0,
        "manual_scale_pixels": 0.0,
    }
    gui._build_analysis_context = lambda: context

    preview_result = _make_batch_analysis_result(r"C:\tmp\a.png", 18.0)
    preview_result["visualization"] = np.zeros((32, 48, 3), dtype=np.uint8)

    calls = []

    def _fake_tasks(tasks, task_context, include_visualization=False, preview_visualization=False):
        calls.append((list(tasks), task_context, include_visualization, preview_visualization))
        return [(0, preview_result, None)]

    gui._run_group_analysis_tasks = _fake_tasks

    results, failures = gui._analyze_image_files([r"C:\tmp\a.png"], "base组")

    assert failures == []
    assert results[0]["visualization"] is not None
    assert calls == [(
        [(0, r"C:\tmp\a.png")],
        context,
        True,
        True,
    )]
    visual_cache = gui._get_cached_analysis_result(gui._make_analysis_cache_key(r"C:\tmp\a.png", context, True))
    assert visual_cache is not None
    assert visual_cache["visualization"].shape == (32, 48, 3)


def test_select_representative_result_reuses_existing_visualization_without_reanalysis():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    representative = _make_batch_analysis_result(r"C:\tmp\rep.png", 20.0)
    representative["name"] = "rep.png"
    representative["visualization"] = np.zeros((12, 20, 3), dtype=np.uint8)
    other = _make_batch_analysis_result(r"C:\tmp\other.png", 18.0)
    other["name"] = "other.png"

    group_summary = {
        "label": "base组",
        "results": [representative, other],
        "file_details": [
            {"name": "rep.png", "shadow_aggregation_score": 40.0, "uniformity_score": 60.0},
            {"name": "other.png", "shadow_aggregation_score": 55.0, "uniformity_score": 45.0},
        ],
        "spatial_stats": {
            "shadow_aggregation_score": {"mean": 41.0, "std": 4.0},
            "uniformity_score": {"mean": 59.0, "std": 4.0},
        },
    }

    def _fail_reanalysis(*args, **kwargs):
        raise AssertionError("should not re-run analysis for representative visualization")

    gui._analyze_image_file = _fail_reanalysis

    selected = gui._select_representative_result(group_summary)

    assert selected is representative


def test_select_representative_result_reuses_stored_compare_context_for_visualization():
    gui = CNTAnalyzerGUI.__new__(CNTAnalyzerGUI)
    context = {"token": "compare"}
    representative = _make_batch_analysis_result(r"C:\tmp\rep.png", 20.0)
    representative["name"] = "rep.png"
    representative["analysis_context"] = context
    other = _make_batch_analysis_result(r"C:\tmp\other.png", 18.0)
    other["name"] = "other.png"
    other["analysis_context"] = context

    group_summary = {
        "label": "base组",
        "results": [representative, other],
        "file_details": [
            {"name": "rep.png", "shadow_aggregation_score": 40.0, "uniformity_score": 60.0},
            {"name": "other.png", "shadow_aggregation_score": 55.0, "uniformity_score": 45.0},
        ],
        "spatial_stats": {
            "shadow_aggregation_score": {"mean": 41.0, "std": 4.0},
            "uniformity_score": {"mean": 59.0, "std": 4.0},
        },
    }

    calls = []
    gui._build_compare_analysis_context = lambda: (_ for _ in ()).throw(AssertionError("should reuse stored context"))
    gui._analyze_image_with_context = lambda path, used_context, include_visualization=False, preview_visualization=True: (
        calls.append((path, used_context, include_visualization, preview_visualization)) or {**representative, "visualization": np.zeros((8, 12, 3), dtype=np.uint8)}
    )

    selected = gui._select_representative_result(group_summary)

    assert selected["visualization"] is not None
    assert calls == [(r"C:\tmp\rep.png", context, True, True)]
