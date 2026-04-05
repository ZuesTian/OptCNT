import matplotlib
import pytest
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

matplotlib.use("Agg")

import gui as gui_module
from gui import CNTAnalyzerGUI


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


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

    def clear_chart_content(self, key: str):
        self.cleared_keys.append(key)

    def refresh_layout(self):
        self.refresh_count += 1

    def scroll_to_top(self):
        self.scroll_count += 1


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
    gui.blur_kernel_var = _DummyVar(9)
    gui.adaptive_block_var = _DummyVar(11)
    gui.adaptive_c_var = _DummyVar(3)
    gui.bridge_strength_var = _DummyVar(2)
    gui.min_length_um_var = _DummyVar(4.0)
    gui.min_slenderness_var = _DummyVar(3.0)
    gui.detect_profile_var = _DummyVar("标准（推荐）")
    gui.split_mode_var = _DummyVar("不拆分")
    gui.merge_distance_px_var = _DummyVar(6)
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


def test_format_comparison_summary_uses_compact_text():
    gui = _make_gui_stub()
    left_result = _make_result("left", 48.0, 60.0, 61.0, 59.0, 60.0, 0.39, 0.41, 0.18)
    right_result = _make_result("right", 41.0, 47.0, 46.0, 48.0, 47.0, 0.51, 0.53, 0.31)

    summary = gui._format_comparison_summary(left_result, right_result, "样品A", "样品B")

    assert summary.startswith("识别参数: ")
    assert "CNT数量" in summary
    assert "样品A: left | CNT=48" in summary
    assert "样品B: right | CNT=41" in summary
    assert "核心指标" in summary
    assert "阴影团聚(0-100，越低越好)" in summary
    assert "均匀度(0-100，越高越好)" in summary
    assert "结论: 样品A更优" in summary
    assert "Mann-Whitney" not in summary


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
    assert 1080 <= width_px <= 1136


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
        'histogram': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'pie': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'cluster': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'heatmap': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
        'comparison': {'fig': None, 'ax': None, 'canvas': None, 'draw_count': 0},
    }

    gui._update_advanced_analysis()

    assert gui.analysis_panel.cleared_keys == ['histogram', 'pie', 'cluster', 'heatmap']
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
