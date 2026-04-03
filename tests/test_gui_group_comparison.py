import matplotlib
import pytest
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

matplotlib.use("Agg")

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
    assert summary["spatial_stats"]["aggregation_score"]["mean"] == pytest.approx(42.5)
    assert summary["spatial_stats"]["aggregation_moran_score"]["max"] == pytest.approx(45.0)


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

    assert "数量统计" in summary
    assert "均匀性对比" in summary
    assert "结论: 实验组显著优于base组" in summary
    assert "团聚风险总分(0-100↓)" in summary
    assert "综合均匀性(0-100↑)" in summary
    assert "Moran's I(↑聚集)" in summary
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
    assert "统计证据有限" in summary
    assert "结论: 实验组显著优于base组" not in summary


def test_format_comparison_summary_uses_compact_text():
    gui = _make_gui_stub()
    left_result = _make_result("left", 48.0, 60.0, 61.0, 59.0, 60.0, 0.39, 0.41, 0.18)
    right_result = _make_result("right", 41.0, 47.0, 46.0, 48.0, 47.0, 0.51, 0.53, 0.31)

    summary = gui._format_comparison_summary(left_result, right_result, "样品A", "样品B")

    assert summary.startswith("识别参数: ")
    assert "分布对比" in summary
    assert "团聚风险总分(0-100↓)" in summary
    assert "结论: 样品A整体更优" in summary
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

    assert ax.get_title() == "团聚风险 vs 均匀性优势"
    assert ax.get_ylim() == (0.0, 100.0)
    assert ax.get_legend() is not None
    assert ax.get_legend().get_title().get_text() == "数值越大 = 越团聚"
    assert ax.patches[0].get_facecolor() == pytest.approx(to_rgba(CNTAnalyzerGUI.MODERN_COLORS["accent_rose"], alpha=0.95))
    assert ax.patches[4].get_facecolor() == pytest.approx(to_rgba(CNTAnalyzerGUI.MODERN_COLORS["accent_teal"], alpha=0.82))


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
        "图像A: sample_a.jpg | CNT=451",
        "图像B: sample_b.jpg | CNT=409",
        "数量差异: 图像A +10.3%",
        "",
        "分布对比",
        "团聚风险总分(0-100): 图像A 41.3 vs 图像B 40.7",
        "综合均匀性: 图像A 58.7 vs 图像B 59.3",
        "最近邻CV: 图像A 0.483 vs 图像B 0.467",
        "Moran's I: 图像A 0.121 vs 图像B 0.124",
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
        title="图像A典型CNT分布\nsample_a.jpg\nCNT=451",
    )

    assert ax.get_title() == ""
    assert len(ax.texts) == 1
    assert "CNT=451" in ax.texts[0].get_text()
