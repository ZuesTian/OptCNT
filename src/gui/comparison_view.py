"""Comparison view helpers for CNTAnalyzerGUI."""

from __future__ import annotations

from types import MethodType
from typing import List, Optional
import tkinter as tk

import cv2
import numpy as np
from matplotlib.figure import Figure

from .chart_manager import ensure_chart_manager


def _fit_comparison_figure_to_frame(
    self,
    figure: Figure,
    chart_frame: Optional[tk.Widget],
    min_width_px: int = 680,
    padding_px: int = 30,
    max_width_px: int = 1400,
    allow_expand: bool = False,
) -> None:
    """按中间栏可用宽度等比缩放对比图。"""
    if chart_frame is None:
        return
    min_width_px = self._scale_px(min_width_px)
    padding_px = self._scale_px(padding_px)
    max_width_px = self._scale_px(max_width_px)
    chart_frame.update_idletasks()
    available_width = max(
        int(chart_frame.winfo_width()) - padding_px,
        int(getattr(self, 'center_notebook', chart_frame).winfo_width()) - padding_px,
        min_width_px,
    )
    target_width_px = max(min_width_px, min(max_width_px, available_width))
    current_width_px = max(1, int(round(figure.get_figwidth() * figure.dpi)))
    current_height_px = max(1, int(round(figure.get_figheight() * figure.dpi)))
    tolerance = 12
    if abs(current_width_px - target_width_px) <= tolerance:
        return
    if current_width_px <= target_width_px and not allow_expand:
        return
    scale = target_width_px / current_width_px
    scale = max(0.5, min(1.5, scale))
    resized_width_px = max(min_width_px, int(round(current_width_px * scale)))
    resized_height_px = max(480, int(round(current_height_px * scale)))
    figure.set_size_inches(resized_width_px / figure.dpi, resized_height_px / figure.dpi, forward=False)


def _schedule_comparison_layout_refresh(self, delay_ms: int = 80) -> None:
    """防抖刷新对比分析页的图表尺寸与滚动区域。"""
    if getattr(self, '_comparison_layout_job', None) is not None:
        self.root.after_cancel(self._comparison_layout_job)
    self._comparison_layout_job = self.root.after(delay_ms, self._refresh_comparison_layout)


def _refresh_comparison_layout(self) -> None:
    """在标签页显示或窗口尺寸变化后，重新贴合对比图尺寸。"""
    self._comparison_layout_job = None
    if not self.comparison_panel or not hasattr(self, 'center_notebook'):
        return
    comparison_tab = self._center_tabs.get('comparison')
    if comparison_tab is None or str(self.center_notebook.select()) != str(comparison_tab):
        return
    chart = ensure_chart_manager(self).get_chart('comparison')
    figure = chart.get('fig')
    canvas = chart.get('canvas')
    chart_frame = self.comparison_panel.get_chart_frame('comparison')
    if figure is None or canvas is None or chart_frame is None:
        self.comparison_panel.refresh_layout()
        return
    old_size = (
        int(round(figure.get_figwidth() * figure.dpi)),
        int(round(figure.get_figheight() * figure.dpi)),
    )
    self._fit_comparison_figure_to_frame(figure, chart_frame, allow_expand=False)
    new_height = min(1800, max(800, int(round(figure.get_figheight() * figure.dpi)) + 50))
    self.comparison_panel.set_section_height('comparison', new_height)
    new_size = (
        int(round(figure.get_figwidth() * figure.dpi)),
        int(round(figure.get_figheight() * figure.dpi)),
    )
    if abs(new_size[0] - old_size[0]) > 5 or abs(new_size[1] - old_size[1]) > 5:
        canvas.draw()
    self.comparison_panel.refresh_layout()


def _render_comparison_figure(self, summary_text: str, figure: Figure):
    """将对比摘要和图表渲染到对比分析面板。"""
    if not self.comparison_panel:
        return
    self._select_center_tab('comparison')
    self.comparison_panel.refresh_layout()
    chart_frame = self.comparison_panel.get_chart_frame('comparison')
    self._fit_comparison_figure_to_frame(figure, chart_frame, allow_expand=True)
    summary_height = self._estimate_comparison_summary_height(summary_text)
    chart_height = min(1600, max(720, int(figure.get_size_inches()[1] * figure.dpi) + 40))
    self.comparison_panel.set_section_height('comparison_summary', summary_height)
    self.comparison_panel.set_section_height('comparison', chart_height)
    self.comparison_panel.set_text_content('comparison_summary', summary_text)
    if chart_frame is None:
        return
    ensure_chart_manager(self).mount_comparison_figure(figure, chart_frame, padx=8, pady=8)
    self.comparison_panel.refresh_layout()
    self.comparison_panel.scroll_to_top()
    self._schedule_comparison_layout_refresh(delay_ms=60)
    self._schedule_comparison_layout_refresh(delay_ms=180)


def _should_stack_comparison_images(
    self,
    *images: Optional[np.ndarray],
    threshold: float = 1.3,
) -> bool:
    """宽图优先采用上下排布，避免代表图被压扁。"""
    aspects = [
        self._get_comparison_image_aspect(image)
        for image in images
        if image is not None and getattr(image, 'size', 0) > 0
    ]
    if not aspects:
        return False
    return max(aspects) >= threshold or (sum(aspects) / len(aspects)) >= (threshold - 0.15)


def _build_comparison_layout(
    self,
    *images: Optional[np.ndarray],
    stacked: bool,
    variant: str,
) -> dict:
    """返回稳定的预设对比布局，减少不同图像宽高比带来的抖动。"""
    if variant == 'pair':
        if stacked:
            return {
                'figsize': (13.4, 11.5),
                'height_ratios': [0.75, 1.6, 1.6],
                'hspace': 0.40,
                'wspace': 0.25,
                'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.05},
            }
        return {
            'figsize': (13.2, 9.5),
            'height_ratios': [0.85, 1.4],
            'hspace': 0.32,
            'wspace': 0.25,
            'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.06},
        }

    if stacked:
        return {
            'figsize': (14.5, 14.2),
            'height_ratios': [0.80, 0.70, 0.75, 1.5, 1.5],
            'hspace': 0.44,
            'wspace': 0.32,
            'adjust': {'left': 0.055, 'right': 0.97, 'top': 0.97, 'bottom': 0.04},
        }
    return {
        'figsize': (14.5, 11.8),
        'height_ratios': [0.90, 0.80, 1.5],
        'hspace': 0.36,
        'wspace': 0.35,
        'adjust': {'left': 0.055, 'right': 0.97, 'top': 0.96, 'bottom': 0.045},
    }


def _prepare_comparison_display_image(
    self,
    image: Optional[np.ndarray],
    max_width: int = 1100,
    max_height: int = 650,
) -> Optional[np.ndarray]:
    """为对比页预缩放大图，降低渲染压力。"""
    if image is None or getattr(image, 'size', 0) == 0:
        return image
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return image
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)
    if scale >= 0.95:
        return image
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _configure_comparison_image_axis(
    self,
    ax,
    image: np.ndarray,
    title: str,
    spatial: Optional[dict] = None,
):
    """以正确比例显示对比中的代表图，并附加热点摘要。"""
    if image is None or getattr(image, 'size', 0) == 0:
        ax.text(0.5, 0.5, "暂无图像", ha='center', va='center', transform=ax.transAxes, color=self.MODERN_COLORS['text_secondary'])
        ax.axis('off')
        return

    display_image = self._prepare_comparison_display_image(image)
    ax.imshow(display_image, interpolation='bilinear', aspect='equal')
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.0)
    ax.set_anchor('C')
    ax.text(
        0.02,
        0.98,
        title + self._summarize_spatial_hotspots_for_caption(spatial),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9.5,
        linespacing=1.2,
        color=self.MODERN_COLORS['text_primary'],
        bbox={
            'boxstyle': 'round,pad=0.35',
            'facecolor': self.MODERN_COLORS['bg_secondary'],
            'edgecolor': self.MODERN_COLORS['border'],
            'alpha': 0.90,
            'linewidth': 1.5,
        },
    )
    ax.axis('off')


def _estimate_comparison_summary_height(self, summary_text: str) -> int:
    """根据摘要文本估算紧凑摘要区高度。"""
    logical_lines = summary_text.splitlines() or [summary_text]
    wrapped_lines = sum(max(1, (len(line) + 71) // 72) for line in logical_lines)
    logical_height = min(220, max(128, 34 + wrapped_lines * 14))
    return self._scale_px(logical_height)


def _format_representative_caption(self, group_summary: dict, representative_result: dict) -> str:
    """生成代表图标题，聚焦分散数量与分散比例。"""
    dispersed_stats = representative_result.get('dispersed_stats') or {}
    total_count = int(round(float(representative_result.get('stats', {}).get('count', 0.0))))
    dispersed_count = int(round(float(dispersed_stats.get('dispersed_count', total_count))))
    dispersed_ratio = float(
        dispersed_stats.get(
            'dispersed_ratio',
            1.0 if total_count > 0 and dispersed_count == total_count else 0.0,
        )
    )
    return (
        f"{group_summary.get('label', '对象')}典型分布\n"
        f"{representative_result.get('name', '未命名图像')}\n"
        f"分散CNT={dispersed_count} | 分散比例={dispersed_ratio:.1%}"
    )


def _format_group_comparison_summary(
    self,
    base_group: dict,
    exp_group: dict,
    note: Optional[str] = None,
    failures: Optional[List[str]] = None,
) -> str:
    """生成精简版组别对比摘要。"""
    base_count = base_group['count_stats']
    exp_count = exp_group['count_stats']
    base_dispersed = base_group.get('dispersed_count_stats', self._summarize_numeric_series([]))
    exp_dispersed = exp_group.get('dispersed_count_stats', self._summarize_numeric_series([]))
    base_agglomerated = base_group.get('agglomerated_count_stats', self._summarize_numeric_series([]))
    exp_agglomerated = exp_group.get('agglomerated_count_stats', self._summarize_numeric_series([]))
    base_ratio = base_group.get('dispersed_ratio_stats', self._summarize_numeric_series([]))
    exp_ratio = exp_group.get('dispersed_ratio_stats', self._summarize_numeric_series([]))
    base_dispersed_length = base_group.get('dispersed_length_mean_stats', self._summarize_numeric_series([]))
    exp_dispersed_length = exp_group.get('dispersed_length_mean_stats', self._summarize_numeric_series([]))
    base_long_thick_count = base_group.get('long_thick_count_stats', self._summarize_numeric_series([]))
    exp_long_thick_count = exp_group.get('long_thick_count_stats', self._summarize_numeric_series([]))
    base_long_thick_ratio = base_group.get('long_thick_ratio_stats', self._summarize_numeric_series([]))
    exp_long_thick_ratio = exp_group.get('long_thick_ratio_stats', self._summarize_numeric_series([]))
    base_spatial = base_group['spatial_stats']
    exp_spatial = exp_group['spatial_stats']
    base_framework = self._build_group_framework_summary(base_group)
    exp_framework = self._build_group_framework_summary(exp_group)
    compare_context = self._get_compare_display_context(
        *(base_group.get('results') or []),
        *(exp_group.get('results') or []),
    )
    tests = self._compute_group_comparison_tests(base_group, exp_group)

    count_tests = tests['count']
    hybrid_tests = tests['hybrid_score']
    long_thick_count_tests = tests['long_thick_count']
    long_thick_ratio_tests = tests['long_thick_ratio']
    shadow_tests = tests['shadow_aggregation_score']
    uniformity_tests = tests['uniformity_score']
    shadow_diff = base_spatial['shadow_aggregation_score']['mean'] - exp_spatial['shadow_aggregation_score']['mean']
    uniformity_diff = exp_spatial['uniformity_score']['mean'] - base_spatial['uniformity_score']['mean']
    evidence_significant = (
        self._is_test_significant(shadow_tests) or
        self._is_test_significant(uniformity_tests)
    )

    strong_exp_advantage = shadow_diff >= 4.0 and uniformity_diff >= 4.0 and evidence_significant
    trend_exp_advantage = shadow_diff > 0 and uniformity_diff > 0
    strong_base_advantage = shadow_diff <= -4.0 and uniformity_diff <= -4.0 and evidence_significant
    trend_base_advantage = shadow_diff < 0 and uniformity_diff < 0

    if strong_exp_advantage:
        conclusion = f"结论: 实验组更优，阴影团聚低 {shadow_diff:.1f} 分，均匀度高 {uniformity_diff:.1f} 分。"
    elif trend_exp_advantage:
        conclusion = "结论: 实验组趋势更优，阴影团聚更轻且均匀度更高，但统计证据仍偏弱。"
    elif strong_base_advantage:
        conclusion = f"结论: base组更优，阴影团聚低 {abs(shadow_diff):.1f} 分，均匀度高 {abs(uniformity_diff):.1f} 分。"
    elif trend_base_advantage:
        conclusion = "结论: base组趋势更优，阴影团聚更轻且均匀度更高，但统计证据仍偏弱。"
    else:
        conclusion = "结论: 两组在阴影团聚与均匀度上接近，当前更适合解读为趋势对比。"

    lines: List[str] = []
    if note:
        lines.extend([note, ""])

    lines.extend([
        self._format_compare_fixed_params(compare_context),
        "",
        "CNT数量",
        (
            f"base: 均值{base_count['mean']:.1f}±{base_count['std']:.1f} | "
            f"总计{int(round(base_count['total']))} | n={base_group['image_count']}"
        ),
        (
            f"实验: 均值{exp_count['mean']:.1f}±{exp_count['std']:.1f} | "
            f"总计{int(round(exp_count['total']))} | n={exp_group['image_count']} | "
            f"{self._format_compact_significance(count_tests)}"
        ),
        "",
        "分散 / 团聚统计",
    ])
    lines.extend(self._format_dispersion_summary_lines(
        'base',
        base_count['mean'],
        base_dispersed['mean'],
        base_agglomerated['mean'],
        base_ratio['mean'],
        base_dispersed_length['mean'],
        base_spatial['uniformity_score']['mean'],
        length_std=base_dispersed_length['std'],
    ))
    lines.extend(self._format_dispersion_summary_lines(
        '实验',
        exp_count['mean'],
        exp_dispersed['mean'],
        exp_agglomerated['mean'],
        exp_ratio['mean'],
        exp_dispersed_length['mean'],
        exp_spatial['uniformity_score']['mean'],
        length_std=exp_dispersed_length['std'],
    ))
    lines.extend([
        "",
        "长粗管对比",
        self._format_metric_comparison(
            '长粗管数量',
            'base',
            base_long_thick_count['mean'],
            '实验',
            exp_long_thick_count['mean'],
            direction='up',
            qualifier='每图均值，长度≥40μm且宽度≥1.0μm',
            precision=1,
            left_std=base_long_thick_count['std'],
            right_std=exp_long_thick_count['std'],
            test_result=long_thick_count_tests,
        ),
        self._format_metric_comparison(
            '长粗管占比',
            'base',
            base_long_thick_ratio['mean'] * 100.0,
            '实验',
            exp_long_thick_ratio['mean'] * 100.0,
            direction='up',
            qualifier='全部 CNT 中占比(%)',
            precision=1,
            left_std=base_long_thick_ratio['std'] * 100.0,
            right_std=exp_long_thick_ratio['std'] * 100.0,
            test_result=long_thick_ratio_tests,
        ),
        "",
        "混合评分分项",
    ])
    lines.extend(self._format_hybrid_score_comparison(
        'base',
        base_framework,
        '实验',
        exp_framework,
        left_std=(base_group.get('hybrid_score_stats') or {}).get('std'),
        right_std=(exp_group.get('hybrid_score_stats') or {}).get('std'),
        test_result=hybrid_tests,
    ))
    lines.extend([
        "",
        "核心指标",
        self._format_metric_comparison(
            '阴影团聚',
            'base',
            base_spatial['shadow_aggregation_score']['mean'],
            '实验',
            exp_spatial['shadow_aggregation_score']['mean'],
            direction='down',
            qualifier='0-100，越低越好',
            precision=1,
            left_std=base_spatial['shadow_aggregation_score']['std'],
            right_std=exp_spatial['shadow_aggregation_score']['std'],
            test_result=shadow_tests,
        ),
        self._format_metric_comparison(
            '均匀度',
            'base',
            base_spatial['uniformity_score']['mean'],
            '实验',
            exp_spatial['uniformity_score']['mean'],
            direction='up',
            qualifier='0-100，越高越好',
            precision=1,
            left_std=base_spatial['uniformity_score']['std'],
            right_std=exp_spatial['uniformity_score']['std'],
            test_result=uniformity_tests,
        ),
        "",
        conclusion,
        "",
    ])

    lines.extend(self._format_group_detail_lines(base_group))
    lines.append("")
    lines.extend(self._format_group_detail_lines(exp_group))

    if failures:
        lines.extend(["", "未成功分析的文件:"])
        lines.extend([f"  - {item}" for item in failures])

    return "\n".join(lines)


def _format_comparison_summary(
    self,
    left_result: dict,
    right_result: dict,
    left_label: str,
    right_label: str,
    note: Optional[str] = None,
) -> str:
    """生成精简版双图对比摘要。"""
    compare_context = self._get_compare_display_context(left_result, right_result)
    left_spatial = left_result['stats'].get('spatial_distribution') or {}
    right_spatial = right_result['stats'].get('spatial_distribution') or {}
    left_metrics = self._get_shadow_aggregation_metrics(left_spatial)
    right_metrics = self._get_shadow_aggregation_metrics(right_spatial)
    left_count = int(left_result['stats'].get('count', 0))
    right_count = int(right_result['stats'].get('count', 0))
    left_dispersed = left_result.get('dispersed_stats') or {}
    right_dispersed = right_result.get('dispersed_stats') or {}
    left_framework = self._get_result_evaluation_framework(left_result)
    right_framework = self._get_result_evaluation_framework(right_result)
    count_diff = left_count - right_count
    count_ratio = (count_diff / right_count * 100.0) if right_count > 0 else 0.0
    left_uniformity_score = left_metrics['uniformity_score']
    right_uniformity_score = right_metrics['uniformity_score']
    uniformity_diff = left_uniformity_score - right_uniformity_score
    shadow_diff = right_metrics['score'] - left_metrics['score']

    left_better = shadow_diff > 0 and uniformity_diff > 0
    right_better = shadow_diff < 0 and uniformity_diff < 0

    if left_better and shadow_diff >= 4.0 and uniformity_diff >= 4.0:
        conclusion = f"结论: {left_label}更优，阴影团聚低 {shadow_diff:.1f} 分，均匀度高 {uniformity_diff:.1f} 分。"
    elif right_better and abs(shadow_diff) >= 4.0 and abs(uniformity_diff) >= 4.0:
        conclusion = f"结论: {right_label}更优，阴影团聚低 {abs(shadow_diff):.1f} 分，均匀度高 {abs(uniformity_diff):.1f} 分。"
    elif left_better:
        conclusion = f"结论: {left_label}趋势更优，阴影团聚更轻且均匀度更高。"
    elif right_better:
        conclusion = f"结论: {right_label}趋势更优，阴影团聚更轻且均匀度更高。"
    else:
        conclusion = "结论: 两张图在阴影团聚与均匀度上接近，当前更适合看作趋势对比。"

    left_dispersed_length = (left_dispersed.get('dispersed_length_stats') or {}).get('length_mean', left_result['stats'].get('length_mean', 0.0))
    right_dispersed_length = (right_dispersed.get('dispersed_length_stats') or {}).get('length_mean', right_result['stats'].get('length_mean', 0.0))

    lines: List[str] = []
    if note:
        lines.extend([note, ""])

    lines.extend([
        self._format_compare_fixed_params(compare_context),
        "",
        "CNT数量",
        f"{left_label}: {left_result['name']} | CNT={left_count}",
        f"{right_label}: {right_result['name']} | CNT={right_count}",
        f"差异: {left_label if count_diff >= 0 else right_label} {abs(count_ratio):.1f}%",
        "",
        "分散 / 团聚统计",
    ])
    lines.extend(self._format_dispersion_summary_lines(
        left_label,
        left_count,
        float(left_dispersed.get('dispersed_count', left_count)),
        float(left_dispersed.get('agglomerated_count', 0.0)),
        float(left_dispersed.get('dispersed_ratio', 1.0 if left_count > 0 else 0.0)),
        float(left_dispersed_length),
        left_uniformity_score,
    ))
    lines.extend(self._format_dispersion_summary_lines(
        right_label,
        right_count,
        float(right_dispersed.get('dispersed_count', right_count)),
        float(right_dispersed.get('agglomerated_count', 0.0)),
        float(right_dispersed.get('dispersed_ratio', 1.0 if right_count > 0 else 0.0)),
        float(right_dispersed_length),
        right_uniformity_score,
    ))
    lines.extend([
        "",
        "核心指标",
        self._format_metric_comparison(
            '阴影团聚',
            left_label,
            left_metrics['score'],
            right_label,
            right_metrics['score'],
            direction='down',
            qualifier='0-100，越低越好',
            precision=1,
        ),
        self._format_metric_comparison(
            '均匀度',
            left_label,
            left_uniformity_score,
            right_label,
            right_uniformity_score,
            direction='up',
            qualifier='0-100，越高越好',
            precision=1,
        ),
        "",
        conclusion,
    ])

    return "\n".join(lines)


def _render_dispersion_comparison_dashboard(
    self,
    left_group: dict,
    right_group: dict,
    summary_text: str,
    tests: Optional[dict] = None,
) -> None:
    """统一渲染双图/组别的分散对比面板。"""
    left_typical = self._select_representative_result(left_group)
    right_typical = self._select_representative_result(right_group)

    left_display_image = self._prepare_comparison_display_image(left_typical.get('visualization'), max_width=1100, max_height=650)
    right_display_image = self._prepare_comparison_display_image(right_typical.get('visualization'), max_width=1100, max_height=650)
    stack_images = self._should_stack_comparison_images(left_display_image, right_display_image, threshold=1.65)
    layout = self._build_comparison_layout(left_display_image, right_display_image, stacked=stack_images, variant='pair')
    figure_width, figure_height = layout['figsize']
    figure = Figure(figsize=(min(13.4, figure_width), min(11.2, figure_height)), dpi=getattr(self, '_chart_dpi', 100))
    figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])

    if stack_images:
        grid_spec = figure.add_gridspec(
            4,
            3,
            height_ratios=[layout['height_ratios'][0], 0.72, layout['height_ratios'][1], layout['height_ratios'][2]],
            hspace=layout['hspace'],
            wspace=layout['wspace'],
        )
        ax_dispersed_count = figure.add_subplot(grid_spec[0, 0])
        ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 1])
        ax_long_thick_ratio = figure.add_subplot(grid_spec[0, 2])
        ax_hybrid = figure.add_subplot(grid_spec[1, :])
        ax_left = figure.add_subplot(grid_spec[2, :])
        ax_right = figure.add_subplot(grid_spec[3, :])
    else:
        grid_spec = figure.add_gridspec(
            3,
            3,
            height_ratios=[layout['height_ratios'][0], 0.78, layout['height_ratios'][1]],
            hspace=layout['hspace'],
            wspace=layout['wspace'],
        )
        ax_dispersed_count = figure.add_subplot(grid_spec[0, 0])
        ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 1])
        ax_long_thick_ratio = figure.add_subplot(grid_spec[0, 2])
        ax_hybrid = figure.add_subplot(grid_spec[1, :])
        ax_left = figure.add_subplot(grid_spec[2, 0])
        ax_right = figure.add_subplot(grid_spec[2, 2])
    figure.subplots_adjust(**layout['adjust'])

    dispersed_count_test = tests.get('dispersed_count') if tests else None
    dispersed_ratio_test = tests.get('dispersed_ratio') if tests else None
    long_thick_ratio_test = tests.get('long_thick_ratio') if tests else None
    hybrid_score_test = tests.get('hybrid_score') if tests else None

    self._plot_dispersion_bar_chart(
        ax_dispersed_count,
        left_group,
        right_group,
        'dispersed_count_stats',
        '分散CNT数量对比',
        '数量',
        "{:.1f}",
        test_result=dispersed_count_test,
    )
    self._plot_dispersion_bar_chart(
        ax_dispersed_ratio,
        left_group,
        right_group,
        'dispersed_ratio_stats',
        '分散比例对比',
        '比例 (%)',
        "{:.1f}%",
        scale=100.0,
        test_result=dispersed_ratio_test,
    )
    self._plot_dispersion_bar_chart(
        ax_long_thick_ratio,
        left_group,
        right_group,
        'long_thick_ratio_stats',
        '长粗管占比对比',
        '比例 (%)',
        "{:.1f}%",
        scale=100.0,
        test_result=long_thick_ratio_test,
    )
    self._plot_dispersion_bar_chart(
        ax_hybrid,
        left_group,
        right_group,
        'hybrid_score_stats',
        '混合评分对比',
        '评分 (0-100)',
        "{:.1f}",
        test_result=hybrid_score_test,
    )

    self._configure_comparison_image_axis(
        ax_left,
        left_display_image,
        self._format_representative_caption(left_group, left_typical),
        spatial=left_typical.get('stats', {}).get('spatial_distribution'),
    )
    self._configure_comparison_image_axis(
        ax_right,
        right_display_image,
        self._format_representative_caption(right_group, right_typical),
        spatial=right_typical.get('stats', {}).get('spatial_distribution'),
    )
    self._render_comparison_figure(summary_text, figure)


def _show_comparison_window(
    self,
    left_result: dict,
    right_result: dict,
    left_label: str,
    right_label: str,
    note: Optional[str] = None,
):
    """将双图对比结果显示到对比分析面板。"""
    left_result = self._ensure_result_visualization(left_result)
    right_result = self._ensure_result_visualization(right_result)
    summary_text = self._format_comparison_summary(left_result, right_result, left_label, right_label, note)
    left_group = self._summarize_group_results(left_label, [left_result])
    right_group = self._summarize_group_results(right_label, [right_result])
    tests = self._compute_group_comparison_tests(left_group, right_group)
    self._render_dispersion_comparison_dashboard(left_group, right_group, summary_text, tests)


def _show_group_comparison_window(
    self,
    base_group: dict,
    exp_group: dict,
    note: Optional[str] = None,
    failures: Optional[List[str]] = None,
):
    """将 base 组与实验组的多图对比结果显示到对比分析面板。"""
    tests = self._compute_group_comparison_tests(base_group, exp_group)
    summary_text = self._format_group_comparison_summary(base_group, exp_group, note, failures)
    self._render_dispersion_comparison_dashboard(base_group, exp_group, summary_text, tests)


_COMPARISON_VIEW_METHODS = {
    '_fit_comparison_figure_to_frame': _fit_comparison_figure_to_frame,
    '_schedule_comparison_layout_refresh': _schedule_comparison_layout_refresh,
    '_refresh_comparison_layout': _refresh_comparison_layout,
    '_render_comparison_figure': _render_comparison_figure,
    '_should_stack_comparison_images': _should_stack_comparison_images,
    '_build_comparison_layout': _build_comparison_layout,
    '_prepare_comparison_display_image': _prepare_comparison_display_image,
    '_configure_comparison_image_axis': _configure_comparison_image_axis,
    '_estimate_comparison_summary_height': _estimate_comparison_summary_height,
    '_format_representative_caption': _format_representative_caption,
    '_format_group_comparison_summary': _format_group_comparison_summary,
    '_format_comparison_summary': _format_comparison_summary,
    '_render_dispersion_comparison_dashboard': _render_dispersion_comparison_dashboard,
    '_show_comparison_window': _show_comparison_window,
    '_show_group_comparison_window': _show_group_comparison_window,
}


def bind_comparison_view_helpers(gui) -> None:
    """Attach extracted comparison view helpers to an existing GUI controller instance."""
    for name, func in _COMPARISON_VIEW_METHODS.items():
        setattr(gui, name, MethodType(func, gui))
