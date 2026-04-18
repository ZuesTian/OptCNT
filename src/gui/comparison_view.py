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
                'figsize': (13.4, 12.2),
                'height_ratios': [0.82, 0.82, 1.55, 1.55],
                'hspace': 0.38,
                'wspace': 0.25,
                'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.05},
            }
        return {
            'figsize': (13.2, 10.4),
            'height_ratios': [0.84, 0.84, 1.45],
            'hspace': 0.32,
            'wspace': 0.25,
            'adjust': {'left': 0.06, 'right': 0.97, 'top': 0.96, 'bottom': 0.06},
        }

    if stacked:
        return {
            'figsize': (14.5, 14.4),
            'height_ratios': [0.84, 0.84, 1.48, 1.48],
            'hspace': 0.40,
            'wspace': 0.32,
            'adjust': {'left': 0.055, 'right': 0.97, 'top': 0.97, 'bottom': 0.04},
        }
    return {
        'figsize': (14.5, 12.4),
        'height_ratios': [0.90, 0.84, 1.52],
        'hspace': 0.34,
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
    """生成代表图标题，统一到五指标口径。"""
    framework = self._get_result_evaluation_framework(representative_result)
    core_metrics = self._build_core_metric_snapshot(
        representative_result.get('stats'),
        representative_result.get('dispersed_stats'),
        framework,
    )
    return (
        f"{group_summary.get('label', '对象')}典型结果\n"
        f"{representative_result.get('name', '未命名图像')}\n"
        f"总CNT={core_metrics['total_count']} | 分散比例={core_metrics['dispersed_ratio']:.1%} | "
        f"网格CV={core_metrics['grid_density_cv']:.2f}"
    )


def _extract_group_core_metrics(self, group_summary: dict) -> dict:
    """从组汇总中提取统一的五指标均值。"""
    return {
        'total_count': float((group_summary.get('count_stats') or {}).get('mean', 0.0) or 0.0),
        'dispersed_count': float((group_summary.get('dispersed_count_stats') or {}).get('mean', 0.0) or 0.0),
        'agglomerated_count': float((group_summary.get('agglomerated_count_stats') or {}).get('mean', 0.0) or 0.0),
        'dispersed_ratio': float((group_summary.get('dispersed_ratio_stats') or {}).get('mean', 0.0) or 0.0),
        'grid_density_cv': float((group_summary.get('grid_density_cv_stats') or {}).get('mean', 0.0) or 0.0),
        'agglomerated_area_ratio': float((group_summary.get('agglomerated_area_ratio_stats') or {}).get('mean', 0.0) or 0.0),
        'width_p90_um': float((group_summary.get('width_p90_um_stats') or {}).get('mean', 0.0) or 0.0),
        'uniformity_score': float((group_summary.get('uniformity_score_stats') or {}).get('mean', 0.0) or 0.0),
        'skeleton_length_mean_um': float((group_summary.get('skeleton_length_mean_stats') or {}).get('mean', 0.0) or 0.0),
    }


def _build_core_metric_conclusion(
    self,
    left_label: str,
    left_metrics: dict,
    right_label: str,
    right_metrics: dict,
    tests: Optional[dict] = None,
) -> str:
    """根据五指标生成结论，明确区分数量维度与空间维度。"""
    ratio_gap = float(left_metrics.get('dispersed_ratio', 0.0)) - float(right_metrics.get('dispersed_ratio', 0.0))
    grid_gap = float(left_metrics.get('grid_density_cv', 0.0)) - float(right_metrics.get('grid_density_cv', 0.0))
    area_gap = float(left_metrics.get('agglomerated_area_ratio', 0.0)) - float(right_metrics.get('agglomerated_area_ratio', 0.0))
    width_gap = float(left_metrics.get('width_p90_um', 0.0)) - float(right_metrics.get('width_p90_um', 0.0))

    left_spatial_better = (
        (grid_gap <= -0.05 and area_gap <= -0.03) or
        grid_gap <= -0.10 or
        area_gap <= -0.05
    )
    right_spatial_better = (
        (grid_gap >= 0.05 and area_gap >= 0.03) or
        grid_gap >= 0.10 or
        area_gap >= 0.05
    )
    left_dispersion_better = ratio_gap >= 0.05
    right_dispersion_better = ratio_gap <= -0.05
    left_bundle_better = width_gap <= -0.12
    right_bundle_better = width_gap >= 0.12
    significant_spatial = bool(tests) and (
        self._is_test_significant((tests or {}).get('grid_density_cv')) or
        self._is_test_significant((tests or {}).get('agglomerated_area_ratio'))
    )

    if left_spatial_better:
        conclusion = f"结论: {left_label}在空间维度更优，网格CV更低且团聚面积占比更小。"
        if left_dispersion_better:
            conclusion += f" 同时 {left_label} 的分散比例也更高。"
        if left_bundle_better:
            conclusion += " P90宽度更低，束化尾部更轻。"
        if tests and not significant_spatial:
            conclusion += " 当前更适合作为趋势判断。"
        return conclusion

    if right_spatial_better:
        conclusion = f"结论: {right_label}在空间维度更优，网格CV更低且团聚面积占比更小。"
        if right_dispersion_better:
            conclusion += f" 同时 {right_label} 的分散比例也更高。"
        if right_bundle_better:
            conclusion += " P90宽度更低，束化尾部更轻。"
        if tests and not significant_spatial:
            conclusion += " 当前更适合作为趋势判断。"
        return conclusion

    if left_dispersion_better:
        return f"结论: {left_label}分散比例更高，但空间维度差异有限；数量多，不等于分布均匀。"
    if right_dispersion_better:
        return f"结论: {right_label}分散比例更高，但空间维度差异有限；数量多，不等于分布均匀。"
    if left_bundle_better:
        return f"结论: 两者空间维度接近，但 {left_label} 的P90宽度更低，束化尾部略轻。"
    if right_bundle_better:
        return f"结论: 两者空间维度接近，但 {right_label} 的P90宽度更低，束化尾部略轻。"
    return "结论: 两者在五个固定指标上接近，建议继续以网格CV和团聚面积占比作为主判断。"


def _format_group_comparison_summary(
    self,
    base_group: dict,
    exp_group: dict,
    note: Optional[str] = None,
    failures: Optional[List[str]] = None,
) -> str:
    """生成按五指标统一口径的组别对比摘要。"""
    base_count = base_group['count_stats']
    exp_count = exp_group['count_stats']
    base_core = self._extract_group_core_metrics(base_group)
    exp_core = self._extract_group_core_metrics(exp_group)
    compare_context = self._get_compare_display_context(
        *(base_group.get('results') or []),
        *(exp_group.get('results') or []),
    )
    tests = self._compute_group_comparison_tests(base_group, exp_group)
    conclusion = self._build_core_metric_conclusion('base组', base_core, '实验组', exp_core, tests=tests)

    lines: List[str] = []
    if note:
        lines.extend([note, ""])

    lines.extend([
        self._format_compare_fixed_params(compare_context),
        "",
        "固定输出（5项）",
        self._format_metric_comparison(
            '总CNT数量',
            'base',
            base_count['mean'],
            '实验',
            exp_count['mean'],
            direction='up',
            qualifier='每图均值',
            precision=1,
            left_std=base_count['std'],
            right_std=exp_count['std'],
            test_result=tests.get('count'),
        ),
        self._format_metric_comparison(
            '分散比例',
            'base',
            base_core['dispersed_ratio'] * 100.0,
            '实验',
            exp_core['dispersed_ratio'] * 100.0,
            direction='up',
            qualifier='数量维度(%)',
            precision=1,
            left_std=(base_group.get('dispersed_ratio_stats') or {}).get('std', 0.0) * 100.0,
            right_std=(exp_group.get('dispersed_ratio_stats') or {}).get('std', 0.0) * 100.0,
            test_result=tests.get('dispersed_ratio'),
        ),
        self._format_metric_comparison(
            '网格CV',
            'base',
            base_core['grid_density_cv'],
            '实验',
            exp_core['grid_density_cv'],
            direction='down',
            qualifier='空间维度，越小越均匀',
            precision=2,
            left_std=(base_group.get('grid_density_cv_stats') or {}).get('std'),
            right_std=(exp_group.get('grid_density_cv_stats') or {}).get('std'),
            test_result=tests.get('grid_density_cv'),
        ),
        self._format_metric_comparison(
            '团聚面积占比',
            'base',
            base_core['agglomerated_area_ratio'] * 100.0,
            '实验',
            exp_core['agglomerated_area_ratio'] * 100.0,
            direction='down',
            qualifier='空间维度(%)',
            precision=1,
            left_std=(base_group.get('agglomerated_area_ratio_stats') or {}).get('std', 0.0) * 100.0,
            right_std=(exp_group.get('agglomerated_area_ratio_stats') or {}).get('std', 0.0) * 100.0,
            test_result=tests.get('agglomerated_area_ratio'),
        ),
        self._format_metric_comparison(
            'P90宽度',
            'base',
            base_core['width_p90_um'],
            '实验',
            exp_core['width_p90_um'],
            direction='down',
            qualifier='束化尾部指标(μm)',
            precision=2,
            left_std=(base_group.get('width_p90_um_stats') or {}).get('std'),
            right_std=(exp_group.get('width_p90_um_stats') or {}).get('std'),
            test_result=tests.get('width_p90_um'),
        ),
        "",
        "数量与空间分开解读",
        "数量多，不等于分布均匀；因此分别用分散比例评价分散程度，用网格CV评价空间均匀性。",
    ])
    lines.extend(self._format_dispersion_summary_lines(
        'base',
        base_core['total_count'],
        base_core['dispersed_count'],
        base_core['agglomerated_count'],
        base_core['dispersed_ratio'],
        base_core['grid_density_cv'],
        base_core['agglomerated_area_ratio'],
        base_core['width_p90_um'],
        skeleton_length_mean_um=base_core['skeleton_length_mean_um'],
        uniformity_score=base_core['uniformity_score'],
    ))
    lines.extend(self._format_dispersion_summary_lines(
        '实验',
        exp_core['total_count'],
        exp_core['dispersed_count'],
        exp_core['agglomerated_count'],
        exp_core['dispersed_ratio'],
        exp_core['grid_density_cv'],
        exp_core['agglomerated_area_ratio'],
        exp_core['width_p90_um'],
        skeleton_length_mean_um=exp_core['skeleton_length_mean_um'],
        uniformity_score=exp_core['uniformity_score'],
    ))
    lines.extend([
        "",
        "补充指标",
        self._format_metric_comparison(
            '平均骨架长度',
            'base',
            base_core['skeleton_length_mean_um'],
            '实验',
            exp_core['skeleton_length_mean_um'],
            direction='up',
            qualifier='补充长度指标(μm)',
            precision=1,
            left_std=(base_group.get('skeleton_length_mean_stats') or {}).get('std'),
            right_std=(exp_group.get('skeleton_length_mean_stats') or {}).get('std'),
            test_result=tests.get('skeleton_length_mean_um'),
        ),
        self._format_metric_comparison(
            '综合均匀性得分',
            'base',
            base_core['uniformity_score'],
            '实验',
            exp_core['uniformity_score'],
            direction='up',
            qualifier='仅展示层，不作为主结论',
            precision=1,
            left_std=(base_group.get('uniformity_score_stats') or {}).get('std'),
            right_std=(exp_group.get('uniformity_score_stats') or {}).get('std'),
            test_result=tests.get('uniformity_score'),
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
    """生成按五指标统一口径的双图对比摘要。"""
    compare_context = self._get_compare_display_context(left_result, right_result)
    left_framework = self._get_result_evaluation_framework(left_result)
    right_framework = self._get_result_evaluation_framework(right_result)
    left_core = self._build_core_metric_snapshot(left_result.get('stats'), left_result.get('dispersed_stats'), left_framework)
    right_core = self._build_core_metric_snapshot(right_result.get('stats'), right_result.get('dispersed_stats'), right_framework)
    conclusion = self._build_core_metric_conclusion(left_label, left_core, right_label, right_core)

    lines: List[str] = []
    if note:
        lines.extend([note, ""])

    lines.extend([
        self._format_compare_fixed_params(compare_context),
        "",
        "固定输出（5项）",
        self._format_metric_comparison(
            '总CNT数量',
            left_label,
            left_core['total_count'],
            right_label,
            right_core['total_count'],
            direction='up',
            qualifier='绝对数量',
            precision=0,
        ),
        self._format_metric_comparison(
            '分散比例',
            left_label,
            left_core['dispersed_ratio'] * 100.0,
            right_label,
            right_core['dispersed_ratio'] * 100.0,
            direction='up',
            qualifier='数量维度(%)',
            precision=1,
        ),
        self._format_metric_comparison(
            '网格CV',
            left_label,
            left_core['grid_density_cv'],
            right_label,
            right_core['grid_density_cv'],
            direction='down',
            qualifier='空间维度，越小越均匀',
            precision=2,
        ),
        self._format_metric_comparison(
            '团聚面积占比',
            left_label,
            left_core['agglomerated_area_ratio'] * 100.0,
            right_label,
            right_core['agglomerated_area_ratio'] * 100.0,
            direction='down',
            qualifier='空间维度(%)',
            precision=1,
        ),
        self._format_metric_comparison(
            'P90宽度',
            left_label,
            left_core['width_p90_um'],
            right_label,
            right_core['width_p90_um'],
            direction='down',
            qualifier='束化尾部指标(μm)',
            precision=2,
        ),
        "",
        "数量与空间分开解读",
        "数量多，不等于分布均匀；因此分别用分散比例评价分散程度，用网格CV评价空间均匀性。",
    ])
    lines.extend(self._format_dispersion_summary_lines(
        left_label,
        left_core['total_count'],
        left_core['dispersed_count'],
        left_core['agglomerated_count'],
        left_core['dispersed_ratio'],
        left_core['grid_density_cv'],
        left_core['agglomerated_area_ratio'],
        left_core['width_p90_um'],
        skeleton_length_mean_um=left_core['skeleton_length_mean_um'],
        uniformity_score=left_core['uniformity_score'],
    ))
    lines.extend(self._format_dispersion_summary_lines(
        right_label,
        right_core['total_count'],
        right_core['dispersed_count'],
        right_core['agglomerated_count'],
        right_core['dispersed_ratio'],
        right_core['grid_density_cv'],
        right_core['agglomerated_area_ratio'],
        right_core['width_p90_um'],
        skeleton_length_mean_um=right_core['skeleton_length_mean_um'],
        uniformity_score=right_core['uniformity_score'],
    ))
    lines.extend([
        "",
        "补充指标",
        self._format_metric_comparison(
            '平均骨架长度',
            left_label,
            left_core['skeleton_length_mean_um'],
            right_label,
            right_core['skeleton_length_mean_um'],
            direction='up',
            qualifier='补充长度指标(μm)',
            precision=1,
        ),
        self._format_metric_comparison(
            '综合均匀性得分',
            left_label,
            left_core['uniformity_score'],
            right_label,
            right_core['uniformity_score'],
            direction='up',
            qualifier='仅展示层，不作为主结论',
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
    variant = 'group' if max(left_group.get('image_count', 1), right_group.get('image_count', 1)) > 1 else 'pair'

    left_display_image = self._prepare_comparison_display_image(left_typical.get('visualization'), max_width=1100, max_height=650)
    right_display_image = self._prepare_comparison_display_image(right_typical.get('visualization'), max_width=1100, max_height=650)
    stack_images = self._should_stack_comparison_images(left_display_image, right_display_image, threshold=1.65)
    layout = self._build_comparison_layout(left_display_image, right_display_image, stacked=stack_images, variant=variant)
    figure_width, figure_height = layout['figsize']
    figure = Figure(figsize=(min(14.5, figure_width), min(12.8, figure_height)), dpi=getattr(self, '_chart_dpi', 100))
    figure.patch.set_facecolor(self.MODERN_COLORS['bg_secondary'])

    if stack_images:
        grid_spec = figure.add_gridspec(
            4,
            6,
            height_ratios=layout['height_ratios'],
            hspace=layout['hspace'],
            wspace=layout['wspace'],
        )
        ax_total_count = figure.add_subplot(grid_spec[0, 0:2])
        ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 2:4])
        ax_grid_cv = figure.add_subplot(grid_spec[0, 4:6])
        ax_agglomerated_area = figure.add_subplot(grid_spec[1, 0:3])
        ax_width_p90 = figure.add_subplot(grid_spec[1, 3:6])
        ax_left = figure.add_subplot(grid_spec[2, :])
        ax_right = figure.add_subplot(grid_spec[3, :])
    else:
        grid_spec = figure.add_gridspec(
            3,
            6,
            height_ratios=layout['height_ratios'],
            hspace=layout['hspace'],
            wspace=layout['wspace'],
        )
        ax_total_count = figure.add_subplot(grid_spec[0, 0:2])
        ax_dispersed_ratio = figure.add_subplot(grid_spec[0, 2:4])
        ax_grid_cv = figure.add_subplot(grid_spec[0, 4:6])
        ax_agglomerated_area = figure.add_subplot(grid_spec[1, 0:3])
        ax_width_p90 = figure.add_subplot(grid_spec[1, 3:6])
        ax_left = figure.add_subplot(grid_spec[2, 0:3])
        ax_right = figure.add_subplot(grid_spec[2, 3:6])
    figure.subplots_adjust(**layout['adjust'])

    count_test = tests.get('count') if tests else None
    dispersed_ratio_test = tests.get('dispersed_ratio') if tests else None
    grid_density_test = tests.get('grid_density_cv') if tests else None
    agglomerated_area_test = tests.get('agglomerated_area_ratio') if tests else None
    width_p90_test = tests.get('width_p90_um') if tests else None

    self._plot_dispersion_bar_chart(
        ax_total_count,
        left_group,
        right_group,
        'count_stats',
        '总CNT数量对比',
        '数量',
        "{:.1f}",
        test_result=count_test,
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
        ax_grid_cv,
        left_group,
        right_group,
        'grid_density_cv_stats',
        '网格CV对比',
        'CV',
        "{:.2f}",
        test_result=grid_density_test,
    )
    self._plot_dispersion_bar_chart(
        ax_agglomerated_area,
        left_group,
        right_group,
        'agglomerated_area_ratio_stats',
        '团聚面积占比对比',
        '占比 (%)',
        "{:.1f}%",
        scale=100.0,
        test_result=agglomerated_area_test,
    )
    self._plot_dispersion_bar_chart(
        ax_width_p90,
        left_group,
        right_group,
        'width_p90_um_stats',
        'P90宽度对比',
        '宽度 (μm)',
        "{:.2f}",
        test_result=width_p90_test,
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
    '_extract_group_core_metrics': _extract_group_core_metrics,
    '_build_core_metric_conclusion': _build_core_metric_conclusion,
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
