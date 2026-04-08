"""Lightweight statistical helpers used by the GUI comparison views.

These functions intentionally avoid heavyweight scientific dependencies so the
packaged application can stay much smaller.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, List, Tuple

import numpy as np


_NORMAL_DIST = NormalDist()
_BETA_FPMIN = 1e-300
_BETA_EPS = 3.0e-14
_BETA_MAX_ITER = 200


def _as_clean_array(values: Iterable[float]) -> np.ndarray:
    """Convert inputs to a finite 1-D float array."""
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - (qab * x / qap)
    if abs(d) < _BETA_FPMIN:
        d = _BETA_FPMIN
    d = 1.0 / d
    fraction = d

    for step in range(1, _BETA_MAX_ITER + 1):
        even = 2 * step

        numerator = step * (b - step) * x
        denominator = (qam + even) * (a + even)
        aa = numerator / denominator

        d = 1.0 + (aa * d)
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + (aa / c)
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d
        fraction *= d * c

        numerator = -(a + step) * (qab + step) * x
        denominator = (a + even) * (qap + even)
        aa = numerator / denominator

        d = 1.0 + (aa * d)
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + (aa / c)
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d

        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) < _BETA_EPS:
            break

    return fraction


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute the regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp((a * math.log(x)) + (b * math.log1p(-x)) + log_beta)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a

    return 1.0 - (front * _beta_continued_fraction(b, a, 1.0 - x) / b)


def _student_t_two_sided_pvalue(statistic: float, degrees_of_freedom: float) -> float:
    """Return a two-sided p-value for a Student/Welch t-statistic."""
    if not math.isfinite(statistic) or not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        return math.nan

    absolute_t = abs(float(statistic))
    if absolute_t == 0.0:
        return 1.0

    x = degrees_of_freedom / (degrees_of_freedom + absolute_t * absolute_t)
    pvalue = _regularized_incomplete_beta(0.5 * degrees_of_freedom, 0.5, x)
    return max(0.0, min(1.0, float(pvalue)))


def ttest_ind(
    first_sample: Iterable[float],
    second_sample: Iterable[float],
    *,
    equal_var: bool = False,
    nan_policy: str = "omit",
) -> Tuple[float, float]:
    """Lightweight Welch two-sample t-test compatible with the GUI needs."""
    if equal_var:
        raise ValueError("Only Welch's t-test is supported.")
    if nan_policy != "omit":
        raise ValueError("Only nan_policy='omit' is supported.")

    first = _as_clean_array(first_sample)
    second = _as_clean_array(second_sample)
    if first.size < 2 or second.size < 2:
        return math.nan, math.nan

    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    first_var = float(np.var(first, ddof=1))
    second_var = float(np.var(second, ddof=1))

    first_term = first_var / first.size
    second_term = second_var / second.size
    denominator = math.sqrt(first_term + second_term)
    if denominator == 0.0:
        return math.nan, math.nan

    statistic = (first_mean - second_mean) / denominator
    numerator = (first_term + second_term) ** 2
    denominator_df = 0.0
    if first.size > 1:
        denominator_df += (first_term * first_term) / (first.size - 1)
    if second.size > 1:
        denominator_df += (second_term * second_term) / (second.size - 1)
    if denominator_df == 0.0:
        return statistic, math.nan

    degrees_of_freedom = numerator / denominator_df
    return statistic, _student_t_two_sided_pvalue(statistic, degrees_of_freedom)


def _average_ranks(values: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Assign average ranks while tracking tie sizes for variance correction."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=float)
    tie_sizes: List[int] = []

    index = 0
    while index < sorted_values.size:
        tie_end = index + 1
        while tie_end < sorted_values.size and sorted_values[tie_end] == sorted_values[index]:
            tie_end += 1

        average_rank = ((index + 1) + tie_end) / 2.0
        ranks[order[index:tie_end]] = average_rank
        tie_sizes.append(tie_end - index)
        index = tie_end

    return ranks, tie_sizes


def mannwhitneyu(
    first_sample: Iterable[float],
    second_sample: Iterable[float],
    *,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Lightweight Mann-Whitney U test using a tie-corrected normal approximation."""
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(f"Unsupported alternative: {alternative}")

    first = _as_clean_array(first_sample)
    second = _as_clean_array(second_sample)
    if first.size == 0 or second.size == 0:
        return math.nan, math.nan

    combined = np.concatenate((first, second))
    ranks, tie_sizes = _average_ranks(combined)

    first_rank_sum = float(np.sum(ranks[: first.size]))
    u1 = first_rank_sum - (first.size * (first.size + 1) / 2.0)
    u2 = first.size * second.size - u1

    total_size = first.size + second.size
    tie_correction = float(sum((size ** 3) - size for size in tie_sizes if size > 1))
    variance = (first.size * second.size / 12.0) * (
        (total_size + 1.0) - (tie_correction / (total_size * max(total_size - 1, 1)))
    )
    if variance <= 0.0:
        return float(u1), math.nan

    mean_u = first.size * second.size / 2.0
    std_u = math.sqrt(variance)

    if alternative == "two-sided":
        chosen_u = min(u1, u2)
        distance = abs(chosen_u - mean_u)
        z_score = max(0.0, distance - 0.5) / std_u
        pvalue = 2.0 * (1.0 - _NORMAL_DIST.cdf(z_score))
    elif alternative == "greater":
        z_score = (u1 - mean_u - 0.5) / std_u
        pvalue = 1.0 - _NORMAL_DIST.cdf(z_score)
    else:
        z_score = (u1 - mean_u + 0.5) / std_u
        pvalue = _NORMAL_DIST.cdf(z_score)

    return float(u1), max(0.0, min(1.0, float(pvalue)))
