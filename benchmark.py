"""
基准测试工具模块
用于测试图像分析性能并生成基准报告
"""
import os
import time
from typing import Dict, List, Optional
import numpy as np

from analyzer_core import CNTAnalyzer
from profiler import PerformanceTimer, PerformanceProfiler


CONSOLE_REPLACEMENTS = str.maketrans({
    '✓': '[OK]',
    '✗': '[FAIL]',
    '±': '+/-',
    '×': 'x',
})


def _console_print(message: str = "") -> None:
    """Print text in a Windows-console-safe form."""
    print(str(message).translate(CONSOLE_REPLACEMENTS))


def benchmark_single_image(image_path: str,
                           blur_kernel: int = 9,
                           adaptive_block: int = 11,
                           adaptive_c: int = 3,
                           min_length_um: float = 0.5,
                           detection_profile: str = "balanced",
                           verbose: bool = True) -> Dict[str, float]:
    """
    对单张图像进行完整的性能基准测试

    Args:
        image_path: 图像文件路径
        blur_kernel: 高斯模糊核大小
        adaptive_block: 自适应阈值块大小
        adaptive_c: 自适应阈值常数
        min_length_um: 最小CNT长度（微米）
        detection_profile: 检测配置文件
        verbose: 是否打印详细信息

    Returns:
        Dict: 包含各阶段耗时的字典
    """
    if verbose:
        _console_print(f"\n{'='*60}")
        _console_print(f"基准测试: {os.path.basename(image_path)}")
        _console_print(f"{'='*60}")

    analyzer = CNTAnalyzer()
    profiler = PerformanceProfiler()
    results = {}

    # 1. 图像加载
    with PerformanceTimer("图像加载", log_result=False) as timer:
        analyzer.load_image(image_path)
    results['load_image'] = timer.get_elapsed_time()
    profiler.record('load_image', results['load_image'])

    # 2. 比例尺检测
    with PerformanceTimer("比例尺检测", log_result=False) as timer:
        analyzer.apply_detected_scale()
    results['scale_detection'] = timer.get_elapsed_time()
    profiler.record('scale_detection', results['scale_detection'])

    # 3. 图像预处理
    with PerformanceTimer("图像预处理", log_result=False) as timer:
        analyzer.preprocess(
            blur_kernel=blur_kernel,
            adaptive_block=adaptive_block,
            adaptive_c=adaptive_c
        )
    results['preprocess'] = timer.get_elapsed_time()
    profiler.record('preprocess', results['preprocess'])

    # 4. CNT检测
    with PerformanceTimer("CNT检测", log_result=False) as timer:
        measurements = analyzer.detect_cnts_hybrid(
            min_length_um=min_length_um,
            detection_profile=detection_profile
        )
    results['detect_cnts'] = timer.get_elapsed_time()
    profiler.record('detect_cnts', results['detect_cnts'])

    # 5. 空间分布分析
    with PerformanceTimer("空间分布分析", log_result=False) as timer:
        spatial_stats = analyzer.analyze_spatial_distribution()
    results['spatial_analysis'] = timer.get_elapsed_time()
    profiler.record('spatial_analysis', results['spatial_analysis'])

    # 6. 总耗时
    results['total'] = sum(results.values())
    results['cnt_count'] = len(measurements)

    if verbose:
        _console_print(f"\n检测到 {len(measurements)} 个CNT")
        _console_print(f"\n各阶段耗时:")
        _console_print(f"  图像加载:       {results['load_image']:.4f} 秒")
        _console_print(f"  比例尺检测:     {results['scale_detection']:.4f} 秒")
        _console_print(f"  图像预处理:     {results['preprocess']:.4f} 秒")
        _console_print(f"  CNT检测:        {results['detect_cnts']:.4f} 秒")
        _console_print(f"  空间分布分析:   {results['spatial_analysis']:.4f} 秒")
        _console_print(f"  总耗时:         {results['total']:.4f} 秒")
        _console_print(f"{'='*60}\n")

    return results


def benchmark_batch_images(image_paths: List[str],
                           blur_kernel: int = 9,
                           adaptive_block: int = 11,
                           adaptive_c: int = 3,
                           min_length_um: float = 0.5,
                           detection_profile: str = "balanced",
                           verbose: bool = True) -> Dict[str, any]:
    """
    对多张图像进行批量基准测试

    Args:
        image_paths: 图像文件路径列表
        blur_kernel: 高斯模糊核大小
        adaptive_block: 自适应阈值块大小
        adaptive_c: 自适应阈值常数
        min_length_um: 最小CNT长度（微米）
        detection_profile: 检测配置文件
        verbose: 是否打印详细信息

    Returns:
        Dict: 包含批量测试统计信息的字典
    """
    if verbose:
        _console_print(f"\n{'='*60}")
        _console_print(f"批量基准测试: {len(image_paths)} 张图像")
        _console_print(f"{'='*60}\n")

    all_results = []
    total_start = time.perf_counter()

    for i, image_path in enumerate(image_paths, 1):
        if verbose:
            _console_print(f"[{i}/{len(image_paths)}] 处理: {os.path.basename(image_path)}")

        try:
            result = benchmark_single_image(
                image_path,
                blur_kernel=blur_kernel,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                min_length_um=min_length_um,
                detection_profile=detection_profile,
                verbose=False
            )
            result['image_path'] = image_path
            result['success'] = True
            all_results.append(result)

            if verbose:
                _console_print(f"  [OK] 完成 - 耗时: {result['total']:.4f} 秒, CNT数量: {result['cnt_count']}")

        except Exception as e:
            if verbose:
                _console_print(f"  [FAIL] 失败: {str(e)}")
            all_results.append({
                'image_path': image_path,
                'success': False,
                'error': str(e)
            })

    total_elapsed = time.perf_counter() - total_start

    # 计算统计信息
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if successful_results:
        summary = {
            'total_images': len(image_paths),
            'successful_images': len(successful_results),
            'failed_images': len(image_paths) - len(successful_results),
            'total_time': total_elapsed,
            'avg_time_per_image': total_elapsed / len(image_paths),
            'stages': {}
        }

        # 计算各阶段平均耗时
        stages = ['load_image', 'scale_detection', 'preprocess', 'detect_cnts', 'spatial_analysis', 'total']
        for stage in stages:
            times = [r[stage] for r in successful_results if stage in r]
            if times:
                summary['stages'][stage] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }

        # 计算CNT数量统计
        cnt_counts = [r['cnt_count'] for r in successful_results if 'cnt_count' in r]
        if cnt_counts:
            summary['cnt_statistics'] = {
                'mean': np.mean(cnt_counts),
                'std': np.std(cnt_counts),
                'min': np.min(cnt_counts),
                'max': np.max(cnt_counts),
                'total': np.sum(cnt_counts)
            }

        if verbose:
            _console_print(f"\n{'='*60}")
            _console_print("批量测试摘要")
            _console_print(f"{'='*60}")
            _console_print(f"总图像数: {summary['total_images']}")
            _console_print(f"成功: {summary['successful_images']}, 失败: {summary['failed_images']}")
            _console_print(f"总耗时: {summary['total_time']:.2f} 秒")
            _console_print(f"平均每张: {summary['avg_time_per_image']:.4f} 秒")
            
            if 'cnt_statistics' in summary:
                _console_print(f"\nCNT统计:")
                _console_print(f"  总数: {summary['cnt_statistics']['total']:.0f}")
                _console_print(f"  平均: {summary['cnt_statistics']['mean']:.1f} +/- {summary['cnt_statistics']['std']:.1f}")
                _console_print(f"  范围: {summary['cnt_statistics']['min']:.0f} - {summary['cnt_statistics']['max']:.0f}")
            
            _console_print(f"\n各阶段平均耗时:")
            for stage, stats in summary['stages'].items():
                _console_print(f"  {stage:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f} 秒")
            _console_print(f"{'='*60}\n")

        return {
            'summary': summary,
            'results': all_results
        }
    else:
        return {
            'summary': {
                'total_images': len(image_paths),
                'successful_images': 0,
                'failed_images': len(image_paths),
                'total_time': total_elapsed
            },
            'results': all_results
        }


def find_test_images(data_dir: str = "Data", max_images: int = 5) -> List[str]:
    """
    在数据目录中查找测试图像

    Args:
        data_dir: 数据目录路径
        max_images: 最大图像数量

    Returns:
        List[str]: 图像文件路径列表
    """
    image_paths = []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    if os.path.exists(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, file))
                    if len(image_paths) >= max_images:
                        return image_paths

    return image_paths


if __name__ == "__main__":
    # 查找测试图像
    test_images = find_test_images(max_images=3)

    if not test_images:
        _console_print("未找到测试图像，请确保 Data 目录中有图像文件")
    else:
        _console_print(f"找到 {len(test_images)} 张测试图像")

        # 运行批量基准测试
        results = benchmark_batch_images(
            test_images,
            blur_kernel=9,
            adaptive_block=11,
            adaptive_c=3,
            min_length_um=0.5,
            detection_profile="balanced",
            verbose=True
        )
