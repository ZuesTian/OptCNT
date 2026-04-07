"""
性能分析工具模块
提供性能计时器和函数性能分析装饰器
"""
import time
import functools
from typing import Callable, Any, Dict
import logging

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, name: str = "Operation", log_result: bool = True):
        """
        初始化性能计时器
        
        Args:
            name: 操作名称
            log_result: 是否自动记录结果到日志
        """
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        """进入上下文时开始计时"""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时结束计时"""
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.log_result:
            logger.info(f"{self.name} 耗时: {self.elapsed_time:.4f} 秒")
        
        return False
    
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.perf_counter() - self.start_time
        return 0.0


def profile_function(func: Callable) -> Callable:
    """
    函数性能分析装饰器
    
    使用方法:
        @profile_function
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        timer = PerformanceTimer(f"{func.__name__}", log_result=True)
        with timer:
            result = func(*args, **kwargs)
        return result
    
    return wrapper


class PerformanceProfiler:
    """性能分析器，用于收集多个操作的性能数据"""
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
    
    def record(self, name: str, elapsed_time: float):
        """记录一次操作的耗时"""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed_time)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计摘要"""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                }
        return summary
    
    def print_summary(self):
        """打印性能统计摘要"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("性能分析摘要")
        print("="*60)
        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  调用次数: {stats['count']}")
            print(f"  总耗时: {stats['total']:.4f} 秒")
            print(f"  平均耗时: {stats['mean']:.4f} 秒")
            print(f"  最小耗时: {stats['min']:.4f} 秒")
            print(f"  最大耗时: {stats['max']:.4f} 秒")
        print("="*60 + "\n")
    
    def clear(self):
        """清空所有记录"""
        self.timings.clear()
