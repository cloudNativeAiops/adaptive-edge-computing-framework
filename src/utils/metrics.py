import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from src.docker_manager import DockerManager

@dataclass
class InferenceMetrics:
    inference_time: float
    memory_usage: float
    cpu_usage: float
    model_name: str
    batch_size: int
    device: str
    accuracy: float = 0.0          # Top-1 accuracy
    top5_accuracy: float = 0.0     # Top-5 accuracy
    predictions: List[int] = None  # predicted results
    ground_truth: List[int] = None # ground truth

@dataclass
class PartitionMetrics:
    partition_points: List[int]          # 模型分区点
    partition_sizes: List[float]         # 每个分区大小(MB)
    partition_compute_costs: List[float]  # 每个分区的计算成本
    communication_costs: float           # 分区间通信成本
    partition_latencies: List[float]     # 每个分区的延迟
    total_transfer_size: float          # 分区间传输的总数据量

@dataclass
class SchedulingMetrics:
    task_distribution: Dict[str, int]    # 每个节点的任务分配数
    queue_lengths: List[int]             # 任务队列长度历史
    scheduling_overhead: float           # 调度决策时间
    load_balancing_score: float         # 负载均衡评分
    resource_utilization: Dict[str, float] # 各资源利用率

@dataclass
class SystemMetrics:
    end_to_end_latency: float           # 端到端延迟
    throughput: float                    # 系统吞吐量
    network_bandwidth_usage: float       # 网络带宽使用
    energy_consumption: float            # 能源消耗
    system_stability_score: float        # 系统稳定性评分

class MetricsCollector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.docker_manager = DockerManager()  # 使用单例模式
            cls._instance.performance_history = {}
            cls._instance.resource_history = {}
        return cls._instance
    
    def __init__(self):
        # 确保不重复初始化
        pass
        
    def collect_comprehensive_metrics(self):
        """收集综合性能指标"""
        try:
            return {
                'resource_metrics': self.collect_resource_metrics(self.docker_manager._containers),
                'model_metrics': {
                    'accuracy': self.performance_history.get('accuracy', 0.0),
                    'inference_time': self.performance_history.get('inference_time', 0.0),
                    'memory_usage': self.performance_history.get('memory_usage', 0.0),
                },
                'system_metrics': {
                    'end_to_end_latency': self.performance_history.get('latency', 0.0),
                    'throughput': self.performance_history.get('throughput', 0.0),
                    'network_bandwidth': self.performance_history.get('bandwidth', 0.0),
                },
                'scheduling_metrics': {
                    'load_balancing': self.performance_history.get('load_balancing', 0.0),
                    'resource_utilization': self.performance_history.get('utilization', 0.0),
                }
            }
        except Exception as e:
            print(f"Error collecting comprehensive metrics: {e}")
            return {}
            
    def clear(self):
        """清理历史数据"""
        self.performance_history.clear()
        self.resource_history.clear()
        
    def collect_resource_metrics(self, containers):
        """改进 CPU 使用率计算"""
        if not containers:
            return {}
            
        resource_metrics = {}
        for node_id, container_id in containers.items():
            try:
                container = self.docker_manager.client.containers.get(container_id)
                stats = container.stats(stream=False)
                
                # CPU 使用率计算
                cpu_stats = stats.get('cpu_stats', {})
                precpu_stats = stats.get('precpu_stats', {})
                
                cpu_usage = 0.0
                try:
                    cpu_total = cpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                    precpu_total = precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                    system_usage = cpu_stats.get('system_cpu_usage', 0)
                    pre_system = precpu_stats.get('system_cpu_usage', 0)
                    
                    if system_usage and pre_system:
                        cpu_delta = cpu_total - precpu_total
                        system_delta = system_usage - pre_system
                        if system_delta > 0:
                            cpu_usage = (cpu_delta / system_delta) * 100.0
                except Exception as e:
                    print(f"Error calculating CPU usage for {node_id}: {e}")
                
                # 内存使用计算
                memory_stats = stats.get('memory_stats', {})
                memory_usage = memory_stats.get('usage', 0) / (1024 * 1024)  # 转换为 MB
                
                resource_metrics[node_id] = {
                    'cpu_usage_percent': round(cpu_usage, 2),
                    'memory_usage_mb': round(memory_usage, 2)
                }
                
            except Exception as e:
                print(f"Error collecting metrics for {node_id}: {e}")
                resource_metrics[node_id] = {
                    'cpu_usage_percent': 0.0,
                    'memory_usage_mb': 0.0,
                    'error': str(e)
                }
                
        return resource_metrics
