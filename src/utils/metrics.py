import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from src.docker_manager import DockerManager
import json
import traceback
import subprocess

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
        try:
            self.docker_manager = DockerManager()
            print("Successfully initialized Docker manager")
        except Exception as e:
            print(f"Failed to initialize Docker manager: {e}")
            print("Docker debug info:")
            try:
                subprocess.run(['docker', 'info'], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Docker info command failed: {e}")
            raise
        
    def collect_comprehensive_metrics(self):
        """收集综合性能指标"""
        try:
            containers = self.docker_manager._containers
            resource_metrics = self.collect_resource_metrics(containers)
            
            print("\nDebug - Resource metrics collected:")
            print(json.dumps(resource_metrics, indent=2))
            
            metrics = {
                'resource_usage': resource_metrics,  # 直接使用每个节点的资源指标
                'model_performance': {
                    'accuracy': self.performance_history.get('accuracy', 0.0),
                    'top5_accuracy': self.performance_history.get('top5_accuracy', 0.0),
                    'inference_time': self.performance_history.get('inference_time', 0.0),
                    'p95_latency': self.performance_history.get('p95_latency', 0.0)
                },
                'system_performance': {
                    'latency': self.performance_history.get('latency', 0.0),
                    'throughput': self.performance_history.get('throughput', 0.0),
                    'network_bandwidth': self.performance_history.get('network_bandwidth', 100.0),
                    'stability_score': self.performance_history.get('stability_score', 0.95)
                },
                'scheduling_efficiency': {
                    'load_balancing_score': self.performance_history.get('load_balancing_score', 0.0),
                    'resource_utilization': self.performance_history.get('resource_utilization', 0.0),
                    'scheduling_overhead': self.performance_history.get('scheduling_overhead', 10.0),
                    'task_queue_length': self.performance_history.get('task_queue_length', 5)
                }
            }
            
            print("\nDebug - Final metrics structure before return:")
            print(json.dumps(metrics, indent=2))
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting comprehensive metrics: {e}")
            traceback.print_exc()  # 打印完整的错误堆栈
            return {}
            
    def clear(self):
        """清理历史数据"""
        self.performance_history.clear()
        self.resource_history.clear()
        
    def collect_resource_metrics(self, containers):
        """改进的 CPU 和内存使用率计算"""
        if not containers:
            return {}
            
        samples = 3  # 采样次数
        resource_metrics = {}
        
        for node_id, container in containers.items():
            cpu_usages = []
            memory_usage = 0.0  # 初始化内存使用变量
            
            for _ in range(samples):
                try:
                    # 直接使用容器对象，而不是尝试获取
                    stats_1 = container.stats(stream=False)
                    time.sleep(1.0)
                    stats_2 = container.stats(stream=False)
                    
                    # CPU 使用率计算
                    cpu_total_1 = stats_1['cpu_stats']['cpu_usage']['total_usage']
                    cpu_total_2 = stats_2['cpu_stats']['cpu_usage']['total_usage']
                    system_usage_1 = stats_1['cpu_stats']['system_cpu_usage']
                    system_usage_2 = stats_2['cpu_stats']['system_cpu_usage']
                    cpu_count = len(stats_2['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))
                    
                    cpu_delta = cpu_total_2 - cpu_total_1
                    system_delta = system_usage_2 - system_usage_1
                    
                    if system_delta > 0:
                        cpu_usage = (cpu_delta / system_delta) * cpu_count * 100.0
                    else:
                        cpu_usage = 0.0
                        
                    # 内存使用计算
                    if 'memory_stats' in stats_2:
                        memory_usage = stats_2['memory_stats'].get('usage', 0) / (1024 * 1024)  # 转换为 MB
                    
                    cpu_usages.append(cpu_usage)
                    
                except Exception as e:
                    print(f"Error collecting metrics for {node_id}: {e}")
                    cpu_usages.append(0.0)
                    continue
            
            # 计算平均 CPU 使用率
            avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
            
            # 存储该节点的指标
            resource_metrics[node_id] = {
                'cpu_usage_percent': avg_cpu_usage,
                'memory_usage_mb': memory_usage
            }
        
        return resource_metrics

    def collect_system_metrics(self):
        """收集系统性能指标"""
        metrics = {
            'latency': [],
            'throughput': [],
            'network_usage': [],
            'stability': []
        }
        
        for container_id in self.docker_manager._containers.values():
            container = self.docker_manager.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # 网络使用统计
            networks = stats.get('networks', {})
            rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            metrics['network_usage'].append({
                'rx_bytes': rx_bytes,
                'tx_bytes': tx_bytes
            })
        
        return metrics

    def get_cpu_usage(self, containers):
        """获取 CPU 使用率并保存其他指标"""
        metrics = {
            'cpu_metrics': {},
            'model_metrics': {
                'accuracy': self.performance_history.get('accuracy', 0.0),
                'top5_accuracy': self.performance_history.get('top5_accuracy', 0.0),
                'inference_time': self.performance_history.get('inference_time', 0.0)
            },
            'system_metrics': {
                'latency': self.performance_history.get('latency', 0.0),
                'throughput': self.performance_history.get('throughput', 0.0),
                'communication_overhead': self.performance_history.get('communication_overhead', 0.0),
                'load_balancing_score': self.performance_history.get('load_balancing_score', 0.0),
                'scheduling_overhead': self.performance_history.get('scheduling_overhead', 0.0)
            }
        }
        
        # 收集 CPU 使用率
        for node_id, container_id in containers.items():
            try:
                container = self.docker_manager.client.containers.get(container_id)
                stats_1 = container.stats(stream=False)
                time.sleep(0.1)
                stats_2 = container.stats(stream=False)
                
                # CPU 使用率计算
                cpu_total_1 = stats_1['cpu_stats']['cpu_usage']['total_usage']
                cpu_total_2 = stats_2['cpu_stats']['cpu_usage']['total_usage']
                system_usage_1 = stats_1['cpu_stats']['system_cpu_usage']
                system_usage_2 = stats_2['cpu_stats']['system_cpu_usage']
                cpu_count = len(stats_2['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))
                
                cpu_delta = cpu_total_2 - cpu_total_1
                system_delta = system_usage_2 - system_usage_1
                
                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * cpu_count * 100.0
                else:
                    cpu_usage = 0.0
                    
                print(f"\n{node_id} CPU Usage: {cpu_usage:.4f}%")
                metrics['cpu_metrics'][node_id] = {
                    'usage_percent': round(cpu_usage, 4),
                    'cpu_limit': self.docker_manager.get_cpu_limit(container_id)
                }
                
            except Exception as e:
                print(f"Error getting CPU metrics for {node_id}: {e}")
                metrics['cpu_metrics'][node_id] = {
                    'usage_percent': 0.0,
                    'cpu_limit': 0.0
                }
        
        return metrics
