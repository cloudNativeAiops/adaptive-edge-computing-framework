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
    partition_points: List[int]          # model partition points
    partition_sizes: List[float]         # size of each partition (MB)
    partition_compute_costs: List[float]  # compute cost of each partition
    communication_costs: float           # communication cost between partitions
    partition_latencies: List[float]     # latency of each partition
    total_transfer_size: float          # total data transferred between partitions

@dataclass
class SchedulingMetrics:
    task_distribution: Dict[str, int]    # task distribution on each node
    queue_lengths: List[int]             # task queue length history
    scheduling_overhead: float           # scheduling decision time
    load_balancing_score: float         # load balancing score
    resource_utilization: Dict[str, float] # resource utilization of each node

@dataclass
class SystemMetrics:
    end_to_end_latency: float           # end-to-end latency
    throughput: float                    # system throughput
    network_bandwidth_usage: float       # network bandwidth usage
    energy_consumption: float            # energy consumption
    system_stability_score: float        # system stability score

class MetricsCollector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.docker_manager = DockerManager()  # use singleton pattern
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
        """collect comprehensive performance metrics"""
        try:
            containers = self.docker_manager._containers
            resource_metrics = self.collect_resource_metrics(containers)
            
            print("\nDebug - Resource metrics collected:")
            print(json.dumps(resource_metrics, indent=2))
            
            metrics = {
                'resource_usage': resource_metrics,  # use the resource metrics of each node directly
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
            traceback.print_exc()  # print the full error stack
            return {}
            
    def clear(self):
        """clear historical data"""
        self.performance_history.clear()
        self.resource_history.clear()
        
    def collect_resource_metrics(self, containers):
        """improved CPU and memory usage rate calculation"""
        if not containers:
            return {}
            
        samples = 3  # sampling times
        resource_metrics = {}
        
        for node_id, container in containers.items():
            cpu_usages = []
            memory_usage = 0.0  # initialize the memory usage variable
            
            for _ in range(samples):
                try:
                    # use the container object directly, not trying to get it
                    stats_1 = container.stats(stream=False)
                    time.sleep(1.0)
                    stats_2 = container.stats(stream=False)
                    
                    # CPU usage rate calculation
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
                        
                    # memory usage calculation
                    if 'memory_stats' in stats_2:
                        memory_usage = stats_2['memory_stats'].get('usage', 0) / (1024 * 1024)  # convert to MB
                    
                    cpu_usages.append(cpu_usage)
                    
                except Exception as e:
                    print(f"Error collecting metrics for {node_id}: {e}")
                    cpu_usages.append(0.0)
                    continue
            
            # calculate the average CPU usage rate
            avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0
            
            # store the metrics of this node
            resource_metrics[node_id] = {
                'cpu_usage_percent': avg_cpu_usage,
                'memory_usage_mb': memory_usage
            }
        
        return resource_metrics

    def collect_system_metrics(self):
        """collect system performance metrics"""
        metrics = {
            'latency': [],
            'throughput': [],
            'network_usage': [],
            'stability': []
        }
        
        for container_id in self.docker_manager._containers.values():
            container = self.docker_manager.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # network usage statistics
            networks = stats.get('networks', {})
            rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            metrics['network_usage'].append({
                'rx_bytes': rx_bytes,
                'tx_bytes': tx_bytes
            })
        
        return metrics

    def get_cpu_usage(self, containers):
        """get CPU usage rate and save other metrics"""
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
        
        # collect CPU usage rate
        for node_id, container_id in containers.items():
            try:
                container = self.docker_manager.client.containers.get(container_id)
                stats_1 = container.stats(stream=False)
                time.sleep(0.1)
                stats_2 = container.stats(stream=False)
                
                # CPU usage rate calculation
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
