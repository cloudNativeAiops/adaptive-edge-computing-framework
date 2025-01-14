import json
import sys
import os
import traceback
from src.utils.metrics import MetricsCollector
from src.docker_manager import DockerManager

def show_metrics_from_file(metrics_file: str = 'results/evaluation/performance_metrics.json'):
    """从文件读取并显示性能指标"""
    try:
        if not os.path.exists(metrics_file):
            print(f"Error: Metrics file not found at {metrics_file}")
            return
            
        with open(metrics_file, 'r') as f:
            results = json.load(f)
            
        for profile, metrics in results.items():
            print(f"\n=== {profile.upper()} PROFILE ===")
            
            # 资源使用情况
            print("\nResource Usage:")
            for node_id, node_metrics in metrics['resource_usage'].items():
                print(f"\n{node_id}:")
                # 使用科学计数法显示小数
                cpu_usage = node_metrics.get('cpu_usage_percent', 0)
                if cpu_usage < 0.0001:
                    print(f"CPU Usage: {cpu_usage:.2e}%")  # 使用科学计数法
                else:
                    print(f"CPU Usage: {cpu_usage:.4f}%")  # 保留4位小数
                print(f"Memory Usage: {node_metrics.get('memory_usage_mb', 0):.2f}MB")
            
            # 模型性能
            print("\nModel Performance:")
            model_perf = metrics['model_performance']
            print(f"Accuracy (Top-1): {model_perf.get('accuracy', 0):.2f}%")
            print(f"Accuracy (Top-5): {model_perf.get('top5_accuracy', 0):.2f}%")
            print(f"Inference Time: {model_perf.get('inference_time', 0):.2f}ms")
            print(f"P95 Latency: {model_perf.get('p95_latency', 0):.2f}ms")
            
            # 系统性能
            print("\nSystem Performance:")
            sys_perf = metrics['system_performance']
            print(f"Latency: {sys_perf.get('latency', 0):.2f}ms")
            print(f"Throughput: {sys_perf.get('throughput', 0):.2f} requests/s")
            print(f"Network Bandwidth: {sys_perf.get('network_bandwidth', 0):.2f}MB/s")
            print(f"Stability Score: {sys_perf.get('stability_score', 0):.2f}")
            
            # 调度效率
            print("\nScheduling Efficiency:")
            sched = metrics['scheduling_efficiency']
            print(f"Load Balance Score: {sched.get('load_balancing_score', 0):.2f}")
            print(f"Resource Utilization: {sched.get('resource_utilization', 0):.2f}%")
            print(f"Scheduling Overhead: {sched.get('scheduling_overhead', 0):.2f}ms")
            print(f"Task Queue Length: {sched.get('task_queue_length', 0)}")
            
    except Exception as e:
        print(f"Error reading or parsing metrics: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    show_metrics_from_file() 