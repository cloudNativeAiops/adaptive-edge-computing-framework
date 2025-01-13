import json
import sys

def show_metrics(metrics_file: str = 'results/evaluation/performance_metrics.json'):
    try:
        with open(metrics_file) as f:
            data = json.load(f)
        
        for profile in data.keys():
            print(f'\n=== Profile: {profile} ===')
            
            # Infrastructure Metrics
            print('\nInfrastructure Metrics:')
            print(f'CPU Usage: {data[profile]["resource_usage"]["cpu_percent"]}%')
            print(f'Memory Usage: {data[profile]["resource_usage"]["memory_percent"]}%')
            
            # Model Metrics
            print('\nModel Metrics:')
            print(f'Accuracy (Top-1): {data[profile]["accuracy"]["top1"]}%')
            print(f'Accuracy (Top-5): {data[profile]["accuracy"]["top5"]}%')
            print(f'Inference Latency: {data[profile]["latency"]["average_ms"]} ms')
            print(f'P95 Latency: {data[profile]["latency"]["p95_ms"]} ms')
            
            # Partitioning Metrics
            if "partition" in data[profile]:
                print('\nPartitioning Metrics:')
                print(f'Communication Cost: {data[profile]["partition"]["communication_costs"]} MB')
                print(f'Partition Latencies: {data[profile]["partition"]["partition_latencies"]} ms')
            
            # Scheduling Metrics
            if "scheduling" in data[profile]:
                print('\nScheduling Metrics:')
                print(f'Load Balance Score: {data[profile]["scheduling"]["load_balancing_score"]}')
                print(f'Scheduling Overhead: {data[profile]["scheduling"]["scheduling_overhead"]} ms')
            
            # System Performance
            if "system" in data[profile]:
                print('\nSystem Performance:')
                print(f'End-to-end Latency: {data[profile]["system"]["end_to_end_latency"]} ms')
                print(f'Throughput: {data[profile]["system"]["throughput"]} inferences/s')
                print(f'Network Usage: {data[profile]["system"]["network_bandwidth_usage"]} MB/s')
    except FileNotFoundError:
        print(f"Error: Metrics file not found at {metrics_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {metrics_file}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key {e} in metrics data")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def format_metrics(metrics):
    """格式化指标输出"""
    return {
        'Resource Usage': {
            node_id: {
                'CPU': f"{metrics['resource_metrics'][node_id]['cpu_usage_percent']}%",
                'Memory': f"{metrics['resource_metrics'][node_id]['memory_usage_mb']}MB"
            }
            for node_id in metrics['resource_metrics']
        },
        'Model Performance': metrics['model_metrics'],
        'System Performance': metrics['system_metrics'],
        'Scheduling Efficiency': metrics['scheduling_metrics']
    }

if __name__ == "__main__":
    show_metrics() 