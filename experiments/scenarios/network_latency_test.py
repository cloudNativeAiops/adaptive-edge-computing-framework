import torch
import time
from src.models.mobilenet import MobileNetWrapper
from src.core.model_partitioner import ModelPartitioner
from src.utils.metrics import MetricsCollector, InferenceMetrics
import subprocess
import numpy as np

class NetworkLatencyExperiment:
    def __init__(self):
        self.model = MobileNetWrapper(pretrained=True)
        self.partitioner = ModelPartitioner(self.model.model)
        self.metrics_collector = MetricsCollector(output_dir="results/network")

    def simulate_network_conditions(self, latency_ms: int, packet_loss: float):
        """Simulate network conditions using tc (traffic control)."""
        try:
            # Add network delay and packet loss
            subprocess.run([
                'sudo', 'tc', 'qdisc', 'add', 'dev', 'docker0', 'root', 'netem',
                'delay', f'{latency_ms}ms', f'{latency_ms//10}ms',
                'loss', f'{packet_loss}%'
            ])
        except subprocess.CalledProcessError as e:
            print(f"Failed to set network conditions: {e}")

    def reset_network_conditions(self):
        """Reset network conditions."""
        try:
            subprocess.run(['sudo', 'tc', 'qdisc', 'del', 'dev', 'docker0', 'root'])
        except subprocess.CalledProcessError:
            pass

    def run_network_scenarios(self):
        """Run experiments under different network conditions."""
        scenarios = [
            {"name": "ideal", "latency": 0, "loss": 0},
            {"name": "good", "latency": 20, "loss": 0.1},
            {"name": "poor", "latency": 100, "loss": 2.0},
            {"name": "bad", "latency": 200, "loss": 5.0}
        ]
        
        input_tensor = torch.randn(1, 3, 224, 224)
        input_tensor = self.model.preprocess_input(input_tensor)
        
        for scenario in scenarios:
            print(f"\nTesting network scenario: {scenario['name']}")
            try:
                self.simulate_network_conditions(
                    scenario['latency'],
                    scenario['loss']
                )
                
                # Run multiple iterations
                latencies = []
                for _ in range(20):
                    start_time = time.time()
                    self.model.inference(input_tensor)
                    latencies.append((time.time() - start_time) * 1000)
                
                # Record metrics
                self.metrics_collector.add_metric(InferenceMetrics(
                    inference_time=np.mean(latencies),
                    memory_usage=0,  # Not relevant for network test
                    cpu_usage=0,     # Not relevant for network test
                    model_name=f"network_{scenario['name']}",
                    batch_size=1,
                    device=str(self.model.device)
                ))
                
                print(f"Average latency: {np.mean(latencies):.2f}ms")
                print(f"Latency std: {np.std(latencies):.2f}ms")
                
            finally:
                self.reset_network_conditions()
        
        self.metrics_collector.save_metrics("network_latency_results.json")

if __name__ == "__main__":
    experiment = NetworkLatencyExperiment()
    experiment.run_network_scenarios() 