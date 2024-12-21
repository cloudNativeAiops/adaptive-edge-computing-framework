import time
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import json
import os

@dataclass
class InferenceMetrics:
    inference_time: float
    memory_usage: float
    cpu_usage: float
    model_name: str
    batch_size: int
    device: str

class MetricsCollector:
    def __init__(self, output_dir: str = "results"):
        """Initialize metrics collector.
        
        Args:
            output_dir: Directory to save metrics
        """
        self.metrics: List[InferenceMetrics] = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def add_metric(self, metric: InferenceMetrics):
        """Add a new metric measurement."""
        self.metrics.append(metric)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        if not self.metrics:
            return {}

        inference_times = [m.inference_time for m in self.metrics]
        memory_usages = [m.memory_usage for m in self.metrics]
        cpu_usages = [m.cpu_usage for m in self.metrics]

        return {
            'inference_time': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times)
            },
            'memory_usage': {
                'mean': np.mean(memory_usages),
                'std': np.std(memory_usages)
            },
            'cpu_usage': {
                'mean': np.mean(cpu_usages),
                'std': np.std(cpu_usages)
            },
            'total_samples': len(self.metrics)
        }

    def save_metrics(self, filename: str):
        """Save metrics to file."""
        metrics_data = [vars(m) for m in self.metrics]
        summary = self.get_summary()
        
        output_data = {
            'summary': summary,
            'detailed_metrics': metrics_data
        }
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def clear(self):
        """Clear collected metrics."""
        self.metrics.clear() 