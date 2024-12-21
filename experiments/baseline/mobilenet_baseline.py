import torch
import time
from src.models.mobilenet import MobileNetWrapper
from src.utils.metrics import MetricsCollector, InferenceMetrics
from src.utils.docker_utils import DockerManager
from src.core.resource_monitor import ResourceMonitor

def run_baseline_experiment(num_iterations: int = 100, batch_size: int = 1):
    """Run baseline MobileNet inference experiment.
    
    Args:
        num_iterations: Number of inference iterations
        batch_size: Batch size for inference
    """
    # Initialize components
    model = MobileNetWrapper(pretrained=True)
    metrics_collector = MetricsCollector(output_dir="results/baseline")
    resource_monitor = ResourceMonitor()
    
    # Create dummy input
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    input_tensor = model.preprocess_input(input_tensor)
    
    print(f"Running baseline experiment with {num_iterations} iterations...")
    
    for i in range(num_iterations):
        # Get resource usage before inference
        host_stats = resource_monitor.get_host_stats()
        
        # Run inference
        start_time = time.time()
        output, metadata = model.inference(input_tensor)
        inference_time = time.time() - start_time
        
        # Collect metrics
        metrics_collector.add_metric(InferenceMetrics(
            inference_time=inference_time * 1000,  # Convert to ms
            memory_usage=host_stats['memory_percent'],
            cpu_usage=host_stats['cpu_percent'],
            model_name='mobilenet_v2',
            batch_size=batch_size,
            device=str(model.device)
        ))
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} iterations")
    
    # Save results
    metrics_collector.save_metrics("baseline_results.json")
    print("\nExperiment completed. Results saved to results/baseline/")
    
    # Print summary
    summary = metrics_collector.get_summary()
    print("\nSummary:")
    print(f"Average inference time: {summary['inference_time']['mean']:.2f} ms")
    print(f"Average CPU usage: {summary['cpu_usage']['mean']:.2f}%")
    print(f"Average memory usage: {summary['memory_usage']['mean']:.2f}%")

if __name__ == "__main__":
    run_baseline_experiment() 