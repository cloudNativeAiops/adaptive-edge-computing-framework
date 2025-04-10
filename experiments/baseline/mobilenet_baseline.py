import torch
import time
from src.models.mobilenet import MobileNetWrapper
from torchvision.models import MobileNet_V2_Weights
from src.utils.metrics import MetricsCollector, InferenceMetrics
from src.utils.docker_utils import DockerManager
from src.core.resource_monitor import ResourceMonitor

# Get recommended preprocessing transforms for the model weights
weights = MobileNet_V2_Weights.IMAGENET1K_V1
preprocess = weights.transforms()


def run_baseline_experiment(num_iterations: int = 100, batch_size: int = 1):
    """Run baseline MobileNet inference experiment.

    Args:
        num_iterations: Number of inference iterations
        batch_size: Batch size for inference
    """
    # Initialize components
    model = MobileNetWrapper(pretrained=True).eval()
    metrics_collector = MetricsCollector()
    resource_monitor = ResourceMonitor()

    # Create dummy raw input
    # We apply preprocessing inside the loop for timing accuracy
    input_tensor_raw = torch.randn(batch_size, 3, 224, 224)

    print(f"Running baseline experiment with {num_iterations} iterations...")

    for i in range(num_iterations):
        # Get resource usage before inference
        host_stats = resource_monitor.get_host_stats()

        # Apply preprocessing
        input_tensor = preprocess(input_tensor_raw)

        # Run inference using standard model call
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        inference_time = time.time() - start_time

        # Collect metrics
        metrics_collector.add_metric(InferenceMetrics(
            inference_time=inference_time * 1000,  # Convert to ms
            memory_usage=host_stats['memory_percent'],
            cpu_usage=host_stats['cpu_percent'],
            model_name='mobilenet_v2_baseline',
            batch_size=batch_size,
            device=str(next(model.model.parameters()).device)
        ))

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} iterations")

    # Save results
    # Use full path including directory
    output_filepath = "results/baseline/baseline_results.json"
    metrics_collector.save_metrics(output_filepath)
    # The print statement below was potentially misleading, adjust if needed or keep as is.
    print(f"\nExperiment completed. Metrics saved to {output_filepath}")

    # Print summary using the keys returned by get_summary
    summary = metrics_collector.get_summary()
    print("\nSummary:")
    print(f"Count: {summary['count']}")
    print(
        f"Average inference time: {summary['avg_inference_time_ms']:.2f} ms (Std: {summary['std_inference_time_ms']:.2f} ms)")
    print(f"Average CPU usage: {summary['avg_cpu_usage_percent']:.2f}%")
    print(f"Average memory usage: {summary['avg_memory_usage_percent']:.2f}%")


if __name__ == "__main__":
    run_baseline_experiment()
