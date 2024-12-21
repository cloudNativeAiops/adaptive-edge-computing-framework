import torch
import time
from src.models.mobilenet import MobileNetWrapper
from src.core.model_partitioner import ModelPartitioner
from src.core.task_scheduler import TaskScheduler, Task
from src.utils.metrics import MetricsCollector, InferenceMetrics
from src.utils.docker_utils import DockerManager
from src.core.resource_monitor import ResourceMonitor

class AdaptiveInference:
    def __init__(self, num_nodes: int = 2):
        """Initialize adaptive inference system.
        
        Args:
            num_nodes: Number of edge nodes to use
        """
        self.docker_manager = DockerManager()
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector(output_dir="results/optimized")
        self.task_scheduler = TaskScheduler()
        
        # Set up edge nodes
        self.setup_edge_nodes(num_nodes)
        
        # Initialize model
        self.model = MobileNetWrapper(pretrained=True)
        self.partitioner = ModelPartitioner(self.model.model)

    def setup_edge_nodes(self, num_nodes: int):
        """Set up Docker containers for edge nodes."""
        self.docker_manager.setup_network()
        self.node_ids = []
        
        for i in range(num_nodes):
            node_id = self.docker_manager.create_edge_node(
                f"edge-node-{i}",
                cpu_limit=0.5,
                memory_limit="512m"
            )
            self.node_ids.append(node_id)

    def run_experiment(self, num_iterations: int = 100, batch_size: int = 1):
        """Run adaptive inference experiment."""
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        input_tensor = self.model.preprocess_input(input_tensor)
        
        print(f"Running adaptive inference experiment with {num_iterations} iterations...")
        
        for i in range(num_iterations):
            # Update resource monitoring
            for node_id in self.node_ids:
                resources = self.resource_monitor.get_container_stats(node_id)
                self.task_scheduler.update_node_resources(node_id, {
                    'cpu': resources['cpu_usage'],
                    'memory': resources['memory_usage']['percentage']
                })
            
            # Run inference with adaptive scheduling
            start_time = time.time()
            self._run_adaptive_inference(input_tensor)
            inference_time = time.time() - start_time
            
            # Collect metrics
            self._collect_metrics(inference_time, batch_size)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} iterations")
        
        # Save results
        self.metrics_collector.save_metrics("adaptive_results.json")
        print("\nExperiment completed. Results saved to results/optimized/")
        
        # Cleanup
        self.docker_manager.cleanup()

    def _run_adaptive_inference(self, input_tensor):
        """Run inference with adaptive scheduling."""
        # TODO: Implement adaptive inference logic
        pass

    def _collect_metrics(self, inference_time: float, batch_size: int):
        """Collect performance metrics."""
        host_stats = self.resource_monitor.get_host_stats()
        
        self.metrics_collector.add_metric(InferenceMetrics(
            inference_time=inference_time * 1000,
            memory_usage=host_stats['memory_percent'],
            cpu_usage=host_stats['cpu_percent'],
            model_name='mobilenet_v2_adaptive',
            batch_size=batch_size,
            device=str(self.model.device)
        ))

if __name__ == "__main__":
    adaptive_system = AdaptiveInference(num_nodes=2)
    adaptive_system.run_experiment() 