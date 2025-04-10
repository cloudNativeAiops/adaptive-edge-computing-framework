import torch
import time
from src.models.mobilenet import MobileNetWrapper
from torchvision.models import MobileNet_V2_Weights
from src.core.model_partitioner import ModelPartitioner
from src.core.task_scheduler import TaskScheduler, TaskRequirements, NodeResources
from src.utils.metrics import MetricsCollector, InferenceMetrics
from src.utils.docker_utils import DockerManager
from src.core.resource_monitor import ResourceMonitor
import argparse

# Get recommended preprocessing transforms for the model weights
weights = MobileNet_V2_Weights.IMAGENET1K_V1
preprocess = weights.transforms()


class AdaptiveInference:
    def __init__(self, num_nodes: int = 2):
        """Initialize adaptive inference system.

        Args:
            num_nodes: Number of edge nodes to use
        """
        self.docker_manager = DockerManager()
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector()
        self.task_scheduler = TaskScheduler()

        # Set up edge nodes
        self.setup_edge_nodes(num_nodes)

        # Initialize model and ensure it's in eval mode
        self.model = MobileNetWrapper(pretrained=True).eval()
        self.partitioner = ModelPartitioner(self.model.model)

    def setup_edge_nodes(self, num_nodes: int):
        """Set up Docker containers for edge nodes with heterogeneous resources."""
        self.docker_manager.setup_network()
        self.node_ids = []

        # Define resource profiles based on paper's Table 1 (High, Medium, Low)
        # CPU values are fractions (e.g., 1.0 means 1 core)
        profiles = [
            {'name': 'high', 'cpu': 1.0, 'mem': '1024m'},  # 1 CPU, 1GB RAM
            {'name': 'medium', 'cpu': 0.6, 'mem': '512m'},  # 0.6 CPU, 512MB RAM
            {'name': 'low', 'cpu': 0.4, 'mem': '512m'}  # 0.4 CPU, 512MB RAM
        ]

        print(f"Setting up {num_nodes} heterogeneous nodes...")

        for i in range(num_nodes):
            # Cycle through profiles if num_nodes > len(profiles)
            profile = profiles[i % len(profiles)]
            node_name = f"edge-node-{i}-{profile['name']}"  # Add profile name to node name
            cpu_limit = profile['cpu']
            memory_limit = profile['mem']

            print(f"Creating node: {node_name} (CPU: {cpu_limit}, Mem: {memory_limit})")

            node_id = self.docker_manager.create_edge_node(
                node_name,  # Use descriptive name
                cpu_limit=cpu_limit,  # Use profile CPU
                memory_limit=memory_limit  # Use profile Memory
            )
            self.node_ids.append(node_id)

            # Register node with initial (placeholder) resources in scheduler
            # Actual resources will be updated by monitoring loop
            initial_resources = NodeResources(
                cpu_available=100.0,  # Placeholder
                memory_available=100.0,  # Placeholder
                current_load=0.0,
                node_id=node_id
            )
            self.task_scheduler.register_node(node_id, initial_resources)

    def run_experiment(self, num_iterations: int = 100, batch_size: int = 1):
        """Run adaptive inference experiment."""
        # Create dummy raw input tensor
        input_tensor_raw = torch.randn(batch_size, 3, 224, 224)

        print(f"Running adaptive inference experiment with {num_iterations} iterations...")

        for i in range(num_iterations):
            # Apply preprocessing inside the loop (or where inference actually happens)
            input_tensor = preprocess(input_tensor_raw)

            # Update resource monitoring and node status in scheduler
            for node_id in self.node_ids:
                resources = self.resource_monitor.get_container_stats(node_id)
                if resources:
                    # Calculate availability (assuming usage is percentage)
                    cpu_avail = 100.0 - resources.get('cpu_usage', 100.0)  # Default to 0 avail if not found
                    mem_avail = 100.0 - resources.get('memory_usage', {}).get('percentage', 100.0)  # Default to 0 avail

                    # Get current load from scheduler to preserve it
                    current_node_data = self.task_scheduler.nodes.get(node_id)
                    current_load = current_node_data.current_load if current_node_data else 0.0

                    # Create NodeResources object
                    node_data = NodeResources(
                        cpu_available=max(0.0, cpu_avail),  # Ensure non-negative
                        memory_available=max(0.0, mem_avail),
                        current_load=current_load,
                        node_id=node_id
                    )

                    # Call the correct method with the correct object type
                    self.task_scheduler.update_node_status(node_id, node_data)
                else:
                    print(f"Warning: Could not get stats for node {node_id}")

            # Run inference with adaptive scheduling
            start_time = time.time()
            self._run_adaptive_inference(input_tensor)
            inference_time = time.time() - start_time

            # Collect metrics
            self._collect_metrics(inference_time, batch_size)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} iterations")

        # Save results
        # Use full path including directory
        output_path = "results/optimized/adaptive_results.json"
        self.metrics_collector.save_metrics(output_path)
        print(f"\nExperiment completed. Results saved to {output_path}")

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
            inference_time=inference_time * 1000,  # ms
            memory_usage=host_stats['memory_percent'],
            cpu_usage=host_stats['cpu_percent'],
            model_name='mobilenet_v2_adaptive',
            batch_size=batch_size,
            # Get device from underlying model parameters
            device=str(next(self.model.model.parameters()).device)
        ))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run adaptive inference experiment.')
    parser.add_argument('--nodes', type=int, default=3,
                        help='Number of edge nodes to simulate (default: 3)')
    # Add other potential arguments here if needed in the future
    # parser.add_argument('--iterations', type=int, default=100, help='Number of inference iterations')
    # parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')

    args = parser.parse_args()

    # Use the parsed arguments
    print(f"Initializing adaptive inference system with {args.nodes} nodes...")
    adaptive_system = AdaptiveInference(num_nodes=args.nodes)
    # Pass other args if added:
    # adaptive_system.run_experiment(num_iterations=args.iterations, batch_size=args.batch_size)
    adaptive_system.run_experiment()
