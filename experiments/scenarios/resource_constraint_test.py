import torch
import logging
from src.models.mobilenet import MobileNetWrapper
from src.utils.metrics import MetricsCollector, InferenceMetrics
from src.utils.docker_utils import DockerManager
from src.core.resource_monitor import ResourceMonitor
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ResourceConstraintExperiment:
    def __init__(self):
        """Initialize experiment components."""
        self.docker_manager = DockerManager()
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector(output_dir="results/scenarios")
        self.logger = logging.getLogger(__name__)

    def run_resource_scenarios(self):
        """Run experiments under different resource constraints."""
        scenarios = [
            {"name": "high_resource", "cpu": 1.0, "memory": "1g"},
            {"name": "medium_resource", "cpu": 0.5, "memory": "512m"},
            {"name": "low_resource", "cpu": 0.2, "memory": "256m"}
        ]
        
        for scenario in scenarios:
            self.logger.info(f"\nRunning scenario: {scenario['name']}")
            self._run_scenario(scenario)
            
        self.metrics_collector.save_metrics("resource_constraint_results.json")

    def _run_scenario(self, scenario):
        """Run experiment for a specific resource constraint scenario."""
        self.logger.info(f"\nRunning scenario: {scenario['name']}")
        
        try:
            # Create container with specified resources
            container_id = self.docker_manager.create_edge_node(
                f"test-node-{scenario['name']}",
                cpu_limit=scenario['cpu'],
                memory_limit=scenario['memory']
            )
            
            # Verify container is running
            container_info = self.docker_manager.get_container_info(container_id)
            self.logger.info(f"Container status: {container_info['status']}")
            
            if container_info['status'] != 'running':
                raise RuntimeError(f"Container failed to start: {container_info}")
            
            # Wait for container to be ready
            time.sleep(5)
            
            # Initialize model
            model = MobileNetWrapper(pretrained=True)
            input_tensor = torch.randn(1, 3, 224, 224)
            input_tensor = model.preprocess_input(input_tensor)
            
            # Warm-up
            self.logger.info("Running warm-up iterations...")
            for _ in range(5):
                model.inference(input_tensor)
            
            # Test runs
            self.logger.info("Starting test iterations...")
            for i in range(20):
                try:
                    container_stats = self.resource_monitor.get_container_stats(container_id)
                    if container_stats is None:
                        self.logger.warning(f"No stats available for iteration {i}")
                        continue
                        
                    start_time = time.time()
                    output, metadata = model.inference(input_tensor)
                    inference_time = time.time() - start_time
                    
                    self.metrics_collector.add_metric(InferenceMetrics(
                        inference_time=inference_time * 1000,
                        memory_usage=container_stats['memory_usage']['percentage'],
                        cpu_usage=container_stats['cpu_usage'],
                        model_name=f"mobilenet_{scenario['name']}",
                        batch_size=1,
                        device=str(model.device)
                    ))
                    
                    if (i + 1) % 5 == 0:
                        self.logger.info(f"Completed {i + 1}/20 iterations")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to collect metrics for iteration {i}: {e}")
                    continue
                
            # Print scenario summary
            scenario_metrics = [m for m in self.metrics_collector.metrics 
                              if scenario['name'] in m.model_name]
            if scenario_metrics:
                avg_time = sum(m.inference_time for m in scenario_metrics) / len(scenario_metrics)
                self.logger.info(f"Average inference time: {avg_time:.2f}ms")
            else:
                self.logger.warning(f"No metrics collected for scenario: {scenario['name']}")
            
        finally:
            # Cleanup
            self.docker_manager.cleanup()

if __name__ == "__main__":
    experiment = ResourceConstraintExperiment()
    experiment.run_resource_scenarios() 