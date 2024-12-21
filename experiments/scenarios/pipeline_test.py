import logging
import torch
import torchvision.models as models
from src.core.model_partitioner import ModelPartitioner
from src.core.task_scheduler import AdaptiveScheduler, TaskRequirements, NodeResources
from src.utils.docker_utils import DockerManager
from src.utils.metrics import MetricsCollector
from src.models.mobilenet import MobileNetWrapper
import time
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineExperiment:
    def __init__(self):
        self.docker_manager = DockerManager()
        self.scheduler = AdaptiveScheduler()
        self.results = []  # Store results as dictionaries instead of using MetricsCollector
        
    def _convert_memory_to_mb(self, memory_str: str) -> float:
        """Convert memory string (e.g., '1g' or '512m') to MB."""
        value = float(memory_str[:-1])
        unit = memory_str[-1].lower()
        
        if unit == 'g':
            return value * 1024
        elif unit == 'm':
            return value
        else:
            raise ValueError(f"Unsupported memory unit: {unit}")
        
    def run_pipeline_test(self):
        """Run full pipeline test combining all components."""
        logger.info("Starting pipeline experiment")
        
        try:
            # 1. Set up edge nodes with different resource profiles
            nodes = [
                ("edge-node-1", {"cpu": 1.0, "memory": "1g"}),
                ("edge-node-2", {"cpu": 0.5, "memory": "512m"}),
                ("edge-node-3", {"cpu": 0.2, "memory": "256m"})
            ]
            
            # Create nodes and register with scheduler
            for node_name, resources in nodes:
                container_id = self.docker_manager.create_edge_node(
                    node_name,
                    cpu_limit=resources["cpu"],
                    memory_limit=resources["memory"]
                )
                
                memory_mb = self._convert_memory_to_mb(resources["memory"])
                self.scheduler.register_node(
                    node_name,
                    NodeResources(
                        cpu_available=resources["cpu"] * 100,
                        memory_available=memory_mb,
                        current_load=0.0,
                        node_id=node_name
                    )
                )
                logger.info(f"Registered node: {node_name} (CPU: {resources['cpu']*100}%, Memory: {memory_mb}MB)")
            
            # 2. Load and partition model
            model = models.mobilenet_v2(pretrained=True)
            partitioner = ModelPartitioner(model)
            
            partitions = partitioner.partition_model(
                num_partitions=3,
                resource_weights={"computation": 0.6, "memory": 0.4}
            )
            
            logger.info(f"Created {len(partitions)} model partitions")
            
            # 3. Create tasks for each partition
            tasks = []
            for i, partition in enumerate(partitions):
                partition_size = len(partition)
                base_memory = 256
                
                task = TaskRequirements(
                    cpu_needed=20.0 * (partition_size / 100),
                    memory_needed=base_memory * (partition_size / 100),
                    priority=len(partitions) - i,
                    model_name=f"mobilenet_part_{i}",
                    batch_size=1
                )
                tasks.append(task)
            
            # 4. Schedule and execute tasks
            for i, task in enumerate(tasks):
                logger.info(f"\nScheduling partition {i}")
                
                selected_node = self.scheduler.select_node(task)
                if not selected_node:
                    logger.warning(f"No suitable node found for partition {i}")
                    continue
                    
                logger.info(f"Selected node: {selected_node}")
                
                # Simulate task execution
                start_time = time.time()
                time.sleep(0.1)  # Simulate processing
                execution_time = (time.time() - start_time) * 1000
                
                # Record completion
                self.scheduler.record_task_completion(selected_node, execution_time)
                
                # Save results
                self.results.append({
                    "partition": i,
                    "node": selected_node,
                    "execution_time_ms": execution_time,
                    "cpu_needed": task.cpu_needed,
                    "memory_needed": task.memory_needed,
                    "priority": task.priority
                })
            
            # 5. Print final statistics
            logger.info("\nFinal node statistics:")
            stats = self.scheduler.get_node_stats()
            for node_id, node_stats in stats.items():
                logger.info(f"{node_id}: {node_stats}")
            
            # Save results to file
            os.makedirs("results/pipeline", exist_ok=True)
            with open("results/pipeline/pipeline_results.json", "w") as f:
                json.dump({
                    "task_assignments": self.results,
                    "node_stats": stats
                }, f, indent=2)
                
            logger.info("Pipeline experiment completed")
            
        finally:
            # Cleanup
            self.docker_manager.cleanup()

if __name__ == "__main__":
    experiment = PipelineExperiment()
    experiment.run_pipeline_test() 