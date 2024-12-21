import logging
from src.core.task_scheduler import AdaptiveScheduler, TaskRequirements, NodeResources
from src.utils.metrics import MetricsCollector
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scheduler_experiment():
    """Test adaptive scheduler with different load scenarios."""
    logger.info("Starting scheduler experiment")
    
    scheduler = AdaptiveScheduler()
    results = []
    
    # Register test nodes
    nodes = [
        ("node1", NodeResources(1.0, 1024, 0.0, "node1")),
        ("node2", NodeResources(0.5, 512, 0.0, "node2")),
        ("node3", NodeResources(0.2, 256, 0.0, "node3"))
    ]
    
    for node_id, resources in nodes:
        scheduler.register_node(node_id, resources)
        logger.info(f"Registered node: {node_id}")
    
    # Test different task scenarios
    tasks = [
        TaskRequirements(0.3, 256, 1, "small_model", 1),
        TaskRequirements(0.5, 512, 2, "medium_model", 1),
        TaskRequirements(0.8, 768, 3, "large_model", 1)
    ]
    
    # Run scheduling tests
    for task in tasks:
        logger.info(f"\nTesting task: {task.model_name}")
        
        # Select node
        selected_node = scheduler.select_node(task)
        if selected_node:
            logger.info(f"Selected node: {selected_node}")
            
            # Simulate task execution
            execution_time = random.uniform(10, 100)
            scheduler.record_task_completion(selected_node, execution_time)
            
            results.append({
                "task": task.model_name,
                "selected_node": selected_node,
                "execution_time": execution_time
            })
        else:
            logger.warning("No suitable node found")
    
    # Get final stats
    stats = scheduler.get_node_stats()
    logger.info("\nFinal node statistics:")
    for node_id, node_stats in stats.items():
        logger.info(f"{node_id}: {node_stats}")
    
    # Save results
    import json
    with open("results/scheduling/scheduler_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Scheduler experiment completed")

if __name__ == "__main__":
    run_scheduler_experiment() 