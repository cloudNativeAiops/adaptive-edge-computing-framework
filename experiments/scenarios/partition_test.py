import logging
import torch
import torchvision.models as models
from src.core.model_partitioner import ModelPartitioner
from src.utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_partition_experiment():
    """Test model partitioning with different configurations."""
    logger.info("Starting model partitioning experiment")
    
    # Load model
    model = models.mobilenet_v2(pretrained=True)
    partitioner = ModelPartitioner(model)
    results = []  # Store results as dictionaries
    
    # Test different partition configurations
    partition_configs = [
        {"num_partitions": 2, "weights": {"computation": 0.7, "memory": 0.3}},
        {"num_partitions": 3, "weights": {"computation": 0.5, "memory": 0.5}},
        {"num_partitions": 4, "weights": {"computation": 0.3, "memory": 0.7}}
    ]
    
    for config in partition_configs:
        logger.info(f"Testing partition config: {config}")
        
        # Create partitions
        partitions = partitioner.partition_model(
            num_partitions=config["num_partitions"],
            resource_weights=config["weights"]
        )
        
        # Analyze communication costs
        costs = partitioner.analyze_communication_cost(partitions)
        
        # Log results
        logger.info(f"Created {len(partitions)} partitions")
        for i, partition in enumerate(partitions):
            logger.info(f"Partition {i}: {len(partition)} layers")
            
        logger.info("Communication costs between partitions:")
        for link, cost in costs.items():
            logger.info(f"{link}: {cost:.2f} bytes")
            
        # Save results
        results.append({
            "config": config,
            "num_partitions": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "communication_costs": costs
        })
    
    # Save results to file
    import json
    with open("results/partitioning/partition_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Partitioning experiment completed")

if __name__ == "__main__":
    run_partition_experiment() 