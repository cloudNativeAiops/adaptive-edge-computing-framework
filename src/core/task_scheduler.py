import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class TaskRequirements:
    cpu_needed: float
    memory_needed: float
    priority: int
    model_name: str
    batch_size: int

@dataclass
class NodeResources:
    cpu_available: float
    memory_available: float
    current_load: float
    node_id: str

class AdaptiveScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, NodeResources] = {}
        self.task_history: Dict[str, List[float]] = {}  # node_id -> execution times
        self.node_task_count: Dict[str, int] = {}  # Track number of tasks per node
        
    def register_node(self, node_id: str, resources: NodeResources):
        """Register a new edge node with its resources."""
        self.nodes[node_id] = resources
        self.task_history[node_id] = []
        self.node_task_count[node_id] = 0
        
    def update_node_status(self, node_id: str, resources: NodeResources):
        """Update node resource status."""
        self.nodes[node_id] = resources
        
    def select_node(self, task: TaskRequirements) -> Optional[str]:
        """Select best node for task based on resources and load balancing."""
        candidate_nodes = []
        
        for node_id, resources in self.nodes.items():
            # Check if node has enough resources
            if (resources.cpu_available >= task.cpu_needed and 
                resources.memory_available >= task.memory_needed):
                
                # Resource score - normalized by task requirements
                resource_score = (
                    (resources.cpu_available / task.cpu_needed) * 0.5 +
                    (resources.memory_available / task.memory_needed) * 0.5
                )
                
                # Load score - penalize heavily loaded nodes
                load_score = 1 - resources.current_load
                
                # Task distribution score - strongly favor less used nodes
                task_count = self.node_task_count[node_id]
                balance_score = 1 / (1 + task_count * 2)  # Increased penalty for task count
                
                # Historical performance score
                if self.task_history[node_id]:
                    avg_exec_time = np.mean(self.task_history[node_id])
                    perf_score = 1 / (1 + avg_exec_time)
                else:
                    perf_score = 0.5
                
                # Final score with adjusted weights
                # Give more weight to balance_score and less to resource_score
                total_score = (
                    0.2 * resource_score +    # Reduced from 0.3
                    0.2 * load_score +
                    0.1 * perf_score +        # Reduced from 0.2
                    0.5 * balance_score       # Increased from 0.3
                )
                
                # Add small random factor to break ties (0-5% of total score)
                randomization = np.random.uniform(0, 0.05) * total_score
                total_score += randomization
                
                self.logger.debug(f"Node {node_id} scores - Resource: {resource_score:.3f}, "
                              f"Load: {load_score:.3f}, Balance: {balance_score:.3f}, "
                              f"Performance: {perf_score:.3f}, Total: {total_score:.3f}")
                
                candidate_nodes.append((node_id, total_score))
        
        if not candidate_nodes:
            return None
            
        # Select node with highest score
        selected_node = max(candidate_nodes, key=lambda x: x[1])[0]
        
        # Update task count for the selected node
        self.node_task_count[selected_node] += 1
        
        return selected_node
    
    def record_task_completion(self, node_id: str, execution_time: float):
        """Record task execution time and update node status."""
        if node_id in self.task_history:
            self.task_history[node_id].append(execution_time)
            # Keep last 100 records
            self.task_history[node_id] = self.task_history[node_id][-100:]
            
            # Update node load based on recent performance
            recent_load = np.mean(self.task_history[node_id][-5:]) if len(self.task_history[node_id]) >= 5 else 0
            self.nodes[node_id].current_load = min(1.0, recent_load / 100.0)  # Normalize to 0-1 range
            
    def get_node_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all nodes."""
        stats = {}
        for node_id, history in self.task_history.items():
            if history:
                stats[node_id] = {
                    'avg_execution_time': np.mean(history),
                    'min_execution_time': np.min(history),
                    'max_execution_time': np.max(history),
                    'std_execution_time': np.std(history),
                    'task_count': self.node_task_count[node_id],
                    'current_load': self.nodes[node_id].current_load
                }
        return stats 