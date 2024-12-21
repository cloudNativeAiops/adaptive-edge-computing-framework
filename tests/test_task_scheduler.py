import unittest
import logging
from src.core.task_scheduler import AdaptiveScheduler, TaskRequirements, NodeResources

class TestTaskScheduler(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.scheduler = AdaptiveScheduler()
        
        # Set up mock nodes with different resource profiles
        self.nodes = {
            'node1': {'cpu': 30.0, 'memory': 40.0},
            'node2': {'cpu': 70.0, 'memory': 60.0}
        }
        
        for node_id, resources in self.nodes.items():
            self.scheduler.register_node(
                node_id,
                NodeResources(
                    cpu_available=resources['cpu'],
                    memory_available=resources['memory'],
                    current_load=0.0,
                    node_id=node_id
                )
            )

    def test_task_scheduling_priority(self):
        """Test if tasks are scheduled based on priority."""
        # Create tasks with different priorities
        tasks = [
            TaskRequirements(
                cpu_needed=20.0,
                memory_needed=30.0,
                priority=1,
                model_name='model1',
                batch_size=1
            ),
            TaskRequirements(
                cpu_needed=20.0,
                memory_needed=30.0,
                priority=2,
                model_name='model2',
                batch_size=1
            )
        ]
        
        # Test each task
        for task in tasks:
            selected_node = self.scheduler.select_node(task)
            self.assertIsNotNone(selected_node)
        
        # Higher priority task should get better node (node2)
        selected_node = self.scheduler.select_node(tasks[1])  # Higher priority task
        self.assertEqual(selected_node, 'node2')

    def test_resource_constraints(self):
        """Test if scheduler respects resource constraints."""
        # Create a task with high resource requirements
        heavy_task = TaskRequirements(
            cpu_needed=80.0,
            memory_needed=90.0,
            priority=1,
            model_name='heavy_model',
            batch_size=1
        )
        
        # Should not find suitable node
        selected_node = self.scheduler.select_node(heavy_task)
        self.assertIsNone(selected_node)

    def test_load_balancing(self):
        """Test if tasks are distributed across nodes."""
        # Create multiple similar tasks
        tasks = [
            TaskRequirements(
                cpu_needed=20.0,
                memory_needed=20.0,
                priority=1,
                model_name=f'model{i}',
                batch_size=1
            )
            for i in range(4)
        ]
        
        # Track node assignments
        node_assignments = {'node1': 0, 'node2': 0}
        
        # Schedule all tasks and log decisions
        for i, task in enumerate(tasks):
            # Get scores for each node before selection
            scores = []
            for node_id, resources in self.scheduler.nodes.items():
                resource_score = (resources.cpu_available / task.cpu_needed + 
                                resources.memory_available / task.memory_needed) / 2
                load_score = 1 - resources.current_load
                task_count = self.scheduler.node_task_count[node_id]
                balance_score = 1 / (1 + task_count)
                
                total_score = (
                    0.3 * resource_score + 
                    0.2 * load_score + 
                    0.2 * 0.5 +  # Default perf_score
                    0.3 * balance_score
                )
                scores.append((node_id, total_score))
            
            self.logger.info(f"\nTask {i} scores before selection:")
            for node_id, score in scores:
                self.logger.info(f"{node_id}: {score:.3f} (tasks: {self.scheduler.node_task_count[node_id]})")
            
            selected_node = self.scheduler.select_node(task)
            self.assertIsNotNone(selected_node)
            node_assignments[selected_node] += 1
            
            self.logger.info(f"Selected node: {selected_node}")
            self.logger.info(f"Current assignments: {node_assignments}")
            
            # Simulate task execution
            self.scheduler.record_task_completion(selected_node, 10.0)
        
        self.logger.info(f"\nFinal node assignments: {node_assignments}")
        self.logger.info(f"Node task counts: {self.scheduler.node_task_count}")
        self.logger.info(f"Node loads: {[(n, r.current_load) for n, r in self.scheduler.nodes.items()]}")
        
        # Both nodes should have received tasks
        self.assertTrue(
            all(count > 0 for count in node_assignments.values()),
            f"Tasks were not distributed properly. Assignments: {node_assignments}"
        )

if __name__ == '__main__':
    unittest.main() 