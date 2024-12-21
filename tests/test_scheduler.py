import pytest
from src.core.task_scheduler import AdaptiveScheduler, TaskRequirements, NodeResources

@pytest.fixture
def scheduler():
    return AdaptiveScheduler()

@pytest.fixture
def sample_task():
    return TaskRequirements(
        cpu_needed=0.5,
        memory_needed=512,
        priority=1,
        model_name="test_model",
        batch_size=1
    )

def test_node_registration(scheduler):
    node_resources = NodeResources(
        cpu_available=1.0,
        memory_available=1024,
        current_load=0.0,
        node_id="test_node"
    )
    
    scheduler.register_node("test_node", node_resources)
    assert "test_node" in scheduler.nodes
    assert scheduler.nodes["test_node"] == node_resources

def test_node_selection(scheduler, sample_task):
    # Register two nodes with different resources
    scheduler.register_node(
        "node1",
        NodeResources(1.0, 1024, 0.2, "node1")
    )
    scheduler.register_node(
        "node2",
        NodeResources(0.5, 512, 0.5, "node2")
    )
    
    # Node1 should be selected (better resources)
    selected_node = scheduler.select_node(sample_task)
    assert selected_node == "node1"

def test_performance_history(scheduler):
    scheduler.register_node(
        "test_node",
        NodeResources(1.0, 1024, 0.0, "test_node")
    )
    
    # Record some execution times
    execution_times = [10.0, 15.0, 12.0]
    for time in execution_times:
        scheduler.record_task_completion("test_node", time)
        
    stats = scheduler.get_node_stats()
    assert "test_node" in stats
    assert abs(stats["test_node"]["avg_execution_time"] - 12.33) < 0.1 