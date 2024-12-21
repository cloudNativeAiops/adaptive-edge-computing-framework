import unittest
from src.core.resource_monitor import ResourceMonitor
from src.utils.docker_utils import DockerManager
import time

class TestResourceMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with a Docker container."""
        cls.docker_manager = DockerManager()
        cls.docker_manager.setup_network()
        cls.container_id = cls.docker_manager.create_edge_node(
            "test-node",
            cpu_limit=0.5,
            memory_limit="512m"
        )
        cls.resource_monitor = ResourceMonitor()
        # Wait for container to start
        time.sleep(2)

    def test_container_stats(self):
        """Test container resource statistics collection."""
        stats = self.resource_monitor.get_container_stats(self.container_id)
        
        # Check if all required metrics are present
        self.assertIn('cpu_usage', stats)
        self.assertIn('memory_usage', stats)
        self.assertIn('network_io', stats)
        
        # Check memory usage format
        self.assertIn('usage_mb', stats['memory_usage'])
        self.assertIn('limit_mb', stats['memory_usage'])
        self.assertIn('percentage', stats['memory_usage'])

    def test_host_stats(self):
        """Test host machine statistics collection."""
        stats = self.resource_monitor.get_host_stats()
        
        self.assertIn('cpu_percent', stats)
        self.assertIn('memory_percent', stats)
        self.assertIn('disk_usage', stats)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.docker_manager.cleanup()

if __name__ == '__main__':
    unittest.main() 