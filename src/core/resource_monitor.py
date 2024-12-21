import docker
import psutil
import time
from typing import Dict, Any
import os
import platform
from pathlib import Path
import logging

class ResourceMonitor:
    def __init__(self):
        """Initialize the resource monitor with Docker client."""
        self.logger = logging.getLogger(__name__)

        # Configure Docker client based on OS
        if platform.system() == "Darwin":  # macOS
            socket_path = str(Path.home() / ".docker/run/docker.sock")
            self.docker_client = docker.DockerClient(
                base_url=f"unix://{socket_path}",
                version='auto'
            )
        else:
            self.docker_client = docker.from_env()

        self.previous_cpu = {}
        self.previous_system = {}

    def get_container_stats(self, container_id: str, max_retries: int = 5) -> Dict[str, Any]:
        """Get resource statistics for a specific container."""
        for attempt in range(max_retries):
            try:
                container = self.docker_client.containers.get(container_id)
                if container.status != 'running':
                    self.logger.info(f"Waiting for container {container_id} to start...")
                    time.sleep(1)
                    continue

                stats = container.stats(stream=False)
                
                # Check if all required stats are available
                if not self._validate_stats(stats):
                    self.logger.info(f"Waiting for complete stats (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(1)
                    continue

                return {
                    'cpu_usage': self._calculate_cpu_percent(stats, container_id),
                    'memory_usage': self._calculate_memory_usage(stats),
                    'network_io': self._get_network_stats(stats)
                }
            except docker.errors.NotFound:
                self.logger.error(f"Container {container_id} not found")
                raise ValueError(f"Container {container_id} not found")
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error getting container stats: {e}")
                    # Return default stats instead of raising
                    return {
                        'cpu_usage': 0.0,
                        'memory_usage': {
                            'usage_mb': 0.0,
                            'limit_mb': 0.0,
                            'percentage': 0.0
                        },
                        'network_io': {
                            'rx_bytes': 0,
                            'tx_bytes': 0
                        }
                    }
                self.logger.info(f"Retrying stats collection (attempt {attempt + 1}/{max_retries})...")
                time.sleep(1)

    def _validate_stats(self, stats: Dict) -> bool:
        """Validate that all required stats are present."""
        required_keys = [
            ('cpu_stats', 'cpu_usage', 'total_usage'),
            ('cpu_stats', 'system_cpu_usage'),
            ('precpu_stats', 'cpu_usage', 'total_usage'),
            ('precpu_stats', 'system_cpu_usage'),
            ('memory_stats', 'usage'),
            ('memory_stats', 'limit'),
            ('networks',)
        ]
        
        for keys in required_keys:
            current = stats
            try:
                for key in keys:
                    current = current[key]
            except (KeyError, TypeError):
                return False
        return True

    def _calculate_cpu_percent(self, stats: Dict, container_id: str) -> float:
        """Calculate CPU usage percentage."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return round(cpu_percent, 2)
            return 0.0
        except KeyError as e:
            self.logger.warning(f"Missing CPU stat: {e}")
            return 0.0

    def _calculate_memory_usage(self, stats: Dict) -> Dict[str, float]:
        """Calculate memory usage statistics."""
        usage = stats['memory_stats']['usage']
        limit = stats['memory_stats']['limit']
        
        return {
            'usage_mb': round(usage / (1024 * 1024), 2),
            'limit_mb': round(limit / (1024 * 1024), 2),
            'percentage': round((usage / limit) * 100, 2)
        }

    def _get_network_stats(self, stats: Dict) -> Dict[str, float]:
        """Get network I/O statistics."""
        networks = stats['networks']
        return {
            'rx_bytes': sum(net['rx_bytes'] for net in networks.values()),
            'tx_bytes': sum(net['tx_bytes'] for net in networks.values())
        }

    def get_host_stats(self) -> Dict[str, Any]:
        """Get host machine resource statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        } 