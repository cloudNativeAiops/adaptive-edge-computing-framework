import docker
from typing import Dict, List, Any
import logging
import os
import platform
import subprocess
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DockerManager:
    def __init__(self):
        """Initialize Docker client."""
        self.logger = logging.getLogger(__name__)
        
        # Configure Docker client based on OS
        if platform.system() == "Darwin":  # macOS
            # Get Docker context information
            try:
                # Try to get the current Docker context
                context_info = subprocess.check_output(
                    ["docker", "context", "inspect"],
                    stderr=subprocess.PIPE
                ).decode()
                self.logger.info(f"Docker context: {context_info}")
                
                # Try common Docker socket locations on macOS
                socket_paths = [
                    str(Path.home() / ".docker/run/docker.sock"),  # Docker Desktop default
                    "/var/run/docker.sock",  # Traditional location
                ]
                
                for socket_path in socket_paths:
                    try:
                        if os.path.exists(socket_path):
                            self.logger.info(f"Found Docker socket at {socket_path}")
                            self.client = docker.DockerClient(
                                base_url=f"unix://{socket_path}",
                                version='auto'
                            )
                            self.client.ping()
                            self.logger.info("Successfully connected to Docker")
                            break
                    except Exception as e:
                        self.logger.debug(f"Failed to connect using {socket_path}: {e}")
                else:
                    raise RuntimeError("No valid Docker socket found")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Docker: {e}")
                raise RuntimeError(
                    f"Could not connect to Docker. Error: {e}\n"
                    "Please ensure Docker Desktop is running and properly configured."
                )
        else:
            self.client = docker.from_env()

    def create_edge_node(self, node_name: str, cpu_limit: float, memory_limit: str) -> str:
        """Create a Docker container for edge node simulation."""
        try:
            # Check if Docker is running
            self._check_docker_running()
            
            # Create network if it doesn't exist
            self.setup_network()
            
            # Remove existing container with the same name if it exists
            try:
                existing_container = self.client.containers.get(node_name)
                self.logger.info(f"Removing existing container: {node_name}")
                existing_container.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Pull the image first
            self.logger.info("Pulling PyTorch image...")
            self.client.images.pull("pytorch/pytorch:latest")
            
            # Create and start the container
            self.logger.info(f"Creating container {node_name} with CPU limit {cpu_limit} and memory limit {memory_limit}")
            container = self.client.containers.run(
                image="pytorch/pytorch:latest",
                name=node_name,
                command="tail -f /dev/null",  # Keep container running
                cpu_count=1,
                cpu_period=100000,
                cpu_quota=int(cpu_limit * 100000),
                mem_limit=memory_limit,
                network="edge-net",
                detach=True
            )
            
            # Wait for container to be running
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                container.reload()  # Refresh container status
                if container.status == 'running':
                    self.logger.info(f"Container {node_name} is running")
                    break
                self.logger.info(f"Waiting for container to start... Status: {container.status}")
                time.sleep(1)
                retry_count += 1
            
            if container.status != 'running':
                self.logger.error(f"Container failed to start. Status: {container.status}")
                self.logger.error(f"Container logs: {container.logs().decode()}")
                raise RuntimeError(f"Container failed to start: {container.status}")
            
            return container.id

        except docker.errors.APIError as e:
            self.logger.error(f"Failed to create edge node: {e}")
            raise

    def _check_docker_running(self):
        """Check if Docker daemon is running."""
        try:
            self.client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Docker daemon is not running or not accessible. Error: {e}\n"
                "Please check Docker Desktop status and permissions."
            )

    def setup_network(self, network_name: str = "edge-net") -> None:
        """Create Docker network for edge nodes."""
        try:
            self.client.networks.create(
                network_name,
                driver="bridge",
                attachable=True
            )
            self.logger.info(f"Created network: {network_name}")
        except docker.errors.APIError as e:
            if "already exists" not in str(e):
                self.logger.error(f"Failed to create network: {e}")
                raise

    def get_container_info(self, container_id: str) -> Dict[str, Any]:
        """Get detailed container information."""
        try:
            container = self.client.containers.get(container_id)
            container.reload()  # Refresh container info
            
            info = {
                'id': container.id,
                'name': container.name,
                'status': container.status,
                'network': container.attrs['NetworkSettings']['Networks'],
                'state': container.attrs['State'],
                'config': container.attrs['Config']
            }
            
            self.logger.info(f"Container info: {info}")
            return info
            
        except docker.errors.NotFound:
            self.logger.error(f"Container not found: {container_id}")
            raise

    def cleanup(self, network_name: str = "edge-net") -> None:
        """Clean up Docker resources."""
        try:
            # Stop and remove containers
            containers = self.client.containers.list(
                all=True,  # Include stopped containers
                filters={'network': network_name}
            )
            for container in containers:
                self.logger.info(f"Removing container: {container.name}")
                container.remove(force=True)

            # Remove network
            try:
                network = self.client.networks.get(network_name)
                network.remove()
                self.logger.info(f"Removed network: {network_name}")
            except docker.errors.NotFound:
                pass  # Network doesn't exist, which is fine
                
        except docker.errors.APIError as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise 
