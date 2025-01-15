import docker
import platform
import os
import time
import subprocess
from pathlib import Path

class DockerManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DockerManager, cls).__new__(cls)
            try:
                # check if the Docker daemon is running
                if not cls._is_docker_running():
                    raise RuntimeError("Docker daemon is not running")
                
                # try multiple connection methods
                connection_methods = [
                    # 1. environment variable method
                    lambda: docker.from_env(),
                    # 2. default path of Docker Desktop (macOS)
                    lambda: docker.DockerClient(base_url='unix:///Users/guilinzhang/.docker/run/docker.sock'),
                    # 3. standard Unix socket path
                    lambda: docker.DockerClient(base_url='unix:///var/run/docker.sock'),
                    # 4. TCP connection (if configured)
                    lambda: docker.DockerClient(base_url='tcp://localhost:2375'),
                ]
                
                last_error = None
                for connect_method in connection_methods:
                    try:
                        cls._instance.client = connect_method()
                        # verify the connection
                        cls._instance.client.ping()
                        print("Successfully connected to Docker daemon")
                        break
                    except Exception as e:
                        last_error = e
                        continue
                else:
                    raise RuntimeError(f"Failed to connect to Docker after trying all methods: {last_error}")
                
                # initialize necessary attributes
                cls._instance._containers = {}
                cls._instance._networks = {}
                cls._instance._images = {}
                cls._instance._cluster_config = {
                    'node_count': 3,
                    'base_image': 'python:3.10-slim',
                    'network_name': 'edge_network',
                    'container_prefix': 'edge_node_',
                    'startup_timeout': 30,
                    'health_check_interval': 1
                }
                
            except Exception as e:
                print(f"\nDetailed Docker connection error:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                cls._print_docker_debug_info()
                raise
                
        return cls._instance
    
    @staticmethod
    def _is_docker_running():
        """check if the Docker daemon is running"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def _print_docker_debug_info():
        """print Docker debug information"""
        print("\nDocker Debug Information:")
        
        # check common socket paths
        socket_paths = [
            "/var/run/docker.sock",
            os.path.expanduser("~/.docker/run/docker.sock"),
            os.path.expanduser("~/Library/Containers/com.docker.docker/Data/docker.sock")
        ]
        
        print("\nChecking Docker socket paths:")
        for path in socket_paths:
            exists = os.path.exists(path)
            readable = os.access(path, os.R_OK) if exists else False
            writable = os.access(path, os.W_OK) if exists else False
            print(f"Path: {path}")
            print(f"  Exists: {exists}")
            print(f"  Readable: {readable}")
            print(f"  Writable: {writable}")
            if exists:
                try:
                    stats = os.stat(path)
                    print(f"  Permissions: {oct(stats.st_mode)}")
                    print(f"  Owner: {stats.st_uid}")
                except Exception as e:
                    print(f"  Error getting stats: {e}")
        
        # check Docker environment variables
        print("\nDocker Environment Variables:")
        for var in ['DOCKER_HOST', 'DOCKER_CERT_PATH', 'DOCKER_TLS_VERIFY']:
            print(f"{var}: {os.environ.get(var, 'Not set')}")
        
        # check Docker version and information
        try:
            version_result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True
            )
            print("\nDocker Version:")
            print(version_result.stdout)
        except Exception as e:
            print(f"Failed to get Docker version: {e}")
        
        try:
            info_result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True
            )
            print("\nDocker Info:")
            print(info_result.stdout)
        except Exception as e:
            print(f"Failed to get Docker info: {e}") 

    def __init__(self):
        # define different resource configuration profiles
        self.resource_profiles = {
            'high': {
                'cpu_limit': 1.0,    # 1 core
                'memory_limit': '1g'  # 1GB
            },
            'medium': {
                'cpu_limit': 0.6,    # 0.6 core
                'memory_limit': '512m'  # 512MB
            },
            'low': {
                'cpu_limit': 0.4,    # 0.4 core
                'memory_limit': '512m'  # 512MB
            }
        }

    def create_edge_cluster(self, node_count=3):
        """dynamically create edge computing clusters"""
        try:
            print(f"Creating edge cluster with {node_count} nodes...")
            
            self._cleanup_existing_resources()
            
            # create network
            network_name = self._cluster_config['network_name']
            try:
                network = self.client.networks.create(
                    network_name,
                    driver="bridge",
                    check_duplicate=True
                )
                self._networks[network_name] = network
                print(f"Created network: {network_name}")
            except docker.errors.APIError as e:
                if 'already exists' in str(e):
                    network = self.client.networks.get(network_name)
                    self._networks[network_name] = network
                    print(f"Using existing network: {network_name}")
                else:
                    raise

            # assign resource configurations based on the number of nodes
            resource_assignments = []
            if node_count == 3:
                resource_assignments = ['high', 'medium', 'low']
            elif node_count == 4:
                resource_assignments = ['high', 'high', 'medium', 'low']
            elif node_count == 2:
                resource_assignments = ['high', 'medium']
            
            # create containers
            for i in range(node_count):
                node_id = f"{self._cluster_config['container_prefix']}{i+1}"
                profile = resource_assignments[i]
                resource_config = self.resource_profiles[profile]
                
                try:
                    # check and remove existing containers with the same name
                    try:
                        existing_container = self.client.containers.get(node_id)
                        existing_container.remove(force=True)
                        print(f"Removed existing container: {node_id}")
                    except docker.errors.NotFound:
                        pass

                    # create new container, add resource constraints
                    container = self.client.containers.run(
                        self._cluster_config['base_image'],
                        name=node_id,
                        command="tail -f /dev/null",
                        detach=True,
                        network=network_name,
                        remove=True,
                        environment={
                            "PYTHONUNBUFFERED": "1",
                            "NODE_ID": node_id,
                            "PATH": "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                        },
                        volumes={
                            '/var/run/docker.sock': {'bind': '/var/run/docker.sock', 'mode': 'rw'}
                        },
                        working_dir="/app",
                        tty=True,
                        stdin_open=True,
                        # add resource constraints
                        cpu_period=100000,
                        cpu_quota=int(100000 * resource_config['cpu_limit']),
                        mem_limit=resource_config['memory_limit']
                    )
                    
                    # step by step install dependencies, use CPU-only version to reduce volume
                    setup_commands = [
                        "apt-get update",
                        "apt-get install -y --no-install-recommends python3-pip",
                        "rm -rf /var/lib/apt/lists/*",  # execute cleanup commands separately
                        "apt-get clean",
                        "pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu"
                    ]
                    
                    for cmd in setup_commands:
                        print(f"Executing: {cmd}")
                        # use /bin/sh -c to execute commands
                        exit_code, output = container.exec_run(
                            cmd,
                            environment={"DEBIAN_FRONTEND": "noninteractive"},
                            user="root",  # ensure using root user to execute commands
                            workdir="/",  # execute commands in the root directory
                            privileged=True  # give necessary permissions
                        )
                        if exit_code != 0:
                            print(f"Command output: {output.decode()}")
                            raise RuntimeError(f"Container setup failed: {cmd}")
                        print(f"Successfully executed: {cmd}")
                    
                    self._containers[node_id] = container
                    print(f"Created container: {node_id}")
                    
                    if not self._wait_for_container_ready(node_id):
                        raise RuntimeError(f"Container {node_id} failed to start")
                    
                except Exception as e:
                    print(f"Error creating container {node_id}: {e}")
                    self._cleanup_cluster()
                    raise

            return self._containers
            
        except Exception as e:
            print(f"Error creating edge cluster: {e}")
            self._cleanup_cluster()
            raise

    def _cleanup_existing_resources(self):
        """clean up existing resources"""
        try:
            # get all containers
            containers = self.client.containers.list(all=True)
            for container in containers:
                if container.name.startswith(self._cluster_config['container_prefix']):
                    try:
                        container.remove(force=True)
                        print(f"Removed existing container: {container.name}")
                    except Exception as e:
                        print(f"Error removing container {container.name}: {e}")

            # get all networks
            networks = self.client.networks.list()
            for network in networks:
                if network.name == self._cluster_config['network_name']:
                    try:
                        network.remove()
                        print(f"Removed existing network: {network.name}")
                    except Exception as e:
                        print(f"Error removing network {network.name}: {e}")
                        
            # clean up internal state
            self._containers = {}
            self._networks = {}
            
        except Exception as e:
            print(f"Error during cleanup of existing resources: {e}")
            raise

    def cleanup(self, container_name=None):
        """
        clean up resources
        Args:
            container_name: optional, specify the container name to clean up. If None, clean up all resources
        """
        try:
            if container_name:
                # clean up the specified container
                if container_name in self._containers:
                    try:
                        container = self._containers[container_name]
                        container.stop()
                        container.remove(force=True)
                        del self._containers[container_name]
                        print(f"Removed container: {container_name}")
                    except Exception as e:
                        print(f"Error removing container {container_name}: {e}")
            else:
                # clean up all resources
                self._cleanup_cluster()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_container_metrics(self, container_id):
        """get the resource usage metrics of the container"""
        try:
            # initialize default values
            cpu_usage = 0.0
            memory_usage = 0.0
            
            # get the container object
            container = self._containers.get(container_id)
            if not container:
                print(f"Container {container_id} not found in managed containers")
                return {
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_mb': memory_usage
                }

            # get the container statistics
            stats = container.stats(stream=False)
            
            # calculate CPU usage rate
            if all(key in stats for key in ['cpu_stats', 'precpu_stats']):
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * 100.0

            # calculate memory usage (MB)
            if 'memory_stats' in stats:
                memory_usage = stats['memory_stats'].get('usage', 0) / (1024 * 1024)

            return {
                'cpu_usage_percent': cpu_usage,
                'memory_usage_mb': memory_usage
            }
            
        except Exception as e:
            print(f"Error getting metrics for container {container_id}: {e}")
            return {
                'cpu_usage_percent': 0.0,
                'memory_usage_mb': 0.0
            }

    def get_container_status(self, container_id):
        """get the container status"""
        container = self._containers.get(container_id)
        if not container:
            return None
        container.reload()  # refresh the container status
        return container.status

    def _wait_for_container_ready(self, node_id, max_retries=30):
        """wait for the container to be ready"""
        retries = 0
        while retries < max_retries:
            try:
                container = self.client.containers.get(node_id)
                container.reload()  # refresh the container status
                status = container.status
                print(f"Container {node_id} status: {status}")
                
                if status == 'running':
                    # execute a simple health check
                    try:
                        exit_code, output = container.exec_run(
                            'python3 -c "print(\'container ready\')"',
                            stderr=True
                        )
                        if exit_code == 0:
                            print(f"Container {node_id} is ready")
                            return True
                    except Exception as e:
                        print(f"Health check failed: {e}")
                
                elif status in ['exited', 'dead']:
                    print(f"Container {node_id} failed with status: {status}")
                    # get the container logs to help diagnose
                    logs = container.logs().decode('utf-8')
                    print(f"Container logs:\n{logs}")
                    return False
                
                time.sleep(self._cluster_config['health_check_interval'])
                retries += 1
                
            except docker.errors.NotFound:
                print(f"Container {node_id} not found")
                return False
            except Exception as e:
                print(f"Error checking container status: {e}")
                return False
                
        print(f"Timeout waiting for container {node_id}")
        return False

    def _cleanup_cluster(self):
        """clean up all cluster resources"""
        try:
            # stop and delete all containers
            for container_id, container in list(self._containers.items()):
                try:
                    container.stop()
                    container.remove(force=True)
                    print(f"Removed container: {container_id}")
                except Exception as e:
                    print(f"Error removing container {container_id}: {e}")
                finally:
                    self._containers.pop(container_id, None)

            # delete all networks
            for network_name, network in list(self._networks.items()):
                try:
                    network.remove()
                    print(f"Removed network: {network_name}")
                except Exception as e:
                    print(f"Error removing network {network_name}: {e}")
                finally:
                    self._networks.pop(network_name, None)
            
        except Exception as e:
            print(f"Error during cluster cleanup: {e}")
        finally:
            # ensure the state is cleaned up
            self._containers = {}
            self._networks = {}

    def get_container(self, node_id):
        """get the container of the specified node"""
        if node_id not in self._containers:
            raise KeyError(f"Container {node_id} not found")
        return self._containers[node_id]

    def get_all_containers(self):
        """get all containers"""
        return self._containers

    def get_container_stats(self, node_id):
        """get the resource usage statistics of the container"""
        container = self.get_container(node_id)
        return container.stats(stream=False) 