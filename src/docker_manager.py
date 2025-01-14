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
                # 检查 Docker 守护进程是否运行
                if not cls._is_docker_running():
                    raise RuntimeError("Docker daemon is not running")
                
                # 尝试多种连接方式
                connection_methods = [
                    # 1. 环境变量方式
                    lambda: docker.from_env(),
                    # 2. Docker Desktop 默认路径 (macOS)
                    lambda: docker.DockerClient(base_url='unix:///Users/guilinzhang/.docker/run/docker.sock'),
                    # 3. 标准 Unix socket 路径
                    lambda: docker.DockerClient(base_url='unix:///var/run/docker.sock'),
                    # 4. TCP 连接 (如果配置了)
                    lambda: docker.DockerClient(base_url='tcp://localhost:2375'),
                ]
                
                last_error = None
                for connect_method in connection_methods:
                    try:
                        cls._instance.client = connect_method()
                        # 验证连接
                        cls._instance.client.ping()
                        print("Successfully connected to Docker daemon")
                        break
                    except Exception as e:
                        last_error = e
                        continue
                else:
                    raise RuntimeError(f"Failed to connect to Docker after trying all methods: {last_error}")
                
                # 初始化必要的属性
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
        """检查 Docker 守护进程是否运行"""
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
        """打印 Docker 调试信息"""
        print("\nDocker Debug Information:")
        
        # 检查常见的 socket 路径
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
        
        # 检查 Docker 环境变量
        print("\nDocker Environment Variables:")
        for var in ['DOCKER_HOST', 'DOCKER_CERT_PATH', 'DOCKER_TLS_VERIFY']:
            print(f"{var}: {os.environ.get(var, 'Not set')}")
        
        # 检查 Docker 版本和信息
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

    def create_edge_cluster(self, node_count=3):
        """创建边缘计算集群"""
        try:
            print(f"Creating edge cluster with {node_count} nodes...")
            
            self._cleanup_existing_resources()
            
            # 创建网络
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

            # 创建容器
            for i in range(node_count):
                node_id = f"{self._cluster_config['container_prefix']}{i+1}"
                try:
                    # 检查并删除同名容器
                    try:
                        existing_container = self.client.containers.get(node_id)
                        existing_container.remove(force=True)
                        print(f"Removed existing container: {node_id}")
                    except docker.errors.NotFound:
                        pass

                    # 创建新容器，使用更小的依赖包
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
                        stdin_open=True
                    )
                    
                    # 分步安装依赖，使用 CPU-only 版本减小体积
                    setup_commands = [
                        "apt-get update",
                        "apt-get install -y --no-install-recommends python3-pip",
                        "rm -rf /var/lib/apt/lists/*",  # 分开执行清理命令
                        "apt-get clean",
                        "pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu"
                    ]
                    
                    for cmd in setup_commands:
                        print(f"Executing: {cmd}")
                        # 使用 /bin/sh -c 来执行命令
                        exit_code, output = container.exec_run(
                            cmd,
                            environment={"DEBIAN_FRONTEND": "noninteractive"},
                            user="root",  # 确保使用 root 用户执行命令
                            workdir="/",  # 在根目录执行命令
                            privileged=True  # 给予必要的权限
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
        """清理已存在的资源"""
        try:
            # 获取所有容器
            containers = self.client.containers.list(all=True)
            for container in containers:
                if container.name.startswith(self._cluster_config['container_prefix']):
                    try:
                        container.remove(force=True)
                        print(f"Removed existing container: {container.name}")
                    except Exception as e:
                        print(f"Error removing container {container.name}: {e}")

            # 获取所有网络
            networks = self.client.networks.list()
            for network in networks:
                if network.name == self._cluster_config['network_name']:
                    try:
                        network.remove()
                        print(f"Removed existing network: {network.name}")
                    except Exception as e:
                        print(f"Error removing network {network.name}: {e}")
                        
            # 清理内部状态
            self._containers = {}
            self._networks = {}
            
        except Exception as e:
            print(f"Error during cleanup of existing resources: {e}")
            raise

    def cleanup(self, container_name=None):
        """
        清理资源
        Args:
            container_name: 可选，指定要清理的容器名称。如果为None，清理所有资源
        """
        try:
            if container_name:
                # 清理指定容器
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
                # 清理所有资源
                self._cleanup_cluster()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_container_metrics(self, container_id):
        """获取容器的资源使用指标"""
        try:
            # 初始化默认值
            cpu_usage = 0.0
            memory_usage = 0.0
            
            # 获取容器对象
            container = self._containers.get(container_id)
            if not container:
                print(f"Container {container_id} not found in managed containers")
                return {
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_mb': memory_usage
                }

            # 获取容器统计信息
            stats = container.stats(stream=False)
            
            # 计算 CPU 使用率
            if all(key in stats for key in ['cpu_stats', 'precpu_stats']):
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * 100.0

            # 计算内存使用量（MB）
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
        """获取容器状态"""
        container = self._containers.get(container_id)
        if not container:
            return None
        container.reload()  # 刷新容器状态
        return container.status

    def _wait_for_container_ready(self, node_id, max_retries=30):
        """等待容器准备就绪"""
        retries = 0
        while retries < max_retries:
            try:
                container = self.client.containers.get(node_id)
                container.reload()  # 刷新容器状态
                status = container.status
                print(f"Container {node_id} status: {status}")
                
                if status == 'running':
                    # 执行简单的健康检查
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
                    # 获取容器日志以帮助诊断
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
        """清理所有集群资源"""
        try:
            # 停止并删除所有容器
            for container_id, container in list(self._containers.items()):
                try:
                    container.stop()
                    container.remove(force=True)
                    print(f"Removed container: {container_id}")
                except Exception as e:
                    print(f"Error removing container {container_id}: {e}")
                finally:
                    self._containers.pop(container_id, None)

            # 删除所有网络
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
            # 确保状态被清理
            self._containers = {}
            self._networks = {}

    def get_container(self, node_id):
        """获取指定节点的容器"""
        if node_id not in self._containers:
            raise KeyError(f"Container {node_id} not found")
        return self._containers[node_id]

    def get_all_containers(self):
        """获取所有容器"""
        return self._containers

    def get_container_stats(self, node_id):
        """获取容器的资源使用统计"""
        container = self.get_container(node_id)
        return container.stats(stream=False) 