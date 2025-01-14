import docker
import logging
from typing import Optional
import time

class DockerManager:
    _instance = None
    _containers = {}  # 类级别的容器映射
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DockerManager, cls).__new__(cls)
            cls._instance.client = docker.from_env()
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def __init__(self):
        # 确保不重复初始化
        pass
        
    def get_container(self, node_id: str):
        """获取指定节点ID的容器"""
        if node_id in self._containers:  # 使用类变量
            try:
                return self.client.containers.get(self._containers[node_id])
            except docker.errors.NotFound:
                del self._containers[node_id]  # 使用类变量
                raise ValueError(f"Container for node {node_id} not found or was removed")
        raise ValueError(f"Container for node {node_id} not found")
    
    def create_edge_node(self, name: str, cpu_limit: float = 0.5, memory_limit: str = "512m") -> str:
        """创建一个模拟边缘设备的Docker容器"""
        try:
            # 确保清理同名容器
            self.cleanup(name)
            
            print(f"Creating container {name}...")
            
            # 只使用 cpu_period/cpu_quota 方式限制 CPU
            container = self.client.containers.run(
                "python:3.10-slim",
                name=name,
                detach=True,
                cpu_period=100000,  # 微秒
                cpu_quota=int(cpu_limit * 100000),  # 转换为配额
                mem_limit=memory_limit,
                command="tail -f /dev/null"
            )
            
            # 验证资源限制是否生效
            inspect_data = self.client.api.inspect_container(container.id)
            host_config = inspect_data['HostConfig']
            print(f"Container {name} resource limits:")
            print(f"CPU Period: {host_config.get('CpuPeriod', 'N/A')}")
            print(f"CPU Quota: {host_config.get('CpuQuota', 'N/A')}")
            print(f"Memory: {host_config.get('Memory', 0)/1024/1024:.0f}MB")
            
            # 使用类变量存储容器ID
            self._containers[name] = container.id
            print(f"Stored container {name} with ID {container.id} in containers map")
            print(f"Current containers map: {self._containers}")
            
            # 安装依赖
            self._install_dependencies(container)
            
            # 添加CPU负载测试
            print(f"Adding CPU load test to container {name}...")
            cpu_load_command = (
                "python -c '"
                "import time; "
                "def cpu_load(): "
                "    while True: "
                "        x = 0; "
                "        for i in range(1000000): x += i; "
                "cpu_load()' &"  # 在后台运行
            )
            container.exec_run(cpu_load_command, detach=True)
            
            # 等待一段时间让CPU负载生效
            time.sleep(2)
            
            print(f"Container {name} created successfully with ID: {container.id}")
            return container.id
            
        except Exception as e:
            print(f"Error creating edge device simulator: {e}")
            raise
            
    def create_edge_cluster(self):
        """Create edge device cluster with better error handling"""
        nodes = {
            "node1": {"cpu": 0.2, "memory": "256m"},
            "node2": {"cpu": 0.5, "memory": "512m"},
            "node3": {"cpu": 1.0, "memory": "1g"}
        }
        
        # First ensure no existing containers
        self.cleanup()
        
        created_containers = {}
        try:
            for name, resources in nodes.items():
                container_id = self.create_edge_node(
                    name=name,
                    cpu_limit=resources["cpu"],
                    memory_limit=resources["memory"]
                )
                created_containers[name] = container_id
                
                # 给容器一些时间来启动和安装依赖
                time.sleep(2)
                
                # Wait for container to be ready before proceeding
                if not self._wait_for_container_ready(name):
                    raise Exception(f"Container {name} failed to become ready")
                    
            print("All containers in cluster are ready!")
            return created_containers
            
        except Exception as e:
            print(f"Error creating cluster: {e}")
            # Only cleanup containers that were created in this attempt
            for name in created_containers:
                self.cleanup(name)
            raise
            
    def _wait_for_container_ready(self, node_id: str, timeout: int = 180, max_retries: int = 5):
        """Wait for container to be ready with better error handling and retries"""
        print(f"Waiting for container {node_id} to be ready...")
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                container = self.get_container(node_id)
                if container.status == "running":
                    # 首先检查基本的Python环境
                    exit_code, output = container.exec_run(
                        "python -c 'print(\"Basic Python check\")'",
                    )
                    
                    if exit_code == 0:
                        print(f"Container {node_id} basic check passed")
                        
                        # 然后检查依赖项
                        exit_code, output = container.exec_run(
                            "python -c 'import numpy; import torch; print(\"Dependencies check passed\")'"
                        )
                        
                        if exit_code == 0:
                            print(f"Container {node_id} is fully ready!")
                            return True
                        else:
                            print(f"Dependencies not yet ready in {node_id}, retrying...")
                            print(f"Output: {output.decode()}")  # 添加更多诊断信息
                    
                    print(f"Container {node_id} not ready, attempt {retry_count + 1}/{max_retries}")
                    time.sleep(5)  # 增加等待时间，给依赖安装更多时间
                    retry_count += 1
                    
                else:
                    print(f"Container {node_id} status: {container.status}")
                    time.sleep(5)
                    retry_count += 1
                    
            except Exception as e:
                print(f"Error checking container {node_id} readiness: {e}")
                # 添加容器状态诊断
                try:
                    container = self.get_container(node_id)
                    print(f"Container status: {container.status}")
                    print(f"Container logs:")
                    print(container.logs().decode())
                except Exception as inner_e:
                    print(f"Failed to get container diagnostics: {inner_e}")
                
                time.sleep(5)
                retry_count += 1
        
        print(f"Container {node_id} failed to become ready after {max_retries} attempts")
        return False

    def _install_dependencies(self, container):
        """安装容器所需的依赖"""
        print("Installing minimal dependencies for edge device simulation...")
        
        # 首先更新 pip
        try:
            print("Updating pip...")
            result = container.exec_run(
                "python -m pip install --upgrade pip",
                stream=False
            )
            if result.exit_code != 0:
                print("Warning: Failed to update pip")
        except Exception as e:
            print(f"Warning: Failed to update pip: {e}")
        
        # 系统更新和安装必要工具
        system_commands = [
            "apt-get clean",
            "rm -rf /var/lib/apt/lists/*",
            "apt-get update -y",  # 移除重定向
            "apt-get install -y build-essential"  # 移除重定向
        ]
        
        try:
            for cmd in system_commands:
                print(f"Executing: {cmd}")
                result = container.exec_run(
                    cmd,
                    stream=False,  # 使用 stream=False 来减少输出
                    environment={
                        "DEBIAN_FRONTEND": "noninteractive",
                        "PYTHONUNBUFFERED": "1"
                    }
                )
                if result.exit_code != 0:
                    print(f"Command failed with exit code {result.exit_code}")
                    print(f"Output: {result.output.decode()}")
                    raise Exception(f"Failed to execute: {cmd}")
            
            # Python 依赖安装
            dependencies = [
                ("numpy<2.0", "pip install --no-cache-dir --quiet 'numpy<2.0'"),
                ("torch and torchvision", 
                 "pip install --no-cache-dir --quiet torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html"),
                ("pillow", "pip install --no-cache-dir --quiet pillow")
            ]
            
            for dep_name, cmd in dependencies:
                print(f"Installing {dep_name}...")
                result = container.exec_run(
                    cmd,
                    stream=False,
                    environment={"PYTHONUNBUFFERED": "1"}
                )
                if result.exit_code != 0:
                    print(f"Failed to install {dep_name}")
                    print(f"Output: {result.output.decode()}")
                    raise Exception(f"Failed to install {dep_name}")
            
            # 验证安装
            print("Verifying installations...")
            verification_command = (
                "python -c '"
                "import numpy; print(f\"numpy {numpy.__version__}\"); "
                "import torch; print(f\"torch {torch.__version__}\"); "
                "import torchvision; print(f\"torchvision {torchvision.__version__}\"); "
                "import PIL; print(f\"pillow {PIL.__version__}\"); "
                "'"
            )
            result = container.exec_run(verification_command, stream=False)
            print(result.output.decode())
            
            if result.exit_code != 0:
                raise Exception("Dependencies verification failed")
                
            print("All dependencies installed and verified successfully!")
            
        except Exception as e:
            print(f"Error during dependency installation: {e}")
            raise
    
    def cleanup(self, container_name: Optional[str] = None):
        """清理Docker容器"""
        try:
            if container_name:
                # 清理指定容器
                try:
                    container = self.client.containers.get(container_name)
                    container.remove(force=True)
                    print(f"Removed container: {container_name}")
                except docker.errors.NotFound:
                    print(f"Container {container_name} not found")
            else:
                # 清理所有相关容器
                containers = self.client.containers.list(
                    all=True,
                    filters={"name": "node"}  # 修改过滤器以匹配我们的容器名称
                )
                for container in containers:
                    try:
                        container.remove(force=True)
                        print(f"Removed container: {container.name}")
                    except Exception as e:
                        print(f"Error removing container {container.name}: {e}")
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
            raise
    
    def get_project_root(self) -> str:
        """获取项目根目录"""
        import os
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
    
    def get_container_stats(self, container_id):
        """获取容器详细资源使用统计"""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # CPU使用率
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # 内存使用率
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory_percent, 2)
            }
        except Exception as e:
            print(f"Error getting container stats: {e}")
            return {'cpu_percent': 0, 'memory_percent': 0} 