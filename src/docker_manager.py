import docker
import logging
from typing import Optional

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.logger = logging.getLogger(__name__)
        
    def create_edge_node(self, 
                        name: str, 
                        cpu_limit: float = 0.5, 
                        memory_limit: str = "512m") -> str:
        """创建一个模拟边缘设备的Docker容器"""
        try:
            self.cleanup(name)
            
            print(f"Creating container {name}...")
            
            container = self.client.containers.run(
                "python:3.10-slim",
                name=name,
                command="tail -f /dev/null",
                detach=True,
                cpu_period=100000,
                cpu_quota=int(cpu_limit * 100000),
                mem_limit=memory_limit,
                shm_size="2g",
                volumes={
                    str(self.get_project_root()): {
                        'bind': '/app',
                        'mode': 'rw'
                    }
                },
                working_dir="/app",
                environment={
                    "PYTHONPATH": "/app",
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": "",
                    "NO_CUDA": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1"
                }
            )
            
            print("Installing minimal dependencies for edge device simulation...")
            # 首先安装固定版本的numpy
            exit_code, output = container.exec_run(
                "pip install --no-cache-dir 'numpy<2.0'",
                stream=True
            )
            for line in output:
                print(line.decode().strip())
            
            # 然后安装CPU版本的PyTorch
            exit_code, output = container.exec_run(
                "pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html",
                stream=True
            )
            for line in output:
                print(line.decode().strip())
            
            # 安装其他必要的轻量级依赖
            exit_code, output = container.exec_run(
                "pip install --no-cache-dir pillow",
                stream=True
            )
            for line in output:
                print(line.decode().strip())
            
            print("Verifying setup...")
            exit_code, output = container.exec_run(
                "python -c 'import numpy; import torch; print(f\"NumPy version: {numpy.__version__}\"); print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\")'"
            )
            print(output.decode())
            
            print(f"Edge device simulator {name} ready!")
            return container.id
            
        except Exception as e:
            print(f"Error creating edge device simulator: {e}")
            raise
    
    def cleanup(self, container_name: Optional[str] = None):
        """清理Docker容器"""
        try:
            if container_name:
                # 清理指定容器
                try:
                    container = self.client.containers.get(container_name)
                    container.remove(force=True)
                    self.logger.info(f"Removed container: {container_name}")
                except docker.errors.NotFound:
                    pass
            else:
                # 清理所有相关容器
                containers = self.client.containers.list(
                    all=True,
                    filters={"name": "eval-node"}
                )
                for container in containers:
                    container.remove(force=True)
                    self.logger.info(f"Removed container: {container.name}")
                    
        except docker.errors.APIError as e:
            self.logger.error(f"Failed to cleanup containers: {e}")
            raise
    
    def get_project_root(self) -> str:
        """获取项目根目录"""
        import os
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 