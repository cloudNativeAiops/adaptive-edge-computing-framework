import asyncio
from concurrent.futures import ThreadPoolExecutor
from src.core.task_scheduler import TaskScheduler, NodeResources
from src.docker_manager import DockerManager
import json
import time
import docker
import subprocess

class DistributedExecutor:
    def __init__(self, partitioner, scheduler):
        self.partitioner = partitioner
        self.scheduler = scheduler
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.docker_manager = DockerManager()
        
    async def execute_parallel(self, input_data, available_nodes):
        # 1. 获取分区方案
        partitions = self.partitioner.partition_for_cluster(available_nodes)
        
        # 2. 创建每个节点的任务
        tasks = []
        for node_id, partition in partitions.items():
            task = self.executor.submit(
                self._execute_partition,
                node_id=node_id,
                partition=partition,
                input_data=input_data
            )
            tasks.append(task)
        
        # 3. 并行等待所有任务完成
        results = await asyncio.gather(*[
            asyncio.wrap_future(task) for task in tasks
        ])
        
        return self._aggregate_results(results)
        
    def _execute_partition(self, node_id, partition, input_data):
        """在单个节点上执行分区"""
        retries = 3
        for attempt in range(retries):
            try:
                container = self.docker_manager.get_container(node_id)
                return self._run_inference(container, partition, input_data)
            except ValueError as e:
                if "not found" in str(e) and attempt < retries - 1:
                    print(f"Container {node_id} not found, retrying... Attempt: {attempt+1}/{retries}")
                    time.sleep(5)  # 增加重试延迟
                else:
                    print(f"Failed to get container {node_id} after multiple retries.")
                    if node_id == "node3": # 针对 node3 增加日志
                        try:
                            # 获取并打印容器的详细状态
                            container = self.docker_manager.client.containers.get(self.docker_manager.containers[node_id])
                            print(f"Container {node_id} status: {container.attrs['State']}")

                            # 执行并打印 docker ps -a
                            print("--- docker ps -a ---")
                            process = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True)
                            print(process.stdout)

                            # 执行并打印 docker inspect node3
                            print(f"--- docker inspect {node_id} ---")
                            process = subprocess.run(["docker", "inspect", self.docker_manager.containers[node_id]], capture_output=True, text=True)
                            print(process.stdout)

                            # 打印 docker events (需要根据实际情况调整时间)
                            print(f"--- docker events --since '{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 600))}' ---") # 打印过去 10 分钟的事件
                            process = subprocess.run(["docker", "events", "--since", time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 600))], capture_output=True, text=True)
                            print(process.stdout)

                            # 执行并打印 docker logs node3
                            print(f"--- docker logs {node_id} ---")
                            process = subprocess.run(["docker", "logs", self.docker_manager.containers[node_id]], capture_output=True, text=True)
                            print(process.stdout)

                        except (docker.errors.NotFound, docker.errors.APIError, KeyError):
                            print(f"Could not retrieve status for container {node_id}.")
                        except Exception as e:
                            print(f"Error during detailed logging: {e}")
                    else:
                        try:
                            container = self.docker_manager.client.containers.get(self.docker_manager.containers[node_id])
                            print(f"Container {node_id} status: {container.attrs['State']}")
                        except (docker.errors.NotFound, docker.errors.APIError, KeyError):
                            print(f"Could not retrieve status for container {node_id}.")
                    raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
        
    def _run_inference(self, container, partition, input_data):
        """在容器中运行推理"""
        try:
            # 准备输入数据
            input_shape = tuple(input_data["input_shape"])  # 转换为元组
            
            # 构建推理命令
            inference_command = (
                f"python -c '"
                f"import torch; "
                f"import json; "
                f"input_tensor = torch.randn({input_shape}); "  # 使用元组形式的shape
                f"print(json.dumps({{"
                f"    \"partition_id\": \"{partition}\", "
                f"    \"input_shape\": {list(input_shape)}, "  # 转回列表以便JSON序列化
                f"    \"output\": \"simulation_output\""
                f"}}))"
                f"'"
            )
            
            # 执行推理
            exit_code, output = container.exec_run(
                inference_command,
                environment={"PYTHONPATH": "/app"}
            )
            
            if exit_code != 0:
                raise Exception(f"Inference failed with exit code {exit_code}: {output.decode()}")
                
            # 解析输出结果
            try:
                result = json.loads(output.decode().strip())
                return {"status": "success", "result": result}
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse inference output"}
                
        except Exception as e:
            print(f"Error during inference: {e}")
            return {"status": "error", "message": str(e)}
            
    def _aggregate_results(self, results):
        """聚合所有分区的结果"""
        # TODO: 实现结果聚合逻辑
        return {
            "aggregated_results": results,
            "status": "success"
        } 