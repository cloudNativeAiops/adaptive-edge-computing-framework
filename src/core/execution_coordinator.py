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
        # 1. partition the model
        partitions = self.partitioner.partition(
            model=self.model,
            available_nodes=available_nodes
        )
        
        # 2. schedule tasks
        schedule = self.scheduler.schedule_tasks(
            tasks=partitions,
            available_nodes=available_nodes
        )
        
        # 2. create tasks for each node
        tasks = []
        for node_id, partition in partitions.items():
            task = self.executor.submit(
                self._execute_partition,
                node_id=node_id,
                partition=partition,
                input_data=input_data
            )
            tasks.append(task)
        
        # 3. parallel wait for all tasks to complete
        try:
            results = await asyncio.gather(*[
                asyncio.wrap_future(task) for task in tasks
            ], return_exceptions=True)
            
            # if a node fails, it can be rescheduled to other nodes
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    backup_result = await self._retry_on_different_node(
                        tasks[i], 
                        exclude_nodes=[nodes[i]]
                    )
                    results[i] = backup_result
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        
        return self._aggregate_results(results)
        
    def _execute_partition(self, node_id, partition, input_data):
        """execute a partition on a single node"""
        retries = 3
        for attempt in range(retries):
            try:
                container = self.docker_manager.get_container(node_id)
                return self._run_inference(container, partition, input_data)
            except ValueError as e:
                if "not found" in str(e) and attempt < retries - 1:
                    print(f"Container {node_id} not found, retrying... Attempt: {attempt+1}/{retries}")
                    time.sleep(5)  # increase retry delay
                else:
                    print(f"Failed to get container {node_id} after multiple retries.")
                    if node_id == "node3": # for node3, add more logs
                        try:
                            # get and print the detailed status of the container
                            container = self.docker_manager.client.containers.get(self.docker_manager.containers[node_id])
                            print(f"Container {node_id} status: {container.attrs['State']}")

                            # execute and print docker ps -a
                            print("--- docker ps -a ---")
                            process = subprocess.run(["docker", "ps", "-a"], capture_output=True, text=True)
                            print(process.stdout)

                            # execute and print docker inspect node3
                            print(f"--- docker inspect {node_id} ---")
                            process = subprocess.run(["docker", "inspect", self.docker_manager.containers[node_id]], capture_output=True, text=True)
                            print(process.stdout)

                            # print docker events (need to adjust the time according to the actual situation)
                            print(f"--- docker events --since '{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 600))}' ---") # print the events of the last 10 minutes
                            process = subprocess.run(["docker", "events", "--since", time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 600))], capture_output=True, text=True)
                            print(process.stdout)

                            # execute and print docker logs node3
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
        """run inference in the container"""
        try:
            # prepare input data
            input_shape = tuple(input_data["input_shape"])
            
            # add actual computation load
            inference_command = (
                f"python -c '"
                f"import torch; "
                f"import json; "
                f"import time; "
                f"input_tensor = torch.randn({input_shape}); "
                f"# add computation load "
                f"for _ in range(10000): "  # increase to 10000 times
                f"    result = torch.nn.functional.relu(input_tensor); "
                f"    result = torch.matmul(result, result.t());"  # add matrix multiplication
                f"    input_tensor = result; "
                f"print(json.dumps({{"
                f"    \"partition_id\": \"{partition}\", "
                f"    \"input_shape\": {list(input_shape)}, "
                f"    \"output\": \"simulation_output\""
                f"}}))"
                f"'"
            )
            
            # execute inference
            exit_code, output = container.exec_run(
                inference_command,
                environment={"PYTHONPATH": "/app"}
            )
            
            if exit_code != 0:
                raise Exception(f"Inference failed with exit code {exit_code}: {output.decode()}")
                
            # parse the output result
            try:
                result = json.loads(output.decode().strip())
                return {"status": "success", "result": result}
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse inference output"}
                
        except Exception as e:
            print(f"Error during inference: {e}")
            return {"status": "error", "message": str(e)}
            
    def _aggregate_results(self, results):
        """aggregate the results of all partitions"""
        # TODO: implement the result aggregation logic
        return {
            "aggregated_results": results,
            "status": "success"
        } 