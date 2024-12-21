import logging
from typing import Dict, Any, List
import torch
import os
from dataclasses import dataclass
from src.core.task_scheduler import AdaptiveScheduler, TaskRequirements
from src.utils.docker_utils import DockerManager

@dataclass
class DeploymentConfig:
    model_name: str
    model_version: str
    resource_requirements: TaskRequirements
    partition_ids: List[str] = None
    batch_size: int = 1
    optimization_level: str = "O1"

class ModelDeployer:
    def __init__(self, scheduler: AdaptiveScheduler, docker_manager: DockerManager):
        self.logger = logging.getLogger(__name__)
        self.scheduler = scheduler
        self.docker_manager = docker_manager
        self.deployed_models: Dict[str, Dict[str, Any]] = {}
        
    def deploy_model(self, model_path: str, config: DeploymentConfig) -> str:
        """Deploy a model to the edge environment."""
        try:
            # 1. Select target node
            node_id = self.scheduler.select_node(config.resource_requirements)
            if not node_id:
                raise RuntimeError("No suitable node found for deployment")
                
            # 2. Optimize model for target platform
            optimized_model = self._optimize_model(model_path, config)
            
            # 3. Deploy to container
            deployment_id = self._deploy_to_container(
                optimized_model, 
                node_id, 
                config
            )
            
            # 4. Record deployment
            self.deployed_models[deployment_id] = {
                'node_id': node_id,
                'config': config,
                'status': 'active'
            }
            
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
            
    def _optimize_model(self, model_path: str, config: DeploymentConfig) -> str:
        """Optimize model for deployment."""
        model = torch.load(model_path)
        
        if config.optimization_level == "O1":
            # Basic optimizations
            model.eval()
            model = torch.jit.script(model)
        elif config.optimization_level == "O2":
            # Advanced optimizations
            model.eval()
            model = torch.jit.script(model)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        # Save optimized model
        optimized_path = f"{os.path.dirname(model_path)}/optimized_{config.model_name}.pt"
        torch.jit.save(model, optimized_path)
        return optimized_path
        
    def _deploy_to_container(self, model_path: str, node_id: str, 
                           config: DeploymentConfig) -> str:
        """Deploy model to target container."""
        # Copy model to container
        container = self.docker_manager.get_container(node_id)
        container.exec_run(f"mkdir -p /models/{config.model_name}")
        
        with open(model_path, 'rb') as f:
            container.put_archive(
                f"/models/{config.model_name}", 
                f.read()
            )
            
        # Start model server
        container.exec_run(
            f"python -m torch.distributed.run "
            f"--nproc_per_node=1 "
            f"model_server.py "
            f"--model_path=/models/{config.model_name} "
            f"--batch_size={config.batch_size}"
        )
        
        return f"{node_id}_{config.model_name}_{config.model_version}"
        
    def undeploy_model(self, deployment_id: str):
        """Remove a deployed model."""
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_id} not found")
            
        deployment = self.deployed_models[deployment_id]
        node_id = deployment['node_id']
        
        # Stop model server
        container = self.docker_manager.get_container(node_id)
        container.exec_run("pkill -f model_server.py")
        
        # Clean up model files
        container.exec_run(f"rm -rf /models/{deployment['config'].model_name}")
        
        # Update records
        deployment['status'] = 'inactive'
        
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployed model."""
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_id} not found")
            
        deployment = self.deployed_models[deployment_id]
        node_id = deployment['node_id']
        
        # Get container stats
        container = self.docker_manager.get_container(node_id)
        stats = container.stats(stream=False)
        
        return {
            'status': deployment['status'],
            'node_id': node_id,
            'resource_usage': stats,
            'config': deployment['config']
        } 