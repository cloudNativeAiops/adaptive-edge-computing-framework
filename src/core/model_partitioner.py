from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from src.core.task_scheduler import NodeResources
import time

class ModelPartitioner:
    def __init__(self, model: nn.Module):
        """Initialize the model partitioner."""
        self.model = model

class ResourceAwarePartitioner(ModelPartitioner):
    def __init__(self, model: nn.Module):
        """Initialize the resource-aware partitioner.
        
        Args:
            model: PyTorch model to be partitioned
        """
        super().__init__(model=model)  # explicitly call the parent class initialization method
        
    def partition_for_cluster(self, available_nodes: Dict[str, NodeResources]):
        """partition the model based on the cluster resources"""
        # 1. analyze the computation capabilities of each node
        node_capabilities = self._analyze_node_capabilities(available_nodes)
        
        # 2. distribute the model layers based on the computation capabilities
        partitions = self._distribute_layers(node_capabilities)
        
        # 3. optimize the partition scheme, considering communication overhead
        optimized_partitions = self._optimize_partitions(partitions)
        
        return optimized_partitions

    def _analyze_node_capabilities(self, nodes):
        """analyze the computation capabilities of each node"""
        capabilities = {}
        for node_id, resources in nodes.items():
            compute_power = resources.cpu_available * 100  # simplified computation capability evaluation
            capabilities[node_id] = compute_power
        return capabilities
        
    def _distribute_layers(self, node_capabilities: Dict[str, float]) -> Dict[str, List[str]]:
        """distribute the model layers based on the computation capabilities of each node"""
        total_capability = sum(node_capabilities.values())
        layers = self._analyze_model_layers()
        total_layers = len(layers)
        
        # initialize the partition results
        partitions = {node_id: [] for node_id in node_capabilities.keys()}
        
        # distribute the layers based on the computation capabilities ratio
        current_layer = 0
        for node_id, capability in node_capabilities.items():
            # calculate the number of layers to be allocated to this node
            layer_count = int((capability / total_capability) * total_layers)
            if current_layer < total_layers:
                end_layer = min(current_layer + layer_count, total_layers)
                partitions[node_id] = [
                    layer["name"] for layer in layers[current_layer:end_layer]
                ]
                current_layer = end_layer
        
        # handle the remaining layers (if any)
        while current_layer < total_layers:
            # assign the remaining layers to the node with the strongest computation capability
            strongest_node = max(node_capabilities.items(), key=lambda x: x[1])[0]
            partitions[strongest_node].append(layers[current_layer]["name"])
            current_layer += 1
            
        return partitions
        
    def _analyze_model_layers(self):
        """analyze the layer structure of the model"""
        
        layers = []
        for name, module in self.model.named_modules():
            # skip the container module (e.g. Sequential)
            if len(list(module.children())) > 0:
                continue
                
            # calculate the number of parameters
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "params": params,
                # can add more layer features
                "input_channels": getattr(module, 'in_channels', 0) if hasattr(module, 'in_channels') else 0,
                "output_channels": getattr(module, 'out_channels', 0) if hasattr(module, 'out_channels') else 0,
            }
            layers.append(layer_info)
            
        return layers
        
    def _optimize_partitions(self, partitions: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """optimize the partition scheme, considering communication overhead and computation load
        
        Args:
            partitions: initial partition scheme
            
        Returns:
            Dict[str, List[str]]: optimized partition scheme
        """
        optimized = partitions.copy()
        
        # calculate the computation load of each partition
        partition_loads = {}
        for node_id, layer_names in optimized.items():
            total_params = 0
            for layer_name in layer_names:
                # find the corresponding layer information
                layer_info = next(
                    (layer for layer in self._analyze_model_layers() if layer["name"] == layer_name),
                    None
                )
                if layer_info:
                    total_params += layer_info["params"]
            partition_loads[node_id] = total_params
            
        # if the load of a partition is significantly higher than the others, try to reallocate
        avg_load = sum(partition_loads.values()) / len(partition_loads)
        threshold = avg_load * 1.5  # set the threshold
        
        for node_id, load in partition_loads.items():
            if load > threshold:
                # find the partition with the smallest load
                min_load_node = min(partition_loads.items(), key=lambda x: x[1])[0]
                # move some layers to the partition with smaller load
                while load > threshold and optimized[node_id]:
                    layer_to_move = optimized[node_id].pop()
                    optimized[min_load_node].append(layer_to_move)
                    # update the load
                    layer_info = next(
                        (layer for layer in self._analyze_model_layers() if layer["name"] == layer_to_move),
                        None
                    )
                    if layer_info:
                        load_delta = layer_info["params"]
                        partition_loads[node_id] -= load_delta
                        partition_loads[min_load_node] += load_delta
                        load = partition_loads[node_id]
                        
        return optimized

    def _record_partition_metrics(self, partitions):
        """record the partition performance metrics"""
        metrics = {
            'partition_sizes': [],
            'communication_costs': 0,
            'partition_latencies': [],
            'total_transfer': 0
        }
        
        for node_id, layers in partitions.items():
            # calculate the partition size
            size = sum(self._get_layer_size(layer) for layer in layers)
            metrics['partition_sizes'].append(size)
            
            # estimate the communication cost
            if node_id != list(partitions.keys())[0]:  # not the first partition
                input_size = self._estimate_intermediate_size(layers[0])
                metrics['communication_costs'] += input_size
                metrics['total_transfer'] += input_size
                
        return metrics

    def optimize_partitions(self, partitions, performance_history):
        """optimize the partition strategy"""
        optimized = {}
        
        for node_id, layers in partitions.items():
            # adjust the partition based on the historical performance
            if node_id in performance_history:
                avg_latency = np.mean(performance_history[node_id])
                if avg_latency > self.latency_threshold:
                    # reduce the partition size
                    layers = self._reduce_partition_size(layers)
            
            optimized[node_id] = layers
        
        return optimized, {
            'total_communication_cost': total_communication_cost,
            'partition_sizes': [len(layers) for layers in optimized.values()],
            'optimization_overhead': time.time() - start_time
        }
