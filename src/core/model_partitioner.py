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
        super().__init__(model=model)  # 显式调用父类的初始化方法
        
    def partition_for_cluster(self, available_nodes: Dict[str, NodeResources]):
        """基于集群资源进行模型分区"""
        # 1. 分析每个节点的计算能力
        node_capabilities = self._analyze_node_capabilities(available_nodes)
        
        # 2. 根据计算能力分配模型层
        partitions = self._distribute_layers(node_capabilities)
        
        # 3. 优化分区方案，考虑通信开销
        optimized_partitions = self._optimize_partitions(partitions)
        
        return optimized_partitions

    def _analyze_node_capabilities(self, nodes):
        """分析节点计算能力"""
        capabilities = {}
        for node_id, resources in nodes.items():
            compute_power = resources.cpu_available * 100  # 简化的计算能力评估
            capabilities[node_id] = compute_power
        return capabilities
        
    def _distribute_layers(self, node_capabilities: Dict[str, float]) -> Dict[str, List[str]]:
        """根据节点计算能力分配模型层"""
        total_capability = sum(node_capabilities.values())
        layers = self._analyze_model_layers()
        total_layers = len(layers)
        
        # 初始化分区结果
        partitions = {node_id: [] for node_id in node_capabilities.keys()}
        
        # 根据计算能力比例分配层
        current_layer = 0
        for node_id, capability in node_capabilities.items():
            # 计算该节点应分配的层数
            layer_count = int((capability / total_capability) * total_layers)
            if current_layer < total_layers:
                end_layer = min(current_layer + layer_count, total_layers)
                partitions[node_id] = [
                    layer["name"] for layer in layers[current_layer:end_layer]
                ]
                current_layer = end_layer
        
        # 处理剩余的层（如果有的话）
        while current_layer < total_layers:
            # 将剩余的层分配给计算能力最强的节点
            strongest_node = max(node_capabilities.items(), key=lambda x: x[1])[0]
            partitions[strongest_node].append(layers[current_layer]["name"])
            current_layer += 1
            
        return partitions
        
    def _analyze_model_layers(self):
        """分析模型的层结构
        
        Returns:
            List[Dict]: 包含每层信息的列表，每层信息包含 name, type, params 等
        """
        layers = []
        for name, module in self.model.named_modules():
            # 跳过容器模块（如Sequential）
            if len(list(module.children())) > 0:
                continue
                
            # 计算参数量
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "params": params,
                # 可以添加更多层的特征
                "input_channels": getattr(module, 'in_channels', 0) if hasattr(module, 'in_channels') else 0,
                "output_channels": getattr(module, 'out_channels', 0) if hasattr(module, 'out_channels') else 0,
            }
            layers.append(layer_info)
            
        return layers
        
    def _optimize_partitions(self, partitions: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """优化分区方案，考虑通信开销和计算负载
        
        Args:
            partitions: 初始分区方案
            
        Returns:
            Dict[str, List[str]]: 优化后的分区方案
        """
        optimized = partitions.copy()
        
        # 计算每个分区的计算负载
        partition_loads = {}
        for node_id, layer_names in optimized.items():
            total_params = 0
            for layer_name in layer_names:
                # 找到对应层的信息
                layer_info = next(
                    (layer for layer in self._analyze_model_layers() if layer["name"] == layer_name),
                    None
                )
                if layer_info:
                    total_params += layer_info["params"]
            partition_loads[node_id] = total_params
            
        # 如果某个分区的负载明显高于其他分区，尝试重新分配
        avg_load = sum(partition_loads.values()) / len(partition_loads)
        threshold = avg_load * 1.5  # 设置阈值
        
        for node_id, load in partition_loads.items():
            if load > threshold:
                # 找到负载最小的分区
                min_load_node = min(partition_loads.items(), key=lambda x: x[1])[0]
                # 移动一些层到负载较小的分区
                while load > threshold and optimized[node_id]:
                    layer_to_move = optimized[node_id].pop()
                    optimized[min_load_node].append(layer_to_move)
                    # 更新负载
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
        """记录分区性能指标"""
        metrics = {
            'partition_sizes': [],
            'communication_costs': 0,
            'partition_latencies': [],
            'total_transfer': 0
        }
        
        for node_id, layers in partitions.items():
            # 计算分区大小
            size = sum(self._get_layer_size(layer) for layer in layers)
            metrics['partition_sizes'].append(size)
            
            # 估算通信成本
            if node_id != list(partitions.keys())[0]:  # 不是第一个分区
                input_size = self._estimate_intermediate_size(layers[0])
                metrics['communication_costs'] += input_size
                metrics['total_transfer'] += input_size
                
        return metrics

    def optimize_partitions(self, partitions, performance_history):
        """优化分区策略"""
        optimized = {}
        total_communication_cost = 0
        
        for node_id, layers in partitions.items():
            # 计算当前分区的通信成本
            comm_cost = self._calculate_communication_cost(layers)
            total_communication_cost += comm_cost
            
            # 基于历史性能调整分区
            if node_id in performance_history:
                avg_latency = np.mean(performance_history[node_id])
                if avg_latency > self.latency_threshold:
                    # 减少分区大小
                    layers = self._reduce_partition_size(layers)
            
            optimized[node_id] = layers
        
        return optimized, {
            'total_communication_cost': total_communication_cost,
            'partition_sizes': [len(layers) for layers in optimized.values()],
            'optimization_overhead': time.time() - start_time
        }
