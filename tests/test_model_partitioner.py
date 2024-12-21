import pytest
import torch
import torch.nn as nn
from src.core.model_partitioner import ModelPartitioner

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return self.fc(x.view(x.size(0), -1))

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def partitioner(model):
    return ModelPartitioner(model)

def test_layer_analysis(partitioner):
    layers = partitioner.layers
    assert len(layers) == 4  # conv1, relu1, conv2, fc
    
    # Check layer information
    conv1_info = next(l for l in layers if l['name'].endswith('conv1'))
    assert conv1_info['type'] == 'Conv2d'
    assert conv1_info['parameters'] > 0

def test_partitioning(partitioner):
    partitions = partitioner.partition_model(
        num_partitions=2,
        resource_weights={'computation': 0.7, 'memory': 0.3}
    )
    
    assert len(partitions) == 2
    assert len(partitions[0]) > 0
    assert len(partitions[1]) > 0

def test_communication_cost(partitioner):
    partitions = partitioner.partition_model(2, {'computation': 1.0})
    costs = partitioner.analyze_communication_cost(partitions)
    
    assert 'partition_0_to_1' in costs
    assert costs['partition_0_to_1'] >= 0 