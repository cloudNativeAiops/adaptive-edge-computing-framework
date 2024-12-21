import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any
import time

class MobileNetWrapper:
    def __init__(self, pretrained: bool = True):
        """Initialize MobileNetV2 model.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """Preprocess input data for inference."""
        return input_data.to(self.device)
    
    def inference(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run inference on input data.
        
        Returns:
            Tuple of (predictions, metadata)
        """
        # Use time.perf_counter() for high-precision timing on both CPU and GPU
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = self.model(input_data)
            
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        metadata = {
            'inference_time_ms': inference_time,
            'device': str(self.device)
        }
        
        return output, metadata
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about model layers."""
        layer_info = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layer_info[name] = {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
        return layer_info
