import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetWrapper, self).__init__()  # correct initialization of the parent class
        if pretrained:
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.model = mobilenet_v2(weights=None)
        
        # modify the last classification layer to match our number of classes (FashionMNIST has 10 classes)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 10)
    
    def forward(self, x):
        return self.model(x)
    
    def eval(self):
        self.model.eval()
        return self
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
