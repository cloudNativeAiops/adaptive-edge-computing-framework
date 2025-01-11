import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetWrapper, self).__init__()  # 正确初始化父类
        if pretrained:
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.model = mobilenet_v2(weights=None)
        
        # 修改最后的分类层以匹配我们的类别数（FashionMNIST有10个类别）
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
