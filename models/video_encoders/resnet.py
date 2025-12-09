import torch
import torch.nn as nn
from torchvision.models import ResNet, ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck

class ResNetEncoder(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)
        return x
    
class ResNet18(ResNetEncoder):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            **kwargs
        )
        if pretrained:
            self.load_state_dict(ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))

class ResNet50(ResNetEncoder):
    def __init__(self, pretrained=False, **kwargs):
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            **kwargs
        )
        if pretrained:
            self.load_state_dict(ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))

if __name__ == "__main__":
    model = ResNet18()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)