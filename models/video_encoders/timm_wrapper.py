import torch
import torch.nn as nn
from timm import create_model
from einops import rearrange

class TimmWrapper(nn.Module):
    def __init__(self, model_name, pretrained=True, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)

    def forward(self, x):
        features = self.model.forward_features(x)

        if features.ndim == 4:
            features = rearrange(features, 'b c h w -> b (h w) c')

        if 'vit' in self.model_name and 'patch' in self.model_name:
            patch_size = int(self.model_name.split('patch')[1].split('_')[0])
            num_tokens_exp = (x.shape[2] // patch_size) * (x.shape[3] // patch_size)
            num_tokens_act = features.shape[1]
            if num_tokens_act > num_tokens_exp:
                features = features[:, num_tokens_act - num_tokens_exp:, :]

        return features

if __name__ == "__main__":
    # Example usage
    model = TimmWrapper('vit_small_patch16_dinov3', pretrained=False)
    input_tensor = torch.randn(8, 3, 256, 224)
    output = model(input_tensor)
    print(output.shape)  # Should print the feature map shape