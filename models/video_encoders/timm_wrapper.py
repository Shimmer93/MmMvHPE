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
        print(f"[DEBUG]: Entered TimmWrapper forward pass with model '{self.model_name}'.")
        print("[DEBUG]: This is the memory usage before forward pass:")
        print(f"[DEBUG]:Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[DEBUG]:Current memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print("[DEBUG]: Input tensor shape:", x.shape)
        print(f"[DEBUG]: Passing through model's forward_features method. Model name is '{self.model_name}'.")
        features = self.model.forward_features(x)

        if features.ndim == 4:
            features = rearrange(features, 'b c h w -> b (h w) c')
        print("[DEBUG]: This is the memory usage after forward_features:")
        print(f"[DEBUG]:Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[DEBUG]:Current memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        if 'vit' in self.model_name and 'patch' in self.model_name:
            print("[DEBUG]: Adjusting token count for ViT model.")
            patch_size = int(self.model_name.split('patch')[1].split('_')[0])
            num_tokens_exp = (x.shape[2] // patch_size) * (x.shape[3] // patch_size)
            num_tokens_act = features.shape[1]
            if num_tokens_act > num_tokens_exp:
                features = features[:, num_tokens_act - num_tokens_exp:, :]
        print(f"[DEBUG]: TimmWrapper forward pass completed. Output shape: {features.shape}")
        print("[DEBUG]: This is the memory usage when the pass is complete:")
        print(f"[DEBUG]:Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[DEBUG]:Current memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        return features

if __name__ == "__main__":
    # Example usage
    model = TimmWrapper('vit_small_patch16_dinov3', pretrained=False)
    input_tensor = torch.randn(8, 3, 256, 224)
    output = model(input_tensor)
    print(output.shape)  # Should print the feature map shape