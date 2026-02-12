import torch
import torch.nn as nn
import re
from timm import create_model
from einops import rearrange


def _extract_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if isinstance(features, dict):
        preferred_keys = ["x_norm_patchtokens", "x_prenorm", "x"]
        for key in preferred_keys:
            value = features.get(key, None)
            if isinstance(value, torch.Tensor):
                return value
        for value in features.values():
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(features, (list, tuple)):
        for value in features:
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Unsupported forward_features output type: {type(features)}")


def _extract_patch_size(model_name):
    match = re.search(r"patch(\d+)", model_name)
    if match is None:
        return None
    return int(match.group(1))


def _postprocess_spatial_tokens(features, model_name, height, width):
    if features.ndim == 4:
        features = rearrange(features, "b c h w -> b (h w) c")
    if features.ndim != 3:
        raise ValueError(f"Expected 3D token tensor after feature extraction, got {tuple(features.shape)}")

    patch_size = _extract_patch_size(model_name)
    if patch_size is None:
        return features

    num_tokens_expected = (height // patch_size) * (width // patch_size)
    num_tokens_actual = features.shape[1]
    if num_tokens_actual > num_tokens_expected:
        features = features[:, num_tokens_actual - num_tokens_expected :, :]
    return features


class TimmWrapper(nn.Module):
    def __init__(self, model_name, pretrained=True, debug=False, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.debug = debug
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)

    def _log(self, message):
        if self.debug:
            print(message)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"TimmWrapper expects input shape [B, C, H, W], got {tuple(x.shape)}")

        self._log(f"[DEBUG] Entered TimmWrapper forward pass with model '{self.model_name}'.")
        features = self.model.forward_features(x)
        features = _extract_feature_tensor(features)
        features = _postprocess_spatial_tokens(features, self.model_name, x.shape[2], x.shape[3])
        return features


class TemporalTimmWrapper(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        gru_hidden_dim=None,
        gru_num_layers=1,
        gru_dropout=0.0,
        causal=True,
        fusion="residual_add",
        token_pool="none",
        debug=False,
        **kwargs,
    ):
        super().__init__()
        if fusion != "residual_add":
            raise ValueError(f"TemporalTimmWrapper supports only fusion='residual_add', got {fusion}.")
        if token_pool != "none":
            raise ValueError(f"TemporalTimmWrapper supports only token_pool='none', got {token_pool}.")

        self.model_name = model_name
        self.debug = debug
        self.causal = causal
        self.fusion = fusion
        self.token_pool = token_pool
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0, **kwargs)

        feature_dim = getattr(self.model, "num_features", None)
        if feature_dim is None:
            feature_dim = getattr(self.model, "embed_dim", None)
        if feature_dim is None:
            raise ValueError("Cannot infer feature dimension from timm model. Please use a ViT-like model.")
        self.feature_dim = int(feature_dim)

        hidden_dim = self.feature_dim if gru_hidden_dim is None else int(gru_hidden_dim)
        bidirectional = not causal
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        if gru_out_dim == self.feature_dim:
            self.temporal_proj = nn.Identity()
        else:
            self.temporal_proj = nn.Linear(gru_out_dim, self.feature_dim)

    def _log(self, message):
        if self.debug:
            print(message)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"TemporalTimmWrapper expects input shape [B, T, C, H, W], got {tuple(x.shape)}")

        batch_size, seq_len, channels, height, width = x.shape
        frames = x.reshape(batch_size * seq_len, channels, height, width)
        spatial_tokens = self.model.forward_features(frames)
        spatial_tokens = _extract_feature_tensor(spatial_tokens)
        spatial_tokens = _postprocess_spatial_tokens(spatial_tokens, self.model_name, height, width)
        spatial_tokens = rearrange(spatial_tokens, "(b t) n c -> b t n c", b=batch_size, t=seq_len)

        token_sequences = rearrange(spatial_tokens, "b t n c -> (b n) t c")
        temporal_tokens, _ = self.gru(token_sequences)
        temporal_tokens = self.temporal_proj(temporal_tokens)
        temporal_tokens = rearrange(temporal_tokens, "(b n) t c -> b t n c", b=batch_size, n=spatial_tokens.shape[2])

        output_tokens = spatial_tokens + temporal_tokens
        self._log(f"[DEBUG] TemporalTimmWrapper output shape: {tuple(output_tokens.shape)}")
        return output_tokens


if __name__ == "__main__":
    # Example usage
    model = TimmWrapper('vit_small_patch16_dinov3', pretrained=False)
    input_tensor = torch.randn(8, 3, 256, 224)
    output = model(input_tensor)
    print(output.shape)  # Should print the feature map shape
