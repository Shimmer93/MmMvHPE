import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.video_encoders.timm_wrapper as timm_wrapper_module
from models.video_encoders.timm_wrapper import TemporalTimmWrapper, TimmWrapper


class DummyViTModel(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.num_features = embed_dim

    def forward_features(self, x):
        batch_size, _, height, width = x.shape
        patch_size = 16
        num_patches = (height // patch_size) * (width // patch_size)
        patch_tokens = torch.randn(batch_size, num_patches, self.num_features, device=x.device, dtype=x.dtype)
        cls_token = torch.zeros(batch_size, 1, self.num_features, device=x.device, dtype=x.dtype)
        return torch.cat([cls_token, patch_tokens], dim=1)


def _mock_create_model(*args, **kwargs):
    return DummyViTModel(embed_dim=8)


def test_timm_wrapper_drops_cls_token_for_vit(monkeypatch):
    monkeypatch.setattr(timm_wrapper_module, "create_model", _mock_create_model)
    model = TimmWrapper(model_name="vit_small_patch16_dinov3", pretrained=False)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 4, 8)


def test_temporal_timm_wrapper_outputs_btnd(monkeypatch):
    monkeypatch.setattr(timm_wrapper_module, "create_model", _mock_create_model)
    model = TemporalTimmWrapper(
        model_name="vit_small_patch16_dinov3",
        pretrained=False,
        gru_hidden_dim=8,
        gru_num_layers=1,
        causal=True,
        fusion="residual_add",
        token_pool="none",
    )
    x = torch.randn(2, 3, 3, 32, 32, dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 3, 4, 8)


def test_temporal_timm_wrapper_supports_seq_len_one(monkeypatch):
    monkeypatch.setattr(timm_wrapper_module, "create_model", _mock_create_model)
    model = TemporalTimmWrapper(
        model_name="vit_small_patch16_dinov3",
        pretrained=False,
        gru_hidden_dim=8,
        gru_num_layers=1,
        causal=True,
        fusion="residual_add",
        token_pool="none",
    )
    x = torch.randn(2, 1, 3, 32, 32, dtype=torch.float32)
    out = model(x)
    assert out.shape == (2, 1, 4, 8)
