from __future__ import annotations

import numpy as np
import torch

from misc.vis import denormalize


def process_image_for_display(
    image,
    denorm_params: dict | None,
    key: str,
    keep_temporal: bool = False,
):
    """Convert CHW/TCHW tensors to displayable image arrays."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if image.ndim == 4 and not keep_temporal:
        image = image[-1]

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)

    if denorm_params is not None:
        if key == "rgb":
            mean = denorm_params.get("rgb_mean", [123.675, 116.28, 103.53])
            std = denorm_params.get("rgb_std", [58.395, 57.12, 57.375])
        else:
            mean = denorm_params.get("depth_mean", [0.0])
            std = denorm_params.get("depth_std", [255.0])
        image = denormalize(image, mean, std)

    if key == "depth":
        if image.shape[-1] == 3:
            image = image[:, :, 0]
        elif image.shape[-1] == 1:
            image = image.squeeze(-1)

    return np.asarray(image)
