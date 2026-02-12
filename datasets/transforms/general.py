import numpy as np

class RandomApply():
    def __init__(self, transforms, prob):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            for t in self.transforms:
                sample = t(sample)
        return sample

class ComposeTransform():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class AddKeypointNoise():
    def __init__(
        self,
        noise_std_3d: float = 0.01,
        noise_std_2d: float = 0.005,
        apply_to_global: bool = True,
    ):
        self.noise_std_3d = float(noise_std_3d)
        self.noise_std_2d = float(noise_std_2d)
        self.apply_to_global = bool(apply_to_global)

    def __call__(self, sample):
        for key, value in list(sample.items()):
            if value is None:
                continue
            if key == "gt_keypoints" and not self.apply_to_global:
                continue
            if not key.startswith("gt_keypoints"):
                continue
            if not hasattr(value, "shape"):
                continue

            if value.shape[-1] == 2 and self.noise_std_2d > 0:
                if hasattr(value, "dtype") and "torch" in type(value).__module__:
                    noise = value.new_empty(value.shape).normal_(mean=0.0, std=self.noise_std_2d)
                    sample[key] = value + noise
                else:
                    noise = np.random.randn(*value.shape).astype(np.float32) * self.noise_std_2d
                    sample[key] = value + noise
            elif value.shape[-1] == 3 and self.noise_std_3d > 0:
                if hasattr(value, "dtype") and "torch" in type(value).__module__:
                    noise = value.new_empty(value.shape).normal_(mean=0.0, std=self.noise_std_3d)
                    sample[key] = value + noise
                else:
                    noise = np.random.randn(*value.shape).astype(np.float32) * self.noise_std_3d
                    sample[key] = value + noise
        return sample
