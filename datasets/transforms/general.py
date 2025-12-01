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