from .base_head import BaseHead


class DirectKeypointHead(BaseHead):
    def __init__(self, losses):
        super().__init__(losses)

    def forward(self, x):
        if isinstance(x, dict) and "pred_keypoints" in x:
            return x["pred_keypoints"]
        return x

    def loss(self, x, data_batch):
        pred_keypoints = self.forward(x)
        losses = {}
        for loss_name, (loss_fn, loss_weight) in self.losses.items():
            losses[loss_name] = (loss_fn(pred_keypoints, data_batch["gt_keypoints"]), loss_weight)
        return losses

    def predict(self, x):
        return self.forward(x)
