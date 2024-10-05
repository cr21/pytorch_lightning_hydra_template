from torchmetrics import Metric

import torch
class CustomAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total',dist_reduce_fx="sum", default=torch.tensor(0))
        self.add_state('correct',dist_reduce_fx="sum", default=torch.tensor(0))


    def update(self, preds, target):
        preds=torch.argmax(preds, dim=1)
        assert preds.shape==target.shape
        self.correct += torch.sum(preds==target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float()/self.total.float()