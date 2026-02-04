import torch
from pytorch_forecasting.metrics import QuantileLoss


class WeightedQuantileLoss(QuantileLoss):

    def __init__(self, quantiles):
        super().__init__(quantiles=quantiles)

    def loss(self, y_pred, target, **kwargs):

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = (target - y_pred[..., i]) / target

            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses


