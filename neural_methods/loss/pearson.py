import torch
import torch.nn as nn

class PearsonLoss(nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor):
        # Flatten the tensors
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        # Mean centering
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)
        x = x - x_mean
        y = y - y_mean

        # Numerator
        cov = torch.sum(x * y, dim=1)

        # Denominator
        x_std = torch.sqrt(torch.sum(x**2, dim=1))
        y_std = torch.sqrt(torch.sum(y**2, dim=1))

        # Pearson Correlation Coefficient
        pcc = cov / ((x_std * y_std) + 1e-8)

        # Pearson **Loss**
        loss = 1 - pcc

        # Sum loss across the batch
        return loss.mean()