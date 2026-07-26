from ml_configs import torch,nn

# class DynamicMissPredictionLoss(nn.Module):
#     def __init__(self):
#         super(DynamicMissPredictionLoss, self).__init__()

#     def forward(self, outputs, targets, weights, rates):
#         y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
#         bce = torch.nn.functional.binary_cross_entropy(y_k, targets, reduction='none') # Loss from the actual target compares to the predicted
#         sample_weights = weights * torch.abs(2.0 * rates - 1.0)
#         return torch.sum(bce * sample_weights) / torch.sum(weights)


class WeightedMissLoss(nn.Module):

    def __init__(self):
        super(WeightedMissLoss, self).__init__()

    def forward(self, outputs, targets, weights, rates):
        y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
        errors = ((1.0 - y_k) * rates * weights) + (y_k * (1.0 - rates) * weights)
        return torch.sum(errors) / torch.sum(weights)


class StaticMissLoss(nn.Module):

    def __init__(self):
        super(StaticMissLoss, self).__init__()

    def forward(self, outputs, targets, weights, rates):
        y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
        errors = (1.0 - y_k) * rates + y_k * (1.0 - rates)
        return torch.mean(errors)
