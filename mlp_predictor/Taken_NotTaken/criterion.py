from ml_configs import nn, torch


class DynamicMissPredictionLoss(nn.Module):
    def __init__(self):
        super(DynamicMissPredictionLoss,self).__init__()
    
    def forward(self,outputs,targets,weights,rates):
        y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs

        emb = (1.0 - y_k) * rates * weights
        bit =  y_k*(1.0 - rates) * weights

        return  torch.sum(emb+bit) / torch.sum(weights)

        
        

