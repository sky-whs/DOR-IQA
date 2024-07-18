
from losses import OrdinalRegression2d
import torch.nn as nn
import torch
def create_lossfunc():
    ignore_index = 0
    weight = None       
    loss_kwargs = {'ignore_index': ignore_index, 'reduction': 'sum', 'use_weights': False, 'weight': weight}

    lossfunc = OrdinalRegression2d(**loss_kwargs)
       
    criterion_kwargs = {'min_score': 0.0, 'max_score': 1.0, 'num_classes': 28,
                        'AppearanceLoss': lossfunc}
    criterion = LossFunc(**criterion_kwargs)
    return criterion

def continuous2discrete(score, d_min, d_max, n_c):
    score = torch.round((score - d_min) /(d_max - d_min) * (n_c-1))
    return score
def discrete2continuous(score, d_min, d_max, n_c):
    score = score * (d_max-d_min) / (n_c-1) + d_min
    return score
class LossFunc(nn.Module):
        def __init__(self, min_score, max_score, num_classes, 
                     AppearanceLoss=None):
            super(LossFunc, self).__init__()
            self.min_score = min_score
            self.max_score = max_score
            self.num_classes = num_classes
            self.AppearanceLoss = AppearanceLoss


        def forward(self, preds, label):
            """
            Parameter
            ---------
            preds: [batch_size, c, h, w] * 2
            label: [batch_size, 1, h, w]
            """
            y = preds
            dis_label = continuous2discrete(label, self.min_score, self.max_score, self.num_classes)
            loss1 = self.AppearanceLoss(y, dis_label.squeeze(1).long())
            return loss1