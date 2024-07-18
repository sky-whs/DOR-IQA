import torch
import torch.nn as nn

def continuous2discrete(score, d_min, d_max, n_c):
    score = torch.round((score - d_min) / (d_max - d_min) * (n_c - 1))
    return score

def discrete2continuous(score, d_min, d_max, n_c):
    score = score / (n_c - 1) * (d_max - d_min) + d_min
    return score
class DOR(nn.Module):

    def __init__(self):
        super(DOR, self).__init__()
        # The output dimension is fully connected from your previous layer. for example 128
        output = 128
        self.classifier = nn.Sequential(
            nn.Linear(output, 56)
        )
    def soft_ordinal_regression(self, pred_prob, d_min, d_max, n_c):
        pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
        Intergral = torch.floor(pred_prob_sum)
        Fraction = pred_prob_sum - Intergral
        score_low = (discrete2continuous(Intergral, d_min, d_max, n_c) +
                     discrete2continuous(Intergral + 1, d_min, d_max, n_c)) / 2
        score_high = (discrete2continuous(Intergral + 1, d_min, d_max, n_c) +
                      discrete2continuous(Intergral + 2, d_min, d_max, n_c)) / 2
        pred_score = score_low * (1 - Fraction) + score_high * Fraction
        return pred_score
    def decode_ord(self, y):
        batch_size, prob = y.shape
        y = torch.reshape(y, (batch_size, prob//2, 2, 1, 1))
        denominator = torch.sum(torch.exp(y), 2)
        pred_score = torch.div(torch.exp(y[:, :, 1, :, :]), denominator)
        return pred_score

    def inference(self, y):
        inferenceFunc = self.soft_ordinal_regression
        # The minimum value of the data set is 0 and the maximum value of the data set is 1.
        # Change the following parameters according to your data set.
        # K = 28 according to your requirement
        pred_score = inferenceFunc(y, 0.0, 1.0, 28)
        return pred_score

    def forward(self, x):
        x = self.classifier(x)
        x = self.decode_ord(x)
        return x

