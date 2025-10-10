
import torch
from stormfusion.training.forecast_metrics import scores

def test_scores_shape():
    pred = torch.zeros(2,1,8,8)
    truth = torch.zeros(2,1,8,8)
    s = scores(pred, truth)
    assert 74 in s and 'CSI' in s[74]
