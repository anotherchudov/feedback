
import torch

# loss function by sergei chudov
# ---------------------------------------------------------------
def custom_ce(preds, gts, class_weight):
    preds = torch.log_softmax(preds, -1)
    loss = -(((preds * gts).sum(-1) * class_weight).sum(-1) / class_weight.sum(-1)).mean()

    return loss

def custom_rce(preds, gts, class_weight):
    """Reverse Cross Entropy
    Later considering using the code from
    https://github.com/HanxunH/SCELoss-Reproduce
    """
    preds = torch.log_softmax(preds, -1)
    loss = -(((torch.exp(preds) * torch.log_softmax(gts, -1)).sum(-1) * class_weight).sum(-1) / class_weight.sum(-1)).mean()

    return loss