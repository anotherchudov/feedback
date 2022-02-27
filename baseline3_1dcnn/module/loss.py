
import torch
from torch import nn


def custom_ce(preds, gts, class_weight):
    loss = -(((preds * gts).sum(-1) * class_weight).sum(-1) / class_weight.sum(-1)).mean()

    return loss

def custom_rce(preds, gts, class_weight):
    loss = -(((torch.exp(preds) * torch.log_softmax(gts, -1)).sum(-1) * class_weight).sum(-1) / class_weight.sum(-1)).mean()

    return loss

class Criterion():
    """Wrapper for multi criterion calculation"""
    def __init__(self, args):
        self.args = args
        self.criterions = self.get_criterions()
        self.criterion_names = self.args.criterion_list
        self.criterion_ratios = args.criterion_ratio

    def get_criterions(self):
        criterions = []
        for criterion in self.args.criterion_list:
            if criterion == "crossentropy":
                criterions.append(nn.CrossEntropyLoss())
            elif criterion == "custom_ce":
                criterions.append(custom_ce)
            elif criterion == "custom_rce":
                criterions.append(custom_rce)
                
        return criterions
    
    def calculate_loss(self, preds, gts, class_weight=None):
        total_loss = 0
        for criterion_name, criterion, ratio in zip(self.criterion_names, self.criterions, self.criterion_ratios):
            if criterion_name in ['custom_ce', 'custom_rce']:
                current_loss = criterion(preds, gts, class_weight)
            else:
                current_loss = criterion(preds, gts)
            total_loss = total_loss + current_loss * ratio

        return total_loss

    def __call__(self, preds, gts, class_weight=None):
        return self.calculate_loss(preds, gts, class_weight)


def get_criterion(args):
    criterion = Criterion(args)

    return criterion


