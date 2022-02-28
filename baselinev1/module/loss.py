
import torch
from torch import nn

from .losses.sergei import custom_ce
from .losses.sergei import custom_rce
from .losses.dice_loss import DiceLoss
from .losses.focal_loss import FocalLoss


class Criterion():
    """Wrapper for multi criterion calculation
    
    OHEM (Online Hard Example Mining) is used to reduce the number of examples
    - Focal Loss
    - Dice Loss
    """
    def __init__(self, args):
        self.args = args
        self.criterions = self.get_criterions()
        self.criterion_names = self.args.criterion_list
        self.criterion_ratios = args.criterion_ratio

    def get_criterions(self):
        criterions = []
        for criterion in self.args.criterion_list:
            # pytorch version
            if criterion == "crossentropy":
                criterions.append(nn.CrossEntropyLoss(weight=self.args.class_weight, label_smoothing=self.args.label_smoothing))

            # sergei chudov
            elif criterion == "custom_ce":
                criterions.append(custom_ce)
            elif criterion == "custom_rce":
                criterions.append(custom_rce)

            # reference - https://github.com/ShannonAI/dice_loss_for_NLP
            elif criterion == "focal":
                criterions.append(FocalLoss(reduction='mean'))
            elif criterion == "dice":
                criterions.append(DiceLoss())
                
        return criterions

    def reshape(self, preds, gts):
        """TODO: fix the dataloader to argmax label version and change this code"""
        return preds.view(-1, 15), gts.argmax(-1).view(-1)
    
    def calculate_loss(self, preds, gts, class_weight=None):
        """Calculate loss for each criterion and return the sum of loss"""
        total_loss = 0
        for criterion_name, criterion, ratio in zip(self.criterion_names, self.criterions, self.criterion_ratios):
            if criterion_name in ['custom_ce', 'custom_rce']:
                current_loss = criterion(preds, gts, class_weight)
            else:
                # TODO: tailored for current dataloader
                preds_, gts_ = self.reshape(preds, gts)
                current_loss = criterion(preds_, gts_)
            total_loss = total_loss + current_loss * ratio

        return total_loss

    def __call__(self, preds, gts, class_weight=None):
        return self.calculate_loss(preds, gts, class_weight)


def get_criterion(args):
    criterion = Criterion(args)

    return criterion


