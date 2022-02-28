
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_scheduler as transformer_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

 
def get_scheduler(args, optimizer):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, mode='max', verbose=True)
    if args.scheduler == 'cosine':
        """ Aiming for One Cycle Learning!
        - T_max is setted as the half of the total steps of 1 epoch
        - min_lr is setted as 1e-7 by default
        """
        half_cycle_steps = args.steps_per_epoch // 2
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=half_cycle_steps,
                                      eta_min=1e-7)
    elif args.scheduler == 'transformer_linear':
        total_steps = args.steps_per_epoch*args.epochs
        scheduler = transformer_scheduler('linear',
                                          optimizer=optimizer,
                                          num_warmup_steps=args.warmup_steps,
                                          num_training_steps=total_steps)
    elif args.scheduler == 'custom_warmup':
        scheduler = np.r_[np.linspace(0, args.lr, args.warmup_steps),
                        (np.cos(np.linspace(0, np.pi, args.steps_per_epoch*args.epochs - args.warmup_steps)) * .5 + .5) * (args.lr - args.min_lr)
                        + args.min_lr]

    return scheduler