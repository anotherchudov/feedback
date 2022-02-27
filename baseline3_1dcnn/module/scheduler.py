
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
 
def get_scheduler(args, optimizer):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, mode='max', verbose=True)
    elif args.scheduler == 'custom_warmup':
        scheduler = np.r_[np.linspace(0, args.lr, args.warmup_steps),
                        (np.cos(np.linspace(0, np.pi, args.steps_per_epoch*args.epochs - args.warmup_steps)) * .5 + .5) * (args.lr - args.min_lr)
                        + args.min_lr]

    return scheduler