

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .debertav3 import DebertaV3Large
from .debertav3_ducky import DebertaV3LargeDucky

def get_model(args):
    if args.model == 'microsoft/deberta-v3-large':
        model = DebertaV3Large(args).to(args.device)

        # dropout layer
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = args.dropout_ratio
    elif args.model == 'microsoft/deberta-v3-large-ducky':
        model = DebertaV3LargeDucky(args).to(args.device)

        # dropout layer
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = args.dropout_ratio

    # distributed training
    if args.ddp:
        model = DDP(model, device_ids=[args.rank], output_device=args.rank)
        model.to(args.device)

    return model