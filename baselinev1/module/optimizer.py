

from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW

from transformers.optimization import Adafactor


def get_optimizer(args, model):
    # scale weight decay
    # args.weight_decay = args.weight_decay * args.batch_size / (args.batch_size * args.grad_acc_steps)
    # args.weight_decay /= args.grad_acc_steps

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.optimizer == 'adamw':
        weights = []
        biases = []
        for n, p in model.named_parameters():
            if n.startswith('feats.embeddings') or 'LayerNorm' in n or n.endswith('bias'):
                # embedding layer & bias layer
                biases.append(p)
            else:
                # except above
                weights.append(p)

        optimizer = AdamW([{'params': weights, 'weight_decay': args.weight_decay, 'lr': args.lr},
                           {'params': biases, 'weight_decay': 0 if not args.decay_bias else args.weight_decay, 'lr': args.lr}])
                           
    elif args.optimizer == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=args.max_grad_norm,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    
    optimizer.zero_grad()
    
    return optimizer