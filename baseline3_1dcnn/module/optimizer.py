

from torch.optim import Adam, AdamW

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == 'adamw':
        weights = []
        biases = []
        for n, p in model.named_parameters():
            if n.startswith('feats.embeddings') or 'LayerNorm' in n or n.endswith('bias'):
                # embedding layer & bias layer
                biases.append(p)
            else:
                # except above
                weights.append(p)

        optimizer = AdamW([{'params': weights, 'weight_decay': args.weight_decay, 'lr': 0},
                           {'params': biases, 'weight_decay': 0 if not args.decay_bias else args.weight_decay, 'lr': 0}])

    
    optimizer.zero_grad()
    
    return optimizer