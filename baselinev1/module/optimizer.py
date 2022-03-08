

import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW
# from torch.optim import RAdam
from adamp import AdamP

from transformers.optimization import Adafactor
import torch_optimizer as optim


class SAM(torch.optim.Optimizer):
    """
    Reference - https://github.com/davda54/sam
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


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
    elif args.optimizer == 'lamb':
        optimizer = optim.Lamb(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'lookahead':
        yogi = optim.Yogi(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-3,
            initial_accumulator=1e-6,
            weight_decay=args.weight_decay,
        )
        optimizer = optim.Lookahead(yogi, k=5, alpha=0.5)
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        # optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.AdamP(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            delta=0.1,
            wd_ratio=0.1
        )
    elif args.optimizer == 'radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
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
                           
    elif args.optimizer == 'adabound':
        optimizer = optim.AdaBound(
            model.parameters(),
            lr=args.lr,
            betas= (0.9, 0.999),
            final_lr=args.lr*100, # final SGD learning rate
            gamma=1e-3,
            eps=1e-8,
            weight_decay=args.weight_decay,
            amsbound=False,
        )
    elif args.optimizer == 'adabelief':
        optimizer = optim.AdaBelief(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-3,
            weight_decay=args.weight_decay,
            amsgrad=False,
            weight_decouple=False,
            fixed_decay=False,
            rectify=False,
        )
    elif args.optimizer == 'adahessian':
        optimizer = optim.Adahessian(
            model.parameters(),
            lr=args.lr*10,
            betas= (0.9, 0.999),
            eps=1e-4,
            weight_decay=args.weight_decay,
            hessian_power=1.0,
        )
    elif args.optimizer == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=args.max_grad_norm,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer == 'diffgrad':
        optimizer = optim.DiffGrad(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'qhadam':
        optimizer = optim.QHAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            nus=(1.0, 1.0),
            weight_decay=args.weight_decay,
            decouple_weight_decay=False,
            eps=1e-8,
        )
    elif args.optimizer == 'qhm':
        optimizer = optim.QHM(
            model.parameters(),
            lr=args.lr,
            momentum=0,
            nu=0.7,
            weight_decay=1e-2,
            weight_decay_type='grad',
        )
    elif args.optimizer == 'yogi':
        optimizer = optim.Yogi(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-3,
            initial_accumulator=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'madgrad':
        optimizer = optim.MADGRAD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            eps=1e-6,
        )
    elif args.optimizer == 'sam':
        """Sharpness-Aware Minimization for Efficiently Improving Generalization

        (Adaptive) SAM Optimizer is using Adafactor for the base optimizer
        you can change it to anything you want for base optimizer
        don't forget to change the **kwargs arguments for base optimizer also!
        """
        base_optimizer = Adafactor
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=args.max_grad_norm,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    # layerwise learning rate
    
    optimizer.zero_grad()
    
    return optimizer