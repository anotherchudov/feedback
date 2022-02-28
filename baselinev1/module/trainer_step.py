import torch
import numpy as np
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from .utils import calc_acc, process_sample, make_match_dict

class Trainer():
    def __init__(self, model, train_loader, valid_loader, lr_schedule, opt, label_names, args):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_schedule = lr_schedule
        self.opt = opt
        self.label_names = label_names
        self.args = args

        self.scaler = GradScaler()
    
    def validate(self):
        ls = []
        self.model.eval()
        val_matches = torch.zeros(16)
        val_labels = torch.zeros(16)
        match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s = np.zeros(8)
        rec = np.zeros(7)
        prec = np.zeros(7)
        for tokens, mask, labels, labels_mask, bounds, gt_dicts, index_map, num_tokens in tqdm(self.valid_loader, total=len(self.valid_loader)):
            with torch.no_grad():
                tokens, mask, label, label_mask = (x.cuda() for x in (tokens, mask, labels, labels_mask))
                with torch.cuda.amp.autocast():
                    outs = self.model(tokens, mask)
                ls.append(-(((outs.float() * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean())
                match_updates = calc_acc(outs, label, label_mask)
                val_matches += match_updates[0]
                val_labels += match_updates[1]
                for sample_ix, num in enumerate(num_tokens):
                    match_stats = process_sample(outs[sample_ix], labels[sample_ix], 
                                                    index_map[sample_ix], bounds[sample_ix],
                                                    gt_dicts[sample_ix],
                                                    num, match_stats, min_len=self.args.min_len)
        for ix in range(1, 8):
            f1s[ix] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + .5 * (match_stats[ix]['fp'] + match_stats[ix]['fn']))
            rec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fn'])
            prec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fp'])
        f1s[0] = np.mean(f1s[1:])

        self.model.train()
        val_accs = (val_matches / val_labels).cpu().numpy()
        val_labels = val_labels.cpu().numpy()

        return torch.stack(ls).mean().item(), val_accs, val_labels, f1s, rec, prec
    
    def train(self):
        ls = []
        norms = []
        train_matches = torch.zeros(16)
        train_labels = torch.zeros(16)
        log_dict = {}
        best_val_score = 0

        past_averaged_params = []
        current_averaged_params = {x: y.clone().double() for x, y in self.model.state_dict().items()}

        step = 0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), position=0, leave=True)
        if self.args.global_attn == 0:
            model_params = [p for n, p in self.model.named_parameters() if not (n.endswith('_global.bias') or n.endswith('_global.weight'))]
        else:
            model_params = [p for p in self.model.parameters()]

        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0/self.args.grad_acc_steps)
        for step_, batch in pbar:
            if step_ % self.args.grad_acc_steps == 0:
                for ix in range(len(self.opt.param_groups)):
                    self.opt.param_groups[ix]['lr'] = self.lr_schedule[step]
                self.opt.zero_grad()
            
            tokens, mask, label, label_mask = (x.cuda() for x in batch)
            with torch.cuda.amp.autocast():
                outs = self.model(tokens, mask)
            ce = -(((outs * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
            if self.args.rce_weight == 0:
                l = ce
            else:
                rce = -(((torch.exp(outs) * torch.log_softmax(label, -1)).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
                l = ce * self.args.ce_weight + rce * self.args.rce_weight
            
            scaler.scale(l).backward()
            ls.append(l.detach())
        
            if step_ % self.args.grad_acc_steps == self.args.grad_acc_steps - 1:
                if self.args.max_grad_norm is not None:
                    scaler.unscale_(self.opt)
                    norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model_params]))
                    # norms.append(norm)
                    if torch.isfinite(norm):
                        grad_mult = (self.args.max_grad_norm / (norm + 1e-6)).clamp(max=1.)
                        for p in model_params:
                            p.grad.detach().mul_(grad_mult)
                scaler.step(self.opt)#.step()
                scaler.update()

                with torch.no_grad():
                    current_averaged_params = {x: y.clone().double() + current_averaged_params[x] for x, y in self.model.state_dict().items()}
                    match_updates = calc_acc(outs, label, label_mask)
                    train_matches += match_updates[0]
                    train_labels += match_updates[1]
                

                description = f"TRAIN  STEP {step} loss: {torch.stack(ls).mean().item(): .4f}"
                pbar.set_description(description)

                # validation
                if step % self.args.eval_interval == (self.args.eval_interval - 1):
                    log_dict = {}
                    norms = []
                    torch.autograd.set_grad_enabled(False)
                    params_backup = {x: y.cpu().clone() for x, y in self.model.state_dict().items()}

                    train_accs = (train_matches / train_labels).cpu().numpy()
                    train_labels = train_labels.cpu().numpy()
                    log_dict.update(make_match_dict(torch.stack(ls).mean().item(), train_accs, train_labels, self.label_names, 'Train'))
                    ls = []
                    train_matches = torch.zeros(16)
                    train_labels = torch.zeros(16)

                    if len(past_averaged_params) == 25:
                        past_averaged_params = past_averaged_params[-24:]

                    past_averaged_params.append({x: (y * (1 / self.args.eval_interval)).cpu().half() for x, y in current_averaged_params.items()})

                    if step > self.args.start_eval_at:
                        evaluation_params = None
                        for params_ix in range(1, len(past_averaged_params) + 1):
                            if evaluation_params is None:
                                evaluation_params = {x: y.double() for x, y in past_averaged_params[-params_ix].items()}
                            else:
                                evaluation_params = {x: y.double() + evaluation_params[x] 
                                                    for x, y in past_averaged_params[-params_ix].items()}
                            # 과거꺼를 더하여?
                            if params_ix >= 10 and params_ix % 5 == 0:
                                self.model.cpu()
                                self.model.load_state_dict({x: (y / params_ix).float() for x, y in evaluation_params.items()})
                                self.model.cuda()
                                val_ce, val_accs, val_labels, f1s, rec, prec = self.validate()
                                val_score = f1s[0]
                                print(val_score)
                                if val_score >= best_val_score:
                                    best_val_score = val_score
                                    torch.save(self.model.state_dict(), f'result/baseline_fold{self.args.val_fold}.pt')

                                log_dict.update(make_match_dict(val_ce, val_accs, val_labels, self.label_names, f'ValSWA{params_ix*self.args.eval_interval}', (f1s, rec, prec)))

                    self.model.cpu()
                    self.model.load_state_dict(params_backup)
                    self.model.cuda()
                    current_averaged_params = {x: y.clone() for x, y in self.model.state_dict().items()}

                    torch.autograd.set_grad_enabled(True)
                    wandb.log(log_dict)        

                step += 1

        return best_val_score
