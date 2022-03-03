
import os.path as osp
import numpy as np
import wandb

from tqdm.auto import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR


from .metric import calc_acc, process_sample, make_match_dict

class Trainer():
    def __init__(self, args, model, train_loader, valid_loader, scheduler, optimizer, criterion, class_names):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.class_names = class_names

        if self.args.swa:
            self.set_swa()

        # swa will run the scheduler so disable it
        self.run_scheduler = True
        self.val_model = self.model
    
    def set_swa(self):
        """Setting for Stochastic Weighted Average
        
        will use the scheduler that is already provided
        """
        # global_step - the total steps among the whole training
        # swa_step - the step after the first update of the swa model
        self.global_step = 0
        self.swa_step = 0

        # SWA - stochastic weight averaging
        self.swa_model = AveragedModel(self.model)
        # self.swa_scheduler = SWALR(self.optimizer,
        #                            swa_lr=self.args.lr,
        #                            anneal_epochs=self.args.steps_per_epoch,
        #                            anneal_strategy='cos')

        # When to start SWA?
        # setting aims to start swa when learning rate hits the minimum
        half_cycle_steps = self.args.steps_per_epoch // 2
        if self.args.scheduler in ['plateau',
                                   'custom_warmup',
                                   'cosine_annealing_warmup_restart']:
            cycle_n = 2
        elif self.args.scheduler in ['cosine_annealing']:
            cycle_n = 3
        self.swa_start = half_cycle_steps * cycle_n

        # swa will be saved every steps
        self.swa_save_per_steps = self.args.steps_per_epoch // self.args.swa_update_per_epoch


    def process_swa(self):
        """Processing for Stochastic Weighted Average
        
        when to update swa model
        - At the beginning when first swa starts
        - Every `swa_save_per_steps` steps
        """
        if (self.global_step + 1) > self.swa_start:
            if self.swa_step == 0:
                self.swa_model.update_parameters(self.model)
                self.val_model = self.swa_model.module
                # self.run_scheduler = False
            elif (self.swa_step + 1) % self.swa_save_per_steps == 0:
                self.swa_model.update_parameters(self.model)

            # self.swa_scheduler.step()
            self.swa_step += 1

        self.global_step += 1

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train_one_epoch(self, epoch):
        self.model.train()

        losses = []
        train_matches = torch.zeros(16)
        train_labels = torch.zeros(16)

        if self.args.global_attn == 0:
            model_params = [p for n, p in self.model.named_parameters() if not (n.endswith('_global.bias') or n.endswith('_global.weight'))]
        else:
            model_params = [p for p in self.model.parameters()]
        
        # scaler = GradScaler(65536.0 / self.args.grad_acc_steps)
        scaler = GradScaler()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, batch in pbar:

            if (step + 1) % self.args.grad_acc_steps == 0:
                self.optimizer.zero_grad()
            
            tokens, mask, label, class_weight = (x.to(self.args.device) for x in batch)
            with autocast():
                outs = self.model(tokens, mask)
                # loss = self.criterion(outs, label, class_weight=class_weight)
                loss = self.criterion(outs, label, class_weight=class_weight) / self.args.grad_acc_steps

            # loss
            scaler.scale(loss).backward()
            losses.append(loss.detach())
        
            # optimizer
            if (step + 1) % self.args.grad_acc_steps == 0:
                if self.args.max_grad_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                scaler.step(self.optimizer)
                scaler.update()

            # SWA - stochastic weight averaging
            """swa scheduler is disabled"""
            if self.args.swa:
                self.process_swa()

            # scheduler
            if self.run_scheduler:
                if self.args.scheduler == 'custom_warmup':
                    for g_i in range(len(self.optimizer.param_groups)):
                        self.optimizer.param_groups[g_i]['lr'] = self.scheduler[step]
                elif self.args.scheduler not in ['plateau']:
                    self.scheduler.step()

            # metric
            with torch.no_grad():
                match_updates = calc_acc(outs, label, class_weight)
                train_matches += match_updates[0]
                train_labels += match_updates[1]
                train_acc_per_class = train_matches / train_labels
                train_acc = train_acc_per_class.mean().item()

                # log 
                if (step + 1) % self.args.print_acc == 0:
                    print(f'Epoch: {epoch} | Step: {step + 1} | Train Acc per class: {train_acc_per_class}')
            
            description = f"[ TRAIN ] epoch {epoch} lr {self.lr():.7f} acc: {train_acc:.4f} loss: {torch.stack(losses).mean().item(): .4f}"
            pbar.set_description(description)

        return train_acc

    
    def valid_one_epoch(self, epoch):
        self.model.eval()

        losses = []
        val_matches = torch.zeros(16)
        val_labels = torch.zeros(16)
        match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s = np.zeros(8)
        rec = np.zeros(7)
        prec = np.zeros(7)

        for tokens, mask, labels, labels_mask, bounds, gt_dicts, index_map, num_tokens in tqdm(self.valid_loader, total=len(self.valid_loader)):
            with torch.no_grad():
                tokens, mask, label, class_weight = (x.to(self.args.device) for x in (tokens, mask, labels, labels_mask))

                with autocast():
                    outs = self.val_model(tokens, mask)
                    loss = self.criterion(outs, label, class_weight=class_weight)

                losses.append(loss)

                match_updates = calc_acc(outs, label, class_weight)
                val_matches += match_updates[0]
                val_labels += match_updates[1]
                for sample_ix, num in enumerate(num_tokens):
                    match_stats = process_sample(outs[sample_ix], labels[sample_ix], 
                                                    index_map[sample_ix], bounds[sample_ix],
                                                    gt_dicts[sample_ix],
                                                    num, match_stats, min_len=self.args.min_len)

        # validation Acc per class log
        valid_acc_per_class = val_matches / val_labels
        print(f'Valid Acc per class: {valid_acc_per_class}')

        for ix in range(1, 8):
            f1s[ix] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + .5 * (match_stats[ix]['fp'] + match_stats[ix]['fn']))
            rec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fn'])
            prec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fp'])

        f1s[0] = np.mean(f1s[1:])

        val_accs = (val_matches / val_labels).cpu().numpy()
        val_labels = val_labels.cpu().numpy()

        return torch.stack(losses).mean().item(), val_accs, val_labels, f1s, rec, prec

    def train(self):

        best_f1 = 0
        for epoch in range(1, self.args.epochs + 1):

            # train
            train_acc = self.train_one_epoch(epoch)

            # validation
            val_ce, val_accs, val_labels, f1s, rec, prec = self.valid_one_epoch(epoch)
            val_score = f1s[0]

            print(f'{val_score}')

            log_dict = {}
            log_dict.update(make_match_dict(val_ce, val_accs, val_labels, self.class_names, f'ValSWA', (f1s, rec, prec)))
            wandb.log(log_dict) 

            if val_score > best_f1:
                best_f1 = val_score
                save_name = f"debertav3_fold{self.args.val_fold}_f1{best_f1:.4f}.pth"
                if best_f1 > 0.683:
                    torch.save(self.val_model.state_dict(), osp.join(self.args.save_path, save_name))
                    print("save model .....")

            # scheduler
            if self.run_scheduler and self.args.scheduler == 'plateau':
                self.scheduler.step(best_f1)
        
        # save before the training ends
        save_name = f"debertav3_fold{self.args.val_fold}_f1{best_f1:.4f}.pth"
        if best_f1 > 0.683:
            torch.save(self.val_model.state_dict(), osp.join(self.args.save_path, save_name))
            print("save model .....")

        return best_f1


