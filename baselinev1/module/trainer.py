
import os.path as osp
import numpy as np
import wandb

from tqdm.auto import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR


from .metric import calc_acc, process_sample, init_match_dict, make_match_dict

# grad scaler enum helper
# reference - https://github.com/pytorch/pytorch/blob/1a8bd1a7eb91063d55f00c120ba29361860e6c42/torch/cuda/amp/grad_scaler.py#L31
# ---------------------------------------------------------------
from enum import Enum

class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2
# ---------------------------------------------------------------

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

        start_epoch = max(int(self.args.epochs * self.args.swa_start_ratio), 1)
        print(f'[ SWA ] - Applied from {start_epoch + 1} epochs')
        if self.args.scheduler in ['plateau',
                                   'custom_warmup',
                                   'cosine_annealing_warmup_restart']:
            cycle_n = start_epoch * 2
        elif self.args.scheduler in ['cosine_annealing']:
            # cosine annealing is setted as one cycle
            # to assure that the learning rate is minimum when the model is saved
            cycle_n = 1 + start_epoch * 2
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
                print('-' * 20)
                print('[ SWA ] - First update of the SWA model')
                print('-' * 20)
                # self.run_scheduler = False
            elif (self.swa_step + 1) % self.swa_save_per_steps == 0:
                self.swa_model.update_parameters(self.model)

            # self.swa_scheduler.step()
            self.swa_step += 1

        self.global_step += 1

    def process_sam(self, tokens, mask, label, class_weight, scaler, losses):
        """Processing for (Adaptive) SAM Optimizer
        
        Sharpness-Aware Minimization for Efficiently Improving Generalization
        - Cannot be used with Gradient Accumulation
        - Every `swa_save_per_steps` steps
        """
        if self.args.grad_acc_steps != 1:
            raise Exception('sam optimizer cannot be worked with gradient accumulation. disable it.')

        ### first
        # --------------------------------------------
        with autocast():
            outs = self.model(tokens, mask)
            loss = self.criterion(outs, label, class_weight=class_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.first_step(zero_grad=True)

        ### Second
        # --------------------------------------------
        with autocast():
            outs = self.model(tokens, mask)
            loss = self.criterion(outs, label, class_weight=class_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.second_step(zero_grad=True)
        # --------------------------------------------

        # save loss
        losses.append(loss.detach())

        return outs

        # below code is not used due to nan loss error
        # --------------------------------------------
        with autocast():
            outs = self.model(tokens, mask)
            loss = self.criterion(outs, label, class_weight=class_weight)

        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        # code for amp compatibility with sam
        # replacement for scaler.step(self.optimizer)
        self.optimizer.first_step(zero_grad=True)
        optimizer_state = scaler._per_optimizer_states[id(self.optimizer)]
        optimizer_state["stage"] = OptState.STEPPED
        scaler.update()
        # --------------------------------------------


    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train_one_epoch(self, epoch):
        self.model.train()

        losses = []
        match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s = np.zeros(8)
        rec = np.zeros(7)
        prec = np.zeros(7)
        # train_matches = torch.zeros(16)
        # train_labels = torch.zeros(16)

        # scaler = GradScaler(65536.0 / self.args.grad_acc_steps)
        scaler = GradScaler()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, batch in pbar:

            if (step + 1) % self.args.grad_acc_steps == 0:
                self.optimizer.zero_grad()
            
            tokens, mask, label, class_weight = (x.to(self.args.device) for x in batch)
            """Sam Optimizer process is decoupled due to different training approach
           
            CAUTION:
            - SAM optimizers cannot be used with Gradient Accumulation
            """
            if self.args.optimizer == 'sam':
                outs = self.process_sam(tokens, mask, label, class_weight, scaler, losses)
            else:
                with autocast():
                    outs = self.model(tokens, mask)
                    # loss = self.criterion(outs, label, class_weight=class_weight)
                    loss = self.criterion(outs, label, class_weight=class_weight) / self.args.grad_acc_steps

                # loss
                create_graph = True if self.args.optimizer == 'adahessian' else False
                scaler.scale(loss).backward(create_graph=create_graph)
                losses.append(loss.item())
            
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
                ...
                # match_updates = calc_acc(outs, label, class_weight)
                # train_matches += match_updates[0]
                # train_labels += match_updates[1]
                # train_acc_per_class = train_matches / train_labels
                # train_acc = train_acc_per_class.mean().item()

                # log 
                if (step + 1) % self.args.print_f1_per_step == 0:
                    ...
                    # print(f'Epoch: {epoch} | Step: {step + 1} | Train Acc per class: {train_acc_per_class}')
            
            description = f"[ TRAIN ] epoch {epoch} lr {self.lr():.7f} loss: {np.array(losses).mean(): .4f}"
            pbar.set_description(description)

    
    def valid_one_epoch(self, epoch):
        self.model.eval()

        losses = []

        # 3-way validation strategy
        # val_matches = torch.zeros(16)
        # val_labels = torch.zeros(16)
        match_stats_bug = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s_bug = np.zeros(8)
        rec_bug = np.zeros(7)
        prec_bug = np.zeros(7)
        match_stats_clean = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s_clean = np.zeros(8)
        rec_clean = np.zeros(7)
        prec_clean = np.zeros(7)
        match_stats_wonho = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s_wonho = np.zeros(8)
        rec_wonho = np.zeros(7)
        prec_wonho = np.zeros(7)

        for tokens, mask, labels, labels_mask, bounds, gt_dicts, index_map, num_tokens in tqdm(self.valid_loader, total=len(self.valid_loader)):
            with torch.no_grad():
                tokens, mask, label, class_weight = (x.to(self.args.device) for x in (tokens, mask, labels, labels_mask))

                with autocast():
                    outs = self.val_model(tokens, mask)
                    loss = self.criterion(outs, label, class_weight=class_weight)

                losses.append(loss)

                # match_updates = calc_acc(outs, label, class_weight)
                # val_matches += match_updates[0]
                # val_labels += match_updates[1]
                for sample_ix, num in enumerate(num_tokens):
                    match_stats_bug = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_bug, version='bug')
                    match_stats_clean = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_clean, version='clean')
                    match_stats_wonho = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_wonho, version='wonho')

        print(f'Currenlty on [ bug ] version')
        print(f'Predicted match stats: {match_stats_bug}\n')
        print(f'Currenlty on [ clean ] version')
        print(f'Predicted match stats: {match_stats_clean}\n')
        print(f'Currenlty on [ wonho ] version')
        print(f'Predicted match stats: {match_stats_wonho}\n')
        print('-' * 50)

        # validation Acc per class log
        # valid_acc_per_class = val_matches / val_labels
        # print(f'Valid Acc per class: {valid_acc_per_class}')

        for ix in range(1, 8):
            f1s_bug[ix] = match_stats_bug[ix]['tp'] / (1e-7 + match_stats_bug[ix]['tp'] + .5 * (match_stats_bug[ix]['fp'] + match_stats_bug[ix]['fn']))
            rec_bug[ix - 1] = match_stats_bug[ix]['tp'] / (1e-7 + match_stats_bug[ix]['tp'] + match_stats_bug[ix]['fn'])
            prec_bug[ix - 1] = match_stats_bug[ix]['tp'] / (1e-7 + match_stats_bug[ix]['tp'] + match_stats_bug[ix]['fp'])

            f1s_clean[ix] = match_stats_clean[ix]['tp'] / (1e-7 + match_stats_clean[ix]['tp'] + .5 * (match_stats_clean[ix]['fp'] + match_stats_clean[ix]['fn']))
            rec_clean[ix - 1] = match_stats_clean[ix]['tp'] / (1e-7 + match_stats_clean[ix]['tp'] + match_stats_clean[ix]['fn'])
            prec_clean[ix - 1] = match_stats_clean[ix]['tp'] / (1e-7 + match_stats_clean[ix]['tp'] + match_stats_clean[ix]['fp'])

            f1s_wonho[ix] = match_stats_wonho[ix]['tp'] / (1e-7 + match_stats_wonho[ix]['tp'] + .5 * (match_stats_wonho[ix]['fp'] + match_stats_wonho[ix]['fn']))
            rec_wonho[ix - 1] = match_stats_wonho[ix]['tp'] / (1e-7 + match_stats_wonho[ix]['tp'] + match_stats_wonho[ix]['fn'])
            prec_wonho[ix - 1] = match_stats_wonho[ix]['tp'] / (1e-7 + match_stats_wonho[ix]['tp'] + match_stats_wonho[ix]['fp'])

        f1s_bug[0] = np.mean(f1s_bug[1:])
        f1s_clean[0] = np.mean(f1s_clean[1:])
        f1s_wonho[0] = np.mean(f1s_wonho[1:])

        # val_accs = (val_matches / val_labels).cpu().numpy()
        # val_labels = val_labels.cpu().numpy()
        val_loss = torch.stack(losses).mean().item()

        return val_loss, f1s_bug, rec_bug, prec_bug, f1s_clean, rec_clean, prec_clean, f1s_wonho, rec_wonho, prec_wonho

    def train(self):

        best_f1_bug = 0
        best_f1_clean = 0
        best_f1_wonho = 0
        for epoch in range(1, self.args.epochs + 1):

            # train
            self.train_one_epoch(epoch)

            # validation
            val_loss, f1s_bug, rec_bug, prec_bug, f1s_clean, rec_clean, prec_clean, f1s_wonho, rec_wonho, prec_wonho = self.valid_one_epoch(epoch)
            val_score_bug = f1s_bug[0]
            val_score_clean = f1s_clean[0]
            val_score_wonho = f1s_wonho[0]

            print(f'Validation [ Bug ] {val_score_bug}')
            print(f'Validation [ Clean ] {val_score_clean}')
            print(f'Validation [ Wonho ] {val_score_wonho}')

            # wandb - logging
            if self.args.wandb:
                log_dict = init_match_dict(val_loss)
                log_dict = make_match_dict(log_dict, self.class_names, 'Bug', (f1s_bug, rec_bug, prec_bug))
                log_dict = make_match_dict(log_dict, self.class_names, 'Clean', (f1s_clean, rec_clean, prec_clean))
                log_dict = make_match_dict(log_dict, self.class_names, 'Wonho', (f1s_wonho, rec_wonho, prec_wonho))
                wandb.log(log_dict) 

            # saving model
            if val_score_bug > best_f1_bug:
                best_f1_bug = val_score_bug
                save_name = f"bug_debertav3_fold{str(self.args.val_fold)}_f1{best_f1_bug:.4f}.pth"
                if best_f1_bug > 0.683:
                    torch.save(self.val_model.state_dict(), osp.join(self.args.save_path, save_name))
                    print("[ Bug version ] saving model")
            if val_score_clean > best_f1_clean:
                best_f1_clean = val_score_clean
                save_name = f"clean_debertav3_fold{str(self.args.val_fold)}_f1{best_f1_clean:.4f}.pth"
                if best_f1_clean > 0.68:
                    torch.save(self.val_model.state_dict(), osp.join(self.args.save_path, save_name))
                    print("[ Clean version ] saving model")
            if val_score_wonho > best_f1_wonho:
                best_f1_wonho = val_score_wonho
                save_name = f"wonho_debertav3_fold{str(self.args.val_fold)}_f1{best_f1_wonho:.4f}.pth"
                if best_f1_wonho > 0.683:
                    torch.save(self.val_model.state_dict(), osp.join(self.args.save_path, save_name))
                    print("[ Wonho version ] saving model")

            # scheduler
            if self.run_scheduler and self.args.scheduler == 'plateau':
                self.scheduler.step(-val_loss)
        
        return best_f1_bug, best_f1_clean, best_f1_wonho


