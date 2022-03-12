
import os
import os.path as osp
import copy
import pickle
import numpy as np
import pandas as pd
import wandb

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from module.utils import get_data_files
from module.dataset import get_augmenter
from module.dataset import get_dataloader
from module.dataset import OnlineTrainDataset
from module.dataset import train_collate_fn

from model.model import get_model
from module.optimizer import get_optimizer
from module.scheduler import get_scheduler
from module.loss import get_criterion

from .metric import calc_acc, process_sample, init_match_dict, make_match_dict

from sklearn.metrics import accuracy_score

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
    def __init__(self, args, class_names):
        self.args = args
        self.class_names = class_names


        # load data
        if self.args.online_dataset:
            self.train_augmenter, self.val_augmenter = get_augmenter(self.args)
            self.train_loader, self.valid_loader = self.get_online_dataloader()
        else:
            self.train_loader, self.valid_loader = self.get_offline_dataloader()

        self.args.steps_per_epoch = len(self.train_loader)

        # prepare model, etc
        self.model = get_model(args)
        self.optimizer = get_optimizer(args, self.model)
        self.scheduler = get_scheduler(args, self.optimizer)
        self.criterion = get_criterion(args)            

        # noise filter
        if self.args.noise_filter:
            self.train_len = len(self.train_text_ids)
            self.mean_teacher = copy.deepcopy(self.model)
            self.ensemble_preds = np.zeros((self.train_len, 2048, 15), dtype='f4')
            self.sequential_loader = self.get_sequential_dataloader()


        if self.args.swa:
            self.set_swa()

        # when swa run the own scheduler flag will be set as False
        self.run_scheduler = True
        self.val_model = self.model


    def get_offline_dataloader(self):
        all_texts, token_weights, data, csv, train_ids, val_ids, train_text_ids, val_text_ids = get_data_files(self.args)
        train_dataloader, val_dataloader = get_dataloader(
            self.args,
            train_ids,
            val_ids,
            data,
            csv,
            all_texts,
            train_text_ids,
            val_text_ids,
            self.class_names,
            token_weights,
        )

        return train_dataloader, val_dataloader

    def get_online_dataloader(self):
        # the train_text_ids will be filtered while noise filtering
        # the dataset will be recreated based on filtered train_text_ids
        all_texts, token_weights, data, csv, train_ids, val_ids, train_text_ids, val_text_ids = get_data_files(self.args)

        # save train data's for further noise filtering
        if self.args.noise_filter:
            self.all_texts = all_texts
            self.train_text_ids = train_text_ids
            self.csv = csv
            self.token_weights = token_weights

        train_dataloader, val_dataloader = get_dataloader(
            self.args,
            train_ids,
            val_ids,
            data,
            csv,
            all_texts,
            train_text_ids,
            val_text_ids,
            self.class_names,
            token_weights,
            train_augmenter=self.train_augmenter,
            valid_augmenter=self.val_augmenter
        )

        return train_dataloader, val_dataloader

    def get_sequential_dataloader(self):
        train_dataset = OnlineTrainDataset(self.args, self.train_text_ids, self.all_texts, self.csv, self.token_weights, self.train_augmenter)
        sequential_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.args.batch_size, num_workers=self.args.num_worker, shuffle=False)

        return sequential_dataloader

    def get_filter_dataloader(self, filter_text_ids):
        filter_dataset = OnlineTrainDataset(self.args, filter_text_ids, self.all_texts, self.csv, self.token_weights, self.train_augmenter)
        filter_dataloader = DataLoader(filter_dataset, collate_fn=train_collate_fn, batch_size=self.args.batch_size, num_workers=self.args.num_worker, shuffle=True)

        return filter_dataloader
    
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

    def filter_one_epoch(self, epoch, filter_loader):
        self.mean_teacher.eval()
        alpha = self.args.noise_filter_alpha
        filter_thres = self.args.noise_filter_acc_thres
        train_accs = np.zeros((self.train_len), dtype='f4')

        with torch.no_grad():
            for step, batch in tqdm(enumerate(filter_loader), total=len(filter_loader)):

                # calculate outs
                tokens, mask, label, class_weight = (x.to(self.args.device) for x in batch)
                outs = self.mean_teacher(tokens, mask)
                outs = outs.cpu().detach().numpy()
                label = label.cpu().numpy()

                # exponential moving average of predictions
                outs_batch_size, outs_seq_size = outs.shape[0], outs.shape[1]
                
                start_idx = step * self.args.batch_size
                end_idx = start_idx + outs_batch_size
            
                self.ensemble_preds[start_idx:end_idx, :outs_seq_size] = alpha * self.ensemble_preds[start_idx:end_idx, :outs_seq_size] + (1 - alpha) * outs

                # calculate accuracy of each text
                preds = self.ensemble_preds[start_idx:end_idx, :outs_seq_size].argmax(-1)
                trues = label.argmax(-1)

                train_accs[start_idx:end_idx] = [accuracy_score(true, pred) for true, pred in zip(trues, preds)]

            # create train texts dataframe with accuracy score
            train_df = pd.DataFrame({'text_ids': self.train_text_ids, 'accuracy': train_accs})

            # filter the texts by accuracy
            filter_text_ids = train_df.query('accuracy > @filter_thres').text_ids.values.tolist()

        # debugging
        # if len(filter_text_ids) == 0:
        #     filter_text_ids = self.train_text_ids[:100]

        # log
        print('-' * 20)
        print(f'[ Filter ] - Process step {epoch - 1}')
        print(f'                - alpha {alpha}')
        print(f'                - filter accuracy threshold {filter_thres}')
        print(f'             Total texts {self.train_len}')
        print(f'             Clean texts {len(filter_text_ids)} ({100 * len(filter_text_ids) / self.train_len:.2f} %)')
        print('-' * 20, '\n')

        return filter_text_ids


    def train_one_epoch(self, epoch, train_loader):
        self.model.train()
        if self.args.noise_filter:
            self.mean_teacher.train()

        losses = []
        match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        f1s = np.zeros(8)
        rec = np.zeros(7)
        prec = np.zeros(7)
        # train_matches = torch.zeros(16)
        # train_labels = torch.zeros(16)

        # scaler = GradScaler(65536.0 / self.args.grad_acc_steps)
        scaler = GradScaler()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
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
            elif False:
                with autocast():
                    outs = self.model(tokens, mask)
                    # loss = self.criterion(outs, label, class_weight=class_weight)
                    loss = self.criterion(outs, label, class_weight=class_weight) / self.args.grad_acc_steps

                # loss
                scaler.scale(loss).backward()
                losses.append(loss.item())
            
                # optimizer
                if (step + 1) % self.args.grad_acc_steps == 0:
                    if self.args.max_grad_norm is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    scaler.step(self.optimizer)
                    scaler.update()

            else:
                outs = self.model(tokens, mask)
                loss = self.criterion(outs, label, class_weight=class_weight)

                # mean teacher
                if self.args.noise_filter:
                    with torch.no_grad():
                        mean_outs = self.mean_teacher(tokens, mask)

                    # consistency loss
                    const_outs = F.log_softmax(outs, -1)
                    const_mean_outs = F.log_softmax(mean_outs, -1)
                    const_loss = F.mse_loss(const_outs, const_mean_outs)
                    # print('consistency_loss', const_loss.item())

                    loss = loss + self.args.const_loss_weight * const_loss

                # don't know why but create_graph=True doens't work..
                if self.args.optimizer == 'adahessian':
                    loss.backward(create_graph=True)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())

                # mean teacher
                if self.args.noise_filter:
                    # update mean teacher model weight
                    alpha = self.args.mean_teacher_alpha
                    for mean_param, param in zip(self.mean_teacher.parameters(), self.model.parameters()):
                        mean_param.data = alpha * mean_param.data + (1 - alpha) * param.data

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
        self.val_model.eval()

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

                # with autocast():
                #     outs = self.val_model(tokens, mask)
                #     loss = self.criterion(outs, label, class_weight=class_weight)
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

    def mean_teacher_valid_one_epoch(self, epoch):
        self.mean_teacher.eval()

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

                # with autocast():
                #     outs = self.val_model(tokens, mask)
                #     loss = self.criterion(outs, label, class_weight=class_weight)
                outs = self.mean_teacher(tokens, mask)
                loss = self.criterion(outs, label, class_weight=class_weight)

                losses.append(loss)

                # match_updates = calc_acc(outs, label, class_weight)
                # val_matches += match_updates[0]
                # val_labels += match_updates[1]
                for sample_ix, num in enumerate(num_tokens):
                    match_stats_bug = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_bug, version='bug')
                    match_stats_clean = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_clean, version='clean')
                    match_stats_wonho = process_sample(outs[sample_ix], labels[sample_ix], index_map[sample_ix], bounds[sample_ix], gt_dicts[sample_ix], num, match_stats_wonho, version='wonho')

        print(f'Currenlty on [ Mean Teacher bug ] version')
        print(f'Predicted match stats: {match_stats_bug}\n')
        print(f'Currenlty on [ Mean Teacher clean ] version')
        print(f'Predicted match stats: {match_stats_clean}\n')
        print(f'Currenlty on [ Mean Teacher wonho ] version')
        print(f'Predicted match stats: {match_stats_wonho}\n')
        print('-' * 50)

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

        self.mean_teacher_val_score_bug = np.mean(f1s_bug[1:])
        self.mean_teacher_val_score_clean = np.mean(f1s_clean[1:])
        self.mean_teacher_val_score_wonho = np.mean(f1s_wonho[1:])

        print(f'Mean Teacher Validation [ Bug ] {self.mean_teacher_val_score_bug}')
        print(f'Mean Teacher Validation [ Clean ] {self.mean_teacher_val_score_clean}')
        print(f'Mean Teacher Validation [ Wonho ] {self.mean_teacher_val_score_wonho}')

    def train(self):

        # create folder to save model
        save_folder_path = osp.join(self.args.save_path, self.args.wandb_comment)
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        best_f1_bug = 0
        best_f1_clean = 0
        best_f1_wonho = 0

        if self.args.noise_filter:
            mean_teacher_best_f1_bug = 0
            mean_teacher_best_f1_clean = 0
            mean_teacher_best_f1_wonho = 0
        for epoch in range(1, self.args.epochs + 1):

            if self.args.noise_filter and epoch > 1:
                # noise filter
                filter_text_ids = self.filter_one_epoch(epoch, self.sequential_loader)
                filter_loader = self.get_filter_dataloader(filter_text_ids)
                self.train_one_epoch(epoch, filter_loader)
            else:
                # train
                self.train_one_epoch(epoch, self.train_loader)

            # validation
            val_loss, f1s_bug, rec_bug, prec_bug, f1s_clean, rec_clean, prec_clean, f1s_wonho, rec_wonho, prec_wonho = self.valid_one_epoch(epoch)
            if self.args.noise_filter:
                self.mean_teacher_valid_one_epoch(epoch)

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
                if self.args.noise_filter:
                    log_dict.update({'bug_mean_teacher_f1macro': self.mean_teacher_val_score_bug})
                    log_dict.update({'clean_mean_teacher_f1macro': self.mean_teacher_val_score_clean})
                    log_dict.update({'wonho_mean_teacher_f1macro': self.mean_teacher_val_score_wonho})

                wandb.log(log_dict) 


            # change saving model to mean teacher
            if self.args.noise_filter:
                temp_val_model = self.val_model
                self.val_model = self.mean_teacher

            # saving model
            if val_score_bug > best_f1_bug:
                best_f1_bug = val_score_bug
                save_name = f"bug_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{best_f1_bug:.4f}.pth"
                if best_f1_bug > 0.67:
                    torch.save(self.val_model.state_dict(), osp.join(save_folder_path, save_name))
                    print("[ Bug version ] saving model")
            if val_score_clean > best_f1_clean:
                best_f1_clean = val_score_clean
                save_name = f"clean_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{best_f1_clean:.4f}.pth"
                if best_f1_clean > 0.67:
                    torch.save(self.val_model.state_dict(), osp.join(save_folder_path, save_name))
                    print("[ Clean version ] saving model")
            if val_score_wonho > best_f1_wonho:
                best_f1_wonho = val_score_wonho
                save_name = f"wonho_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{best_f1_wonho:.4f}.pth"
                if best_f1_wonho > 0.67:
                    torch.save(self.val_model.state_dict(), osp.join(save_folder_path, save_name))
                    print("[ Wonho version ] saving model")

            # change saving model to original
            if self.args.noise_filter:
                self.val_model = temp_val_model

                # saving model
                if self.mean_teacher_val_score_bug > mean_teacher_best_f1_bug:
                    mean_teacher_best_f1_bug = self.mean_teacher_val_score_bug
                    save_name = f"bug_mean_teacher_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{mean_teacher_best_f1_bug:.4f}.pth"
                    if mean_teacher_best_f1_bug > 0.67:
                        torch.save(self.mean_teacher.state_dict(), osp.join(save_folder_path, save_name))
                        print("[ Bug version ] mean teacher saving model")
                if self.mean_teacher_val_score_clean > mean_teacher_best_f1_clean:
                    mean_teacher_best_f1_clean = self.mean_teacher_val_score_clean
                    save_name = f"clean_mean_teacher_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{mean_teacher_best_f1_clean:.4f}.pth"
                    if mean_teacher_best_f1_clean > 0.67:
                        torch.save(self.mean_teacher.state_dict(), osp.join(save_folder_path, save_name))
                        print("[ Clean version ] mean teacher saving model")
                if self.mean_teacher_val_score_wonho > mean_teacher_best_f1_wonho:
                    mean_teacher_best_f1_wonho = self.mean_teacher_val_score_wonho
                    save_name = f"wonho_mean_teacher_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}_f1{mean_teacher_best_f1_wonho:.4f}.pth"
                    if mean_teacher_best_f1_wonho > 0.67:
                        torch.save(self.mean_teacher.state_dict(), osp.join(save_folder_path, save_name))
                        print("[ Wonho version ] mean teacher saving model")

            # scheduler
            if self.run_scheduler and self.args.scheduler == 'plateau':
                self.scheduler.step(-val_loss)


        # save ensemble predictions and text_ids
        if self.args.noise_filter:
            token_distillation = [self.train_text_ids, self.ensemble_preds]

            # save as pickle
            save_name = f"token_distillation_debertav3_fold{str(self.args.val_fold)}_{self.args.wandb_comment}.pkl"
            with open(osp.join(save_folder_path, save_name), 'wb') as f:
                pickle.dump(token_distillation, f)
        
        return best_f1_bug, best_f1_clean, best_f1_wonho


