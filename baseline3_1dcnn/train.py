import warnings
import sys
warnings.filterwarnings('ignore')

import argparse
import os
import random

import torch
import wandb
import pandas as pd
import numpy as np

import torch.distributed as dist
import torch.nn as nn

from ast import literal_eval
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from module.utils import *
from module.dataset import *
from module.trainer import Trainer
from model.model import TvmLongformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_config():
    parser = argparse.ArgumentParser(description="use huggingface models")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--min_len", default=0, type=int)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--weights_pow", default=0.1, type=float)
    parser.add_argument("--use_groupped_weights", default=False, type=bool)
    parser.add_argument("--global_attn", default=False, type=int)
    parser.add_argument("--label_smoothing", default= 0.1, type=float)
    parser.add_argument("--extra_dense", default= False, type=bool)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_acc_steps", default=2, type=int)
    parser.add_argument("--grad_checkpt", default=True, type=bool)
    parser.add_argument("--data_prefix", default='', type=str)
    parser.add_argument("--max_grad_norm", default=35 * 8, type=int)
    parser.add_argument("--start_eval_at", default=0, type=int)
    parser.add_argument("--lr", default=32e-6, type=float)
    parser.add_argument("--min_lr", default=32e-6, type=float)
    parser.add_argument("--dataset_version", default=2, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--rce_weight", default=0.1, type=float)
    parser.add_argument("--ce_weight", default=0.9, type=float)
    parser.add_argument("--decay_bias", default=False, type=bool)
    parser.add_argument("--val_fold", default=0, type=int)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.local_rank !=-1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device("cuda", args.local_rank)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()  

    else:
       args.device = torch.device("cuda")
       args.rank = 0

    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wandb_setting(args):
    wandb.login()
    run = wandb.init(entity='dlrgy22', project='feedback_deberta_large')
    run.name = f'v3_fold{args.val_fold}_minlr{args.min_lr}_maxlr{args.lr}_wd{args.wd}_warmup{args.warmup_steps}_gradnorm{args.max_grad_norm}_biasdecay{args.decay_bias}_ls{args.label_smoothing}_wp{args.weights_pow}_data{args.dataset_version}_rce{args.rce_weight}'

def get_data_files(args):
    token_weights = get_token_weights(args.use_groupped_weights, args.weights_pow)
    data = get_prepare_data()
    csv = pd.read_csv('../input/feedback-prize-2021/sergei_train.csv')
    all_texts = get_all_texts()
    id_to_ix_map = get_id_to_ix_map()
    data_splits = get_fold_data()

    train_ids = [id_to_ix_map[x] for fold in range(5) if fold != args.val_fold for x in data_splits[args.seed][250]['normed'][fold]]
    val_files = data_splits[args.seed][250]['normed'][args.val_fold]
    val_ids = [id_to_ix_map[x] for x in val_files]

    return all_texts, token_weights, data, csv, train_ids, val_ids, val_files

def get_dataloader(train_ids, val_ids, data, csv, all_texts, val_files, label_names, token_weights, args):
    train_dataset = TrainDataset(train_ids, data, args.label_smoothing, token_weights, args.data_prefix)
    val_dataset = ValDataset(val_ids, data, csv, all_texts, val_files, label_names, token_weights)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=args.batch_size, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=4, num_workers=8, persistent_workers=True)

    return train_dataloader, val_dataloader

def get_model(args, train_dataloader):
    model = TvmLongformer(args.grad_checkpt, args.extra_dense).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0

    weights = []
    biases = []
    for n, p in model.named_parameters():
        if n.startswith('feats.embeddings') or 'LayerNorm' in n or n.endswith('bias'):
            biases.append(p)
        else:
            weights.append(p)

    opt = torch.optim.AdamW([{'params': weights, 'weight_decay': args.wd, 'lr': 0},
                     {'params': biases, 'weight_decay': 0 if not args.decay_bias else args.wd, 'lr': 0}])

    lr_schedule = np.r_[np.linspace(0, args.lr, args.warmup_steps),
                    (np.cos(np.linspace(0, np.pi, len(train_dataloader)*args.epochs - args.warmup_steps)) * .5 + .5) * (args.lr - args.min_lr)
                    + args.min_lr]


    return model, opt, lr_schedule

if __name__ == "__main__":
    seed_everything(42)
    args = get_config()
    wandb_setting(args)
    label_names = ['None', 'Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Counterclaim', 'Rebuttal']
    all_texts, token_weights, data, csv, train_ids, val_ids, val_files = get_data_files(args)
    train_dataloader, val_dataloader = get_dataloader(train_ids, val_ids, data, csv, all_texts, val_files, label_names, token_weights, args)
    model, opt, lr_schedule = get_model(args, train_dataloader)
    trainer = Trainer(model, train_dataloader, val_dataloader, lr_schedule, opt, label_names, "result/noSWA", args)
    best_score = trainer.train()
    print(best_score)




    
