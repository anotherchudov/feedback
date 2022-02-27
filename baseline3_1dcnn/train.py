
import warnings
warnings.filterwarnings('ignore')

import os
import os.path as osp
import sys
import random

import argparse
import wandb
import pandas as pd
import numpy as np
from pickletools import optimize

import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification

# local codes
from module.utils import get_data_files
from module.dataset import get_dataloader
from module.loss import get_criterion
from module.optimizer import get_optimizer
from module.scheduler import get_scheduler
from model.model import get_model

from module.trainer import Trainer

# Hugging Face's Issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_config():
    parser = argparse.ArgumentParser(description="use huggingface models")
    parser.add_argument("--wandb_user", default='ducky', type=str)
    parser.add_argument("--wandb_project", default='feedback_deberta_large', type=str)
    parser.add_argument("--dataset_path", default='../../feedback-prize-2021', type=str)
    parser.add_argument("--save_path", default='result', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--min_len", default=0, type=int)
    parser.add_argument("--use_groupped_weights", default=False, type=bool)
    parser.add_argument("--global_attn", default=False, type=int)
    parser.add_argument("--label_smoothing", default= 0.1, type=float)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_acc_steps", default=2, type=int)
    parser.add_argument("--grad_checkpt", default=True, type=bool)
    parser.add_argument("--data_prefix", default='', type=str)
    parser.add_argument("--max_grad_norm", default=35 * 8, type=int)
    parser.add_argument("--start_eval_at", default=0, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--min_lr", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--weights_pow", default=0.1, type=float)
    parser.add_argument("--dataset_version", default=2, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--decay_bias", default=False, type=bool)
    parser.add_argument("--val_fold", default=0, type=int)
    parser.add_argument("--num_worker", default=8, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="do not modify!")
    parser.add_argument("--device", type=int, default=0, help="select the gpu device to train")

    # optimizer
    parser.add_argument("--rce_weight", default=0.1, type=float)
    parser.add_argument("--ce_weight", default=0.9, type=float)

    # model related arguments
    parser.add_argument("--model", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--cnn1d", default=False, type=bool)
    parser.add_argument("--extra_dense", default= False, type=bool)
    parser.add_argument("--dropout_ratio", default=0.0, type=float)

    args = parser.parse_args()

    if args.local_rank !=-1:
        print('[ DDP ] local rank', args.local_rank)
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device("cuda", args.local_rank)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()  

        # checking settings for distributed training
        assert args.batch_size % args.world_size == 0, f'--batch_size {args.batch_size} must be multiple of world size'
        assert torch.cuda.device_count() > args.local_rank, 'insufficient CUDA devices for DDP command'

        args.ddp = True
    else:
        args.device = torch.device("cuda", args.device)
        args.rank = -1
        args.ddp = False

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
    run = wandb.init(entity=args.wandb_user, project=args.wandb_project)
    run.name = f'v3_fold{args.val_fold}_minlr{args.min_lr}_maxlr{args.lr}_wd{args.weight_decay}_warmup{args.warmup_steps}_gradnorm{args.max_grad_norm}_biasdecay{args.decay_bias}_ls{args.label_smoothing}_wp{args.weights_pow}_data{args.dataset_version}_rce{args.rce_weight}'


if __name__ == "__main__":
    """
    supported model list
    - microsoft/deberta-v3-large

    DISCLAIMER:
    - currently only support single GPU training
    - arguments related to optimizer, scheduler, and loss function
      are not supported and directly assigned at the beginning of the code
    """
    args = get_config()
    seed_everything(args.seed)
    wandb_setting(args)

    class_names = ['None',
                   'Lead',
                   'Position',
                   'Evidence',
                   'Claim',
                   'Concluding Statement',
                   'Counterclaim',
                   'Rebuttal']

    # create directory to save model
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    # data
    all_texts, token_weights, data, csv, train_ids, val_ids, train_text_ids, val_text_ids = get_data_files(args)
    train_dataloader, val_dataloader = get_dataloader(args, train_ids, val_ids, data, csv, all_texts, val_text_ids, class_names, token_weights)

    # loss
    # args.criterion_list = ["crossentropy"]
    # args.criterion_ratio = [1.]

    args.criterion_list = ["custom_ce", "custom_rce"]
    args.criterion_ratio = [args.ce_weight, args.rce_weight]
    criterion = get_criterion(args)            

    # model
    args.model = 'microsoft/deberta-v3-large'
    model = get_model(args)

    # optimizer
    args.optimizer = "adamw"
    optimizer = get_optimizer(args, model)

    # scheduler
    args.steps_per_epoch = len(train_dataloader)
    args.scheduler = "custom_warmup"
    scheduler = get_scheduler(args, optimizer)

    # train
    trainer = Trainer(args, model, train_dataloader, val_dataloader, scheduler, optimizer, criterion, class_names)
    best_f1 = trainer.train()

    print(best_f1)
    