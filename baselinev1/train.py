

# system path for custom Hugging Face
# ------------------------------------------------------
import sys
sys.path.insert(0, './codes')
# ------------------------------------------------------

from lib2to3.pgen2 import token
import warnings
warnings.filterwarnings('ignore')

import os
import os.path as osp
import random

import argparse
import wandb
import pandas as pd
import numpy as np
from pickletools import optimize

import torch
import torch.nn as nn
import torch.distributed as dist

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
    parser.add_argument("--dataset_path", default='../../feedback-prize-2021', type=str)
    parser.add_argument("--save_path", default='result', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--min_len", default=0, type=int)
    parser.add_argument("--use_groupped_weights", default=False, type=bool)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_acc_steps", default=1, type=int)
    parser.add_argument("--grad_checkpt", default=True, type=bool)
    parser.add_argument("--data_prefix", default='', type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--start_eval_at", default=0, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--weights_pow", default=0.1, type=float)
    parser.add_argument("--dataset_version", default=2, type=int)
    parser.add_argument("--decay_bias", default=False, type=bool)
    parser.add_argument("--val_fold", default=0, type=int)
    parser.add_argument("--num_worker", default=8, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="do not modify!")
    parser.add_argument("--device", type=int, default=0, help="select the gpu device to train")

    # logging
    parser.add_argument("--wandb_user", default='ducky', type=str)
    parser.add_argument("--wandb_project", default='feedback_deberta_large', type=str)
    parser.add_argument("--wandb_comment", default="", type=str, help="comment will be added at the back of wandb project name")
    parser.add_argument("--print_f1_per_step", default=500, type=int, help="print f1 of each class every `print_acc` steps")

    # optimizer
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--rce_weight", default=0.1, type=float)
    parser.add_argument("--ce_weight", default=0.9, type=float)
    parser.add_argument("--nesterov", default=True, type=bool, help="use nesterov for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for SGD")

    # scheduler
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--gamma", default=0.9, type=float, help="gamma for cosine annealing warmup restart scheduler")
    parser.add_argument("--cycle_mult", default=1.0, type=float, help="cycle length adjustment for cosine annealing warmup restart scheduler")

    # model related arguments
    parser.add_argument("--model", default="microsoft/deberta-v3-large", type=str)
    parser.add_argument("--cnn1d", default=False, type=bool)
    parser.add_argument("--extra_dense", default= False, type=bool)
    parser.add_argument("--dropout_ratio", default=0.0, type=float)

    # swa
    parser.add_argument("--swa", action="store_true", help="use stochastic weight averaging")
    parser.add_argument("--swa_update_per_epoch", default=1, type=int)
    parser.add_argument("--swa_start_ratio", default=0.1, type=float, help="start swa after this ratio of total epochs")

    # online dataset
    parser.add_argument("--online_dataset", action="store_true", help="use dataset that directly preprocess text online")
    parser.add_argument("--noise_injection", default=False, type=bool, help="use noise injection")
    parser.add_argument("--back_translation", default=False, type=bool, help="use back translation")
    parser.add_argument("--grammer_correction", default=False, type=bool, help="use grammer correction")

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
    wandb_run = wandb.init(entity=args.wandb_user, project=args.wandb_project)

    return wandb_run


if __name__ == "__main__":
    """
    supported model list
    - microsoft/deberta-v3-large

    DISCLAIMER:
    - currently only support single GPU training
    - arguments related to optimizer, scheduler, and loss function
      are not supported and directly assigned at the beginning of the code
    """
    # configuration
    seed_everything(42)
    args = get_config()

    # wandb setting
    wandb_run = wandb_setting(args)
    wandb_run.name = f'debertav3_fold_lr{args.lr}_{args.wandb_comment}'

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

    # optimizing with 1 data doesn't work good here so opted out
    # make trainloader batch_size to 1
    # args.grad_acc_steps = args.batch_size
    # args.max_grad_norm = 35 * args.batch_size

    # data
    all_texts, token_weights, data, csv, train_ids, val_ids, train_text_ids, val_text_ids = get_data_files(args)
    train_dataloader, val_dataloader = get_dataloader(args, train_ids, val_ids, data, csv, all_texts, train_text_ids, val_text_ids, class_names, token_weights)

    # loss
    # args.class_weight = torch.Tensor(token_weights).to(args.device).half()
    args.criterion_list = ["custom_ce", "custom_rce"]
    args.criterion_ratio = [args.ce_weight, args.rce_weight]
    # args.criterion_list = ["custom_ce"]
    # args.criterion_ratio = [1.]
    # args.criterion_list = ["dice"]
    # args.criterion_ratio = [1.]
    criterion = get_criterion(args)            

    # model
    # args.model = 'microsoft/deberta-v3-large'
    args.model = 'microsoft/deberta-v3-large-ducky'
    model = get_model(args)

    # optimizer
    args.optimizer = "adamw"
    # args.optimizer = "adafactor"
    # args.optimizer = "sam"
    # args.optimizer = "adamp"
    optimizer = get_optimizer(args, model)

    # scheduler
    # cosine - (one cycle learning) the learning rate will be decayed by a factor of 0.5 every 1 epochs
    args.steps_per_epoch = len(train_dataloader)
    args.scheduler = "plateau"
    # args.scheduler = "custom_warmup"
    # args.scheduler = "cosine_annealing"
    # args.scheduler = 'cosine_annealing_warmup_restart'
    scheduler = get_scheduler(args, optimizer)

    # configuration log
    print(args)

    # train
    trainer = Trainer(args, model, train_dataloader, val_dataloader, scheduler, optimizer, criterion, class_names)
    best_f1_bug, best_f1_clean, best_f1_wonho = trainer.train()

    print(f'[ Bug ] best f1 - {best_f1_bug}')
    print(f'[ Clean ] best f1 - {best_f1_clean}')
    print(f'[ Wonho ] best f1 - {best_f1_wonho}')
    