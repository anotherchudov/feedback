
import os
import os.path as osp
from glob import glob

import pickle
import h5py

import numpy as np
import pandas as pd

import torch

def get_fold_data():
    #datasplits {seed :{n_neighbors :{normed or not_normed : [id]}}}
    with open('./data_file/data_splits.pickle', 'rb') as f:
        data_splits = pickle.load(f)
    return data_splits

def get_prepare_data():
    #h5 file make by notebook/prepare_data_for_Debertav3.ipynb
    data = h5py.File(f'./data_file/deberta_spm_data_v2.h5py')
    return data

def get_id_to_ix_map():
    with open('./data_file/id_to_ix_map.pickle', 'rb') as f:
        id_to_ix_map = {x.split('/')[-1].split('.')[0]: y for x, y in pickle.load(f).items()}
    return id_to_ix_map

def get_token_weights(use_groupped_weights, weights_pow):
    with open('./data_file/token_counts.pickle', 'rb') as f:
        groupped_token_counts, ungroupped_token_counts = pickle.load(f)
    if use_groupped_weights:
        counts = groupped_token_counts
    else:
        counts = ungroupped_token_counts
    token_weights = (counts.mean() / counts) ** weights_pow

    return token_weights

def get_all_texts(args):
    all_texts = {}
    # key : id value txt
    for text_file in glob(osp.join(args.dataset_path, 'train/*.txt')):
        with open(text_file) as f:
            all_texts[text_file.split('/')[-1].split('.')[0]] = f.read()
    
    return all_texts

def get_data_files(args):
    token_weights = get_token_weights(args.use_groupped_weights, args.weights_pow)
    data = get_prepare_data()
    csv = pd.read_csv(osp.join(args.dataset_path, 'train.csv'))
    all_texts = get_all_texts(args)
    id_to_ix_map = get_id_to_ix_map()
    data_splits = get_fold_data()

    # text_id example `16585724607E`
    train_text_ids = [text_id for fold in range(5) if fold != args.val_fold for text_id in data_splits[args.seed][250]['normed'][fold]]
    val_text_ids = data_splits[args.seed][250]['normed'][args.val_fold]

    train_ids = [id_to_ix_map[text_id] for text_id in train_text_ids]
    val_ids = [id_to_ix_map[text_id] for text_id in val_text_ids]

    return all_texts, token_weights, data, csv, train_ids, val_ids, train_text_ids, val_text_ids
