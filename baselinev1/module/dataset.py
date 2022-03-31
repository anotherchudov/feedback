from random import shuffle
import re
import torch
import numpy as np
import pickle

from torch.utils.data import DataLoader

from module.utils import get_data_files
from .text import TextAugmenter

def split_predstring(predstring):
    vals = predstring.split()
    return int(vals[0]), int(vals[-1])

class OnlineTrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, text_ids, all_texts, df, token_weights, text_augmenter):
        self.args = args
        self.all_texts = all_texts
        self.text_ids = text_ids
        self.df = df

        self.token_weights = token_weights

        # you could add more text augmenter here and take turns using it
        """
        args.back_translation = True
        self.back_aug = TextAugmenter(args, text_ids, all_texts, df)
        args.back_translation = False
        self.no_aug = TextAugmenter(args, text_ids, all_texts, df)
        
        if epochs < 5:
            label = self.back_aug.get_label(text_id)  
        else:
            label = self.no_aug.get_label(text_id)  
        """
        self.text_augmenter = text_augmenter

        if self.args.distill:
            distill_path = 'result/online_no_wd_noise_filtering0.4_mean_teacher0.99/token_distillation_debertav3_fold0_online_no_wd_noise_filtering0.4_mean_teacher0.99.pkl'
            with open(distill_path, 'rb') as f:
                self.distill_info = pickle.load(f)

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        text_id = self.text_ids[idx]

        # load train data
        label = self.text_augmenter.get_label(text_id, cache=False)  
        tokens, token_labels, attention_mask, num_tokens = label

        # distillation
        if self.args.distill:
            distill_i = self.distill_info[0].index(text_id)
            distill_labels = self.distill_info[1][distill_i]

        # label smoothing
        token_labels *= (1 - self.args.label_smoothing)
        token_labels += self.args.label_smoothing / 15

        # TODO: use token label from EMA processed self-ensembled prediction

        # class weight per token
        class_weight = np.zeros_like(attention_mask)
        argmax_labels = token_labels.argmax(-1)

        for class_i in range(1, 15):
            class_weight[argmax_labels == class_i] = self.token_weights[class_i]

        class_none_index = argmax_labels == 0      # 0 is the text that is not entity
        class_none_index[num_tokens - 1:] = False  # special token & padding
        class_weight[class_none_index] = self.token_weights[0]
        class_weight[0] = 0

        if self.args.distill:
            return tokens, attention_mask, token_labels, distill_labels, class_weight, num_tokens
        else:
            return tokens, attention_mask, token_labels, class_weight, num_tokens


class OnlineValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, text_ids, all_texts, df, class_names, token_weights, text_augmenter):
        self.args = args
        self.all_texts = all_texts
        self.text_ids = text_ids
        self.df = df

        self.class_names = class_names
        self.token_weights = token_weights

        self.text_augmenter = text_augmenter

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        text_id = self.text_ids[idx]
        text = self.all_texts[text_id]

        # load valid data
        sample_df, label = self.text_augmenter.get_label(text_id, cache=False)  
        tokens, token_labels, attention_mask, token_offset, num_tokens = label

        # load ground truth prediction string for f1macro metric
        gt_dict = {}
        for class_i in range(1, 8):
            class_name = self.class_names[class_i]
            class_df = sample_df.query('discourse_type == @class_name')   
            if len(class_df):
                gt_dict[class_i] = [(x[0], x[1]) for x in class_df.predictionstring.map(split_predstring)]

        # class weight per token
        class_weight = np.zeros_like(attention_mask)
        argmax_labels = token_labels.argmax(-1)

        for class_i in range(1, 15):
            class_weight[argmax_labels == class_i] = self.token_weights[class_i]

        class_none_index = argmax_labels == 0
        class_none_index[num_tokens - 1:] = False
        class_weight[class_none_index] = self.token_weights[0]
        class_weight[0] = 0
        
        # calculate index map
        # I'm quarking on the kaggle QUARK!!!
        # 00001111111112223333444444455555555
        index_map = []
        current_word = 0
        blank = False
        for char_ix in range(text.index(text.strip()[0]), len(text)):
            if text[char_ix] in [' ', '\n']:
                blank = True
            elif blank:
                current_word += 1
                blank = False
            index_map.append(current_word)
        
        return tokens, attention_mask, token_labels, class_weight, token_offset, gt_dict, index_map, num_tokens



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data, label_smoothing, token_weights, data_prefix):
        self.ids = ids
        self.data = data
        self.label_smoothing = label_smoothing
        self.token_weights = token_weights
        self.data_prefix = data_prefix

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]

        # load train data
        tokens = self.data['tokens'][i]
        attention_mask = self.data['attention_masks'][i]
        num_tokens = self.data['num_tokens'][i, 0]

        # label smoothing
        cbio_labels = self.data[f'{self.data_prefix}cbio_labels'][i]
        cbio_labels *= (1 - self.label_smoothing)
        cbio_labels += self.label_smoothing / 15

        # class weight per token
        class_weight = np.zeros_like(attention_mask)
        argmax_labels = cbio_labels.argmax(-1)

        for class_i in range(1, 15):
            class_weight[argmax_labels == class_i] = self.token_weights[class_i]

        class_none_index = argmax_labels == 0      # 0 is the text that is not entity
        class_none_index[num_tokens - 1:] = False  # special token & padding
        class_weight[class_none_index] = self.token_weights[0]
        class_weight[0] = 0

        return tokens, attention_mask, cbio_labels, class_weight, num_tokens
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data, csv, all_texts, val_text_ids, class_names, token_weights):
        self.ids = ids
        self.data = data
        self.csv = csv
        self.all_texts = all_texts
        self.val_text_ids = val_text_ids
        self.class_names = class_names
        self.token_weights = token_weights

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]

        # load text data & text dataframe
        text_id = self.val_text_ids[idx]
        text = self.all_texts[text_id]
        sample_df = self.csv.query('id == @text_id')

        # load ground truth prediction string for f1macro metric
        gt_dict = {}
        for class_i in range(1, 8):
            class_name = self.class_names[class_i]
            class_df = sample_df.query('discourse_type == @class_name')   
            if len(class_df):
                gt_dict[class_i] = [(x[0], x[1]) for x in class_df.predictionstring.map(split_predstring)]
        
        # load valid data
        tokens = self.data['tokens'][i]
        attention_mask = self.data['attention_masks'][i]
        num_tokens = self.data['num_tokens'][i, 0]
        token_bounds = self.data['token_offsets'][i]
        cbio_labels = self.data['cbio_labels'][i]
        
        # class weight per token
        class_weight = np.zeros_like(attention_mask)
        argmax_labels = cbio_labels.argmax(-1)

        for class_i in range(1, 15):
            class_weight[argmax_labels == class_i] = self.token_weights[class_i]

        class_none_index = argmax_labels == 0
        class_none_index[num_tokens - 1:] = False
        class_weight[class_none_index] = self.token_weights[0]
        class_weight[0] = 0
        
        # calculate index map
        # I'm quarking on the kaggle QUARK!!!
        # 00001111111112223333444444455555555
        index_map = []
        current_word = 0
        blank = False
        for char_ix in range(text.index(text.strip()[0]), len(text)):
            if text[char_ix] in [' ', '\n']:
                blank = True
            elif blank:
                current_word += 1
                blank = False
            index_map.append(current_word)
        
        return tokens, attention_mask, cbio_labels, class_weight, token_bounds, gt_dict, index_map, num_tokens

first_batch = True
def train_collate_fn(ins):
    global first_batch
    if first_batch:
        max_len = 2048
        first_batch = False
    else:
        max_len = (max(x[-1] for x in ins) + 7) // 8 * 8
        
    return tuple(torch.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) for x in range(len(ins[0]) - 1))
    
def val_collate_fn(ins):
    max_len = (max(x[-1] for x in ins) + 7) // 8 * 8
    return tuple(torch.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) for x in range(len(ins[0]) - 3)) + ([x[-3] for x in ins], [x[-2] for x in ins], np.array([x[-1] for x in ins]),)


def get_augmenter(args):
    # data
    all_texts, _, _, csv, _, _, train_text_ids, val_text_ids = get_data_files(args)

    train_augmenter = TextAugmenter(args, train_text_ids, all_texts, csv)
    valid_augmenter = TextAugmenter(args, val_text_ids, all_texts, csv, valid=True)

    return train_augmenter, valid_augmenter


def get_dataloader(args, train_ids, val_ids, data, csv, all_texts, train_text_ids, val_text_ids, class_names, token_weights, train_augmenter=None, valid_augmenter=None):
    if args.online_dataset:
        train_dataset = OnlineTrainDataset(args, train_text_ids, all_texts, csv, token_weights, train_augmenter)
        val_dataset = OnlineValidDataset(args, val_text_ids, all_texts, csv, class_names, token_weights, valid_augmenter)
    else:
        train_dataset = TrainDataset(train_ids, data, args.label_smoothing, token_weights, args.data_prefix)
        val_dataset = ValDataset(val_ids, data, csv, all_texts, val_text_ids, class_names, token_weights)

    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=args.batch_size, num_workers=args.num_worker)

    return train_dataloader, val_dataloader