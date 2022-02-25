import re
import torch
import numpy as np

def split_predstring(x):
    vals = x.split()
    return int(vals[0]), int(vals[-1])

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data, label_smoothing, token_weights, data_prefix):
        self.ids = ids
        self.data = data
        self.label_smoothing = label_smoothing
        self.token_weights = token_weights
        self.data_prefix = data_prefix

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        x = self.ids[ix]
        tokens = self.data['tokens'][x]
        attention_mask = self.data['attention_masks'][x]
        num_tokens = self.data['num_tokens'][x, 0]
        cbio_labels = self.data[f'{self.data_prefix}cbio_labels'][x]
        cbio_labels *= (1 - self.label_smoothing)
        cbio_labels += self.label_smoothing / 15

        label_mask = np.zeros_like(attention_mask)
        argmax_labels = cbio_labels.argmax(-1)
        for ix in range(1, 15):
            label_mask[argmax_labels==ix] = self.token_weights[ix]
        zero_label_mask = argmax_labels==0
        zero_label_mask[num_tokens - 1:] = False
        label_mask[zero_label_mask] = self.token_weights[0]
        label_mask[0] = 0

        return tokens, attention_mask, cbio_labels, label_mask, num_tokens
    
class ValDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data, csv, all_texts, val_files, label_names, token_weights):
        self.ids = ids
        self.data = data
        self.csv = csv
        self.space_regex = re.compile('[\s\n]')
        self.all_texts = all_texts
        self.val_files = val_files
        self.label_names = label_names
        self.token_weights = token_weights

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ix):
        x = self.ids[ix]
        text = self.all_texts[self.val_files[ix]]
        gt_dict = {}
        sample_df = self.csv.loc[self.csv.id==self.val_files[ix]]
        for cat_ix in range(1, 8):
            cat_name = self.label_names[cat_ix]
            cat_entities = sample_df.loc[sample_df.discourse_type==cat_name]
            if len(cat_entities):
                gt_dict[cat_ix] = [(x[0], x[1]) for x in cat_entities.predictionstring.map(split_predstring)]
        
        tokens = self.data['tokens'][x]
        attention_mask = self.data['attention_masks'][x]
        num_tokens = self.data['num_tokens'][x, 0]
        token_bounds = self.data['token_offsets'][x]

        cbio_labels = self.data['cbio_labels'][x]
        
        label_mask = np.zeros_like(attention_mask)
        argmax_labels = cbio_labels.argmax(-1)
        for cat_ix in range(1, 15):
            label_mask[argmax_labels==cat_ix] = self.token_weights[cat_ix]
        zero_label_mask = argmax_labels==0
        zero_label_mask[num_tokens - 1:] = False
        label_mask[zero_label_mask] = self.token_weights[0]
        label_mask[0] = 0
        
        index_map = []
        current_word = 0
        blank = False
        for char_ix in range(text.index(text.strip()[0]), len(text)):
            if self.space_regex.match(text[char_ix]) is not None:
                blank = True
            elif blank:
                current_word += 1
                blank = False
            index_map.append(current_word)
        
        return tokens, attention_mask, cbio_labels, label_mask, token_bounds, gt_dict, index_map, num_tokens