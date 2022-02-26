import re
import torch
import numpy as np

def split_predstring(predstring):
    vals = predstring.split()
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
        self.space_regex = re.compile('[\s\n]')
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
        
        # ???
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
        
        return tokens, attention_mask, cbio_labels, class_weight, token_bounds, gt_dict, index_map, num_tokens