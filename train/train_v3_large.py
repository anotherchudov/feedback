import sys
sys.path.insert(0, '../input/codes')
gpun = sys.argv[1]
val_fold = int(sys.argv[2])
seed = 0
min_len=0
wd = 1e-2
weights_pow = 0.1
use_groupped_weights = False
global_attn = 0
label_smoothing = 0.1
extra_dense = False
epochs = 9
batch_size = 8
grad_acc_steps = batch_size
grad_checkpt = True
data_prefix = ''
max_grad_norm = 1
start_eval_at = 3000
lr = 32e-6
min_lr = 8e-6
dataset_version = 2
warmup_steps = 500
rce_weight = 0.1

ce_weight = 1 - rce_weight

decay_bias = False
eval_interval = 200    

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpun
import torch as t
# t.use_deterministic_algorithms(True)
# from longformer.longformer import Longformer, LongformerConfig, RobertaModel
# from longformer.sliding_chunks import pad_to_window_size

from transformers import DebertaV2Model
import h5py
import ftfy
import random
import numpy as np
import dill as pickle
from random import shuffle
from tqdm import tqdm
from glob import glob
import pandas as pd
import wandb
import re
from torch.nn import functional as F

with open('../data/token_counts.pickle', 'rb') as f:
    groupped_token_counts, ungroupped_token_counts = pickle.load(f)
    
if use_groupped_weights:
    counts = groupped_token_counts
else:
    counts = ungroupped_token_counts

# 15개 토큰 weight
token_weights = (counts.mean() / counts) ** weights_pow

wandb.login()
run = wandb.init(entity='dlrgy22', project='feedback_deberta_large')
print(val_fold)
run.name = f'nomalv3_scheduler_1dcnn_unk100_0.8_fold{val_fold}_minlr{min_lr}_maxlr{lr}_wd{wd}_warmup{warmup_steps}_gradnorm{max_grad_norm}_biasdecay{decay_bias}_ls{label_smoothing}_wp{weights_pow}_data{dataset_version}_rce{rce_weight}'

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    
    
label_names = ['None', 'Lead', 'Position', 'Evidence', 'Claim',
               'Concluding Statement', 'Counterclaim', 'Rebuttal']

# def fix_text(text):
#     return ftfy.fix_text(text).translate(str.maketrans({'\xad': '-', '`': '\'',
#                                                           '´': '\'', '¨': '"',
#                                                         '\xa0': ' '}))

all_texts = {}
for f in glob('../input/feedback-prize-2021/train/*.txt'):
    with open(f) as x:
        all_texts[f.split('/')[-1].split('.')[0]] = x.read()

seed_everything(seed)

class TrainDataset(t.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.data = h5py.File(f'../data/deberta_spm_data_v{dataset_version}.h5py')

    def __len__(self):
        return len(self.ids)

    def mask_aug(self, tokens):
        with_out_padding = range(np.max(np.where(tokens > 0)))
        mask_idxs = random.sample(with_out_padding, len(with_out_padding) // 100)
        for mask_idx in mask_idxs:
            # mask token 128000
            tokens[mask_idx] = 3
        
        return tokens

    def __getitem__(self, ix):
        x = self.ids[ix]
        tokens = self.data['tokens'][x]
        p = random.random()
        if p < 0.8:
            tokens = self.mask_aug(tokens)
        
        attention_mask = self.data['attention_masks'][x]
        num_tokens = self.data['num_tokens'][x, 0]

        cbio_labels = self.data[f'{data_prefix}cbio_labels'][x]
        cbio_labels *= (1 - label_smoothing)
        cbio_labels += label_smoothing / 15
        label_mask = np.zeros_like(attention_mask)
        argmax_labels = cbio_labels.argmax(-1)
        for ix in range(1, 15):
            label_mask[argmax_labels==ix] = token_weights[ix]
        zero_label_mask = argmax_labels==0
        zero_label_mask[num_tokens - 1:] = False
        label_mask[zero_label_mask] = token_weights[0]
        label_mask[0] = 0
        return tokens, attention_mask, cbio_labels, label_mask, num_tokens
    
    
class ValDataset(t.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.data = h5py.File(f'../data/deberta_spm_data_v{dataset_version}.h5py')
        self.csv = pd.read_csv('../input/feedback-prize-2021/train.csv')
        self.space_regex = re.compile('[\s\n]')
        self.val_files = val_files
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ix):
        x = self.ids[ix]
        text = all_texts[val_files[ix]]
        gt_dict = {}
        sample_df = self.csv.loc[self.csv.id==val_files[ix]]
        for cat_ix in range(1, 8):
            cat_name = label_names[cat_ix]
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
            label_mask[argmax_labels==cat_ix] = token_weights[cat_ix]
        zero_label_mask = argmax_labels==0
        zero_label_mask[num_tokens - 1:] = False
        label_mask[zero_label_mask] = token_weights[0]
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

    
first_batch = True
def train_collate_fn(ins):
    global first_batch
    if first_batch:
        max_len = 2048
        first_batch = False
    else:
        max_len = (max(x[-1] for x in ins) + 7) // 8 * 8

    return tuple(t.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) 
                 for x in range(len(ins[0]) - 1))

def val_collate_fn(ins):
    max_len = (max(x[-1] for x in ins) + 7) // 8 * 8
    return tuple(t.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) 
                 for x in range(len(ins[0]) - 3)) \
                 + ([x[-3] for x in ins], [x[-2] for x in ins], np.array([x[-1] for x in ins]),)

def extract_entities_bugged(ps, n):
    cat_ps = ps.argmax(-1)
    all_entities = {}
    current_cat = None
    current_start = None
    for ix in range(n):
        if cat_ps[ix] % 2 == 1:
            if current_cat is not None:
                if current_cat not in all_entities:
                    all_entities[current_cat] = []
                all_entities[current_cat].append((current_start, ix - 1))
            current_cat = (cat_ps[ix] + 1) // 2
            current_start = ix        
    if current_cat is not None:
        if current_cat not in all_entities:
            all_entities[current_cat] = []
        all_entities[current_cat].append((current_start, ix))
            
    return all_entities

def extract_entities_clean(ps, n):
    cat_ps = ps.argmax(-1)
    all_entities = {}
    current_cat = None
    current_start = None
    for ix in range(n):
        if cat_ps[ix] % 2 == 1:
            if current_cat is not None:
                if current_cat not in all_entities:
                    all_entities[current_cat] = []
                all_entities[current_cat].append((current_start, ix - 1))
            current_cat = (cat_ps[ix] + 1) // 2
            current_start = ix        
        elif cat_ps[ix] == 0:
            if current_cat is not None:
                if current_cat not in all_entities:
                    all_entities[current_cat] = []
                all_entities[current_cat].append((current_start, ix - 1))
            current_cat = None
        elif current_cat is not None and cat_ps[ix] != current_cat * 2:
            if current_cat not in all_entities:
                all_entities[current_cat] = []
            all_entities[current_cat].append((current_start, ix - 1))
            current_cat = None
    if current_cat is not None:
        if current_cat not in all_entities:
            all_entities[current_cat] = []
        all_entities[current_cat].append((current_start, ix))
            
    return all_entities

def extract_entities_with_exras(ps, n):
    START_WITH_I = True
    LOOK_AHEAD = True
    max_ps = ps.max(-1)
    
    ps = ps.argsort(-1)[...,::-1]
    # argmax
    cat_ps = ps[:, 0]
    # argmax2
    cat_ps2 = ps[:, 1]
    
    all_entities = {}
    new_entity = True
    current_cat = current_start = current_end = None
    
    # except for special tokens
    for ix in range(n):

        # logic on new entity
        if new_entity:
            # Background - ignore
            if cat_ps[ix] == 0:
                pass

            # B-LABEL(1,3,5,7,...) - start entity
            elif cat_ps[ix] % 2 == 1:
                current_cat = (cat_ps[ix] + 1) // 2
                current_start = current_end = ix
                new_entity = False
                
                if current_cat in [6, 7]:
                    LOOK_AHEAD = False
                else:
                    LOOK_AHEAD = True

            # I-LABEL(2,4,6,8,...) - conditional start
            elif cat_ps[ix] % 2 == 0:
                if START_WITH_I:
                    # Condition: I-LABEL in argmax with B-LABEL in argmax2
                    if cat_ps[ix] == (cat_ps2[ix]+1):
                        current_cat = cat_ps[ix] // 2
                        current_start = current_end = ix
                        new_entity = False
                        
                        if current_cat in [6, 7]:
                            LOOK_AHEAD = False
                        else:
                            LOOK_AHEAD = True
        
        # logic on ongoing entity
        else:
            # Background - save current entity and init current
            if cat_ps[ix] == 0:
                if LOOK_AHEAD:
                    if ix < n - 1 and (cat_ps[ix+1] == current_cat*2) and (cat_ps2[ix] == current_cat*2):
                        current_end = ix
                    else:
                        # update current
                        if current_cat not in all_entities:
                            all_entities[current_cat] = []
                        all_entities[current_cat].append((current_start, current_end))

                        # init current for new start
                        new_entity = True
                        current_cat = current_start = current_end = None
                
                else:
                    # update current
                    if current_cat not in all_entities:
                        all_entities[current_cat] = []
                    all_entities[current_cat].append((current_start, current_end))

                    # init current for new start
                    new_entity = True
                    current_cat = current_start = current_end = None

            # B-LABEL(1,3,5,7,...) - save current entity and start new
            elif cat_ps[ix] % 2 == 1:
                if cat_ps[ix] == (current_cat*2-1):
                    # update current
                    if current_cat not in all_entities:
                        all_entities[current_cat] = []
                    all_entities[current_cat].append((current_start, current_end))

                    # start new current
                    current_cat = (cat_ps[ix] + 1) // 2
                    current_start = current_end = ix
                    new_entity = False
                    
                    if current_cat in [6, 7]:
                        LOOK_AHEAD = False
                    else:
                        LOOK_AHEAD = True
                
                else:
                    if LOOK_AHEAD:
                        if ix < n - 1 and (cat_ps[ix+1] == current_cat*2) and (cat_ps2[ix] == current_cat*2):
                            current_end = ix
                        else:
                            # update current
                            if current_cat not in all_entities:
                                all_entities[current_cat] = []
                            all_entities[current_cat].append((current_start, current_end))

                            # start new current
                            current_cat = (cat_ps[ix] + 1) // 2
                            current_start = current_end = ix
                            new_entity = False
                        
                            if current_cat in [6, 7]:
                                LOOK_AHEAD = False
                            else:
                                LOOK_AHEAD = True
                            
                    else:
                        # update current
                        if current_cat not in all_entities:
                            all_entities[current_cat] = []
                        all_entities[current_cat].append((current_start, current_end))

                        # start new current
                        current_cat = (cat_ps[ix] + 1) // 2
                        current_start = current_end = ix
                        new_entity = False
                        
                        if current_cat in [6, 7]:
                            LOOK_AHEAD = False
                        else:
                            LOOK_AHEAD = True
                
            # I-LABEL(2,4,6,8,...) - conditional continue
            elif cat_ps[ix] % 2 == 0:
                # B-LABEL0, I-LABEL0 - continue
                if cat_ps[ix] == current_cat*2:
                    current_end = ix
                # B-LBAEL0, I-LABEL1 - conditional finish current entity
                else:
                    if LOOK_AHEAD:
                        if ix < n - 1 and (cat_ps[ix+1] == current_cat*2) and (cat_ps2[ix] == current_cat*2):
                            current_end = ix
                        else:
                            # update current
                            if current_cat not in all_entities:
                                all_entities[current_cat] = []
                            all_entities[current_cat].append((current_start, current_end))

                            # init current
                            new_entity = True
                            current_cat = current_start = current_end = None
                    else:
                        # update current
                        if current_cat not in all_entities:
                            all_entities[current_cat] = []
                        all_entities[current_cat].append((current_start, current_end))

                        # init current
                        new_entity = True
                        current_cat = current_start = current_end = None
    
    # last entity
    if not new_entity:
        # update current
        if current_cat not in all_entities:
            all_entities[current_cat] = []
        all_entities[current_cat].append((current_start, current_end))
    
    return all_entities


def process_sample(raw_ps, index_map, bounds, gt_spans, num_tokens, match_stats, extract_fn):
    
    predicted_spans = {x: [map_span_to_word_indices(span, index_map, bounds) for span in y] 
                       for x, y in extract_fn(raw_ps, num_tokens).items()}
    
    for cat_ix in range(1, 8):
        
        pspans = predicted_spans.get(cat_ix, [])
        gspans = gt_spans.get(cat_ix, [])
        if not len(pspans) or not len(gspans):
            match_stats[cat_ix]['fn'] += len(gspans)
            match_stats[cat_ix]['fp'] += len(pspans)
        else:
            all_overlaps = np.zeros((len(pspans), len(gspans)))
            for x1 in range(len(pspans)):
                pspan = pspans[x1]
                for x2 in range(len(gspans)):
                    gspan = gspans[x2]
                    start_ix = max(pspan[0], gspan[0])
                    end_ix = min(pspan[1], gspan[1])
                    overlap = max(0, end_ix - start_ix + 1)
                    if overlap > 0:
                        o1 = overlap / (pspan[1] - pspan[0] + 1)
                        o2 = overlap / (gspan[1] - gspan[0] + 1)
                        if min(o1, o2) >= .5:
                            all_overlaps[x1, x2] = max(o1, o2)
            unused_p_ix = set(range(len(pspans)))
            unused_g_ix = set(range(len(gspans)))
            col_size = len(pspans)
            row_size = len(gspans)
            for ix in np.argsort(all_overlaps.ravel())[::-1]:
                if not len(unused_g_ix) or not len(unused_p_ix) or all_overlaps.ravel()[ix] == 0:
                    match_stats[cat_ix]['fp'] += len(unused_p_ix)
                    match_stats[cat_ix]['fn'] += len(unused_g_ix)
                    break
                p_ix = ix // row_size
                g_ix = ix % row_size
                if p_ix not in unused_p_ix or g_ix not in unused_g_ix:
                    continue
                match_stats[cat_ix]['tp'] += 1
                unused_g_ix.remove(g_ix)
                unused_p_ix.remove(p_ix)
    return match_stats

def map_span_to_word_indices(span, index_map, bounds):
    return (index_map[bounds[span[0], 0]], index_map[bounds[span[1], 1] - 1])

def split_predstring(x):
    vals = x.split()
    return int(vals[0]), int(vals[-1])
    

with open('../data/id_to_ix_map.pickle', 'rb') as f:
    id_to_ix_map = {x.split('/')[-1].split('.')[0]: y for x, y in pickle.load(f).items()}
with open('../data/data_splits.pickle', 'rb') as f:
    data_splits = pickle.load(f)

train_ids = [id_to_ix_map[x] for fold in range(5) if fold != val_fold for x in data_splits[seed][250]['normed'][fold]]
val_files = data_splits[seed][250]['normed'][val_fold]
val_ids = [id_to_ix_map[x] for x in val_files]

all_train_ids = []
for epoch in range(epochs):
    epoch_train_ids = [x for x in train_ids]
    for _ in range(3):
        shuffle(epoch_train_ids)
    all_train_ids.extend(epoch_train_ids)
    
# if shuffle_data:
#     shuffle(all_train_ids)

class TvmLongformer(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        self.feats.pooler = None
        if grad_checkpt:
            self.feats.gradient_checkpointing_enable()
        self.feats.train()

        self.conv1d_layer1 = t.nn.Conv1d(1024, 1024, kernel_size=1)
        self.conv1d_layer3 = t.nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.conv1d_layer5 = t.nn.Conv1d(1024, 1024, kernel_size=5, padding=2)

        if extra_dense:
            self.class_projector = re.T.nn.Sequential(
                t.nn.LayerNorm(1024*3),
                t.nn.Linear(1024*3, 256),
                t.nn.GELU(),
                t.nn.Linear(256, 15)
            )
        else:
            self.class_projector = t.nn.Sequential(
                t.nn.LayerNorm(1024*3),
                t.nn.Linear(1024*3, 15)
            )
    def forward(self, tokens, mask):
        transformer_output = self.feats(tokens, mask, return_dict=False)[0]
        conv_input = transformer_output.transpose(1, 2) # batch, hidden, seq

        conv_output1 = F.relu(self.conv1d_layer1(conv_input)) 
        conv_output3 = F.relu(self.conv1d_layer3(conv_input)) 
        conv_output5 = F.relu(self.conv1d_layer5(conv_input)) 

        concat_output = t.cat((conv_output1, conv_output3, conv_output5), dim=1).transpose(1, 2)

        output = self.class_projector(concat_output)
        return t.log_softmax(output, -1)


train_dataset = t.utils.data.DataLoader(TrainDataset(all_train_ids), collate_fn=train_collate_fn, 
                                        batch_size=batch_size//grad_acc_steps,
                                        num_workers=8)
                                        
val_dataset = t.utils.data.DataLoader(ValDataset(val_ids), collate_fn=val_collate_fn, batch_size=1, num_workers=8,
                                      persistent_workers=True)

model = TvmLongformer().cuda()
for m in model.modules():
    if isinstance(m, t.nn.Dropout):
        m.p = 0
        
weights = []
biases = []
for n, p in model.named_parameters():
    if n.startswith('feats.embeddings') or 'LayerNorm' in n or n.endswith('bias'):
        biases.append(p)
    else:
        weights.append(p)

opt = t.optim.AdamW([{'params': weights, 'weight_decay': wd, 'lr': 0},
                     {'params': biases, 'weight_decay': 0 if not decay_bias else wd, 'lr': 0}])

def validate(model, dataset):
    ls = []
    model.eval();
    f1s = np.zeros(8)
    rec = np.zeros(7)
    prec = np.zeros(7)
    all_val_outs = np.zeros((len(dataset), 2048, 15), 'f4')
    all_val_nums = np.zeros((len(dataset),), 'i4')
    all_val_bounds = np.zeros((len(dataset), 2048, 2), 'i4')
    all_gt_dicts = []
    all_index_maps = []
    
    
    for sample_ix, (tokens, mask, labels, labels_mask, bounds, gt_dicts, index_map, num_tokens) in enumerate(dataset):
        with t.no_grad():
            tokens, mask, label, label_mask = (x.cuda() for x in (tokens, mask, labels, labels_mask))
            num_tokens = num_tokens[0]
            with t.cuda.amp.autocast():
                outs = model(tokens, mask)
            ls.append(-(((outs.float() * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean())
            all_val_nums[sample_ix] = num_tokens - 2
            all_val_bounds[sample_ix, : num_tokens - 2] = bounds[0, 1:num_tokens - 1]
            all_gt_dicts.append(gt_dicts[0])
            all_index_maps.append(index_map[0])
            all_val_outs[sample_ix, : num_tokens - 2] = outs.cpu().numpy()[0, 1:num_tokens - 1]
    
    all_f1s = []
    for extract_fn in (extract_entities_bugged, extract_entities_clean, extract_entities_with_exras):
        match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
        for sample_ix in range(len(all_val_nums)):
            match_stats = process_sample(all_val_outs[sample_ix],
                                                       all_index_maps[sample_ix], all_val_bounds[sample_ix],
                                                       all_gt_dicts[sample_ix],
                                                       all_val_nums[sample_ix], match_stats, extract_fn)
        for ix in range(1, 8):
            f1s[ix] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + .5 * (match_stats[ix]['fp'] + match_stats[ix]['fn']))
            rec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fn'])
            prec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fp'])
        f1s[0] = np.mean(f1s[1:])
        all_f1s.append(f1s[0])

    model.train()
    return all_f1s

def calc_acc(raw_ps, raw_labels, valid_mask):
    valid_mask = (valid_mask > 0).float()
    all_matches = t.zeros(16)
    all_labels = t.zeros(16)
    ps = raw_ps.argmax(-1)
    labels = raw_labels.argmax(-1) - (1 - valid_mask)
    matched = (ps == labels)
    for x in range(15):
        class_mask = labels==x
        class_labels = (class_mask).sum()
        class_matches = (matched[class_mask].sum())
        all_matches[x] = class_matches
        all_labels[x] = class_labels
    all_matches[-1] = matched.sum()
    all_labels[-1] = valid_mask.sum()
    return all_matches, all_labels

def make_match_dict(ce, prefix, f1s=None):
    log_dict = {}
    if ce is not None:
        log_dict.update({f'{prefix}_ce': ce})
    if f1s is not None:
        log_dict.update({f'{prefix}_MacroF1': f1s[0]})
    return log_dict

lr_schedule = np.r_[np.linspace(lr, lr, warmup_steps),
                    (np.cos(np.linspace(0, np.pi, len(train_dataset) - warmup_steps)) * .5 + .5) * (lr - min_lr)
                    + min_lr]

ls = []
norms = []
train_matches = t.zeros(16)
train_labels = t.zeros(16)
log_dict = {}
best_val_scores = [0] * 3

past_averaged_params = []
current_averaged_params = {x: y.clone().double() for x, y in model.state_dict().items()}
from tqdm import tqdm
step = 0
pbar = tqdm(total=(len(train_dataset) + grad_acc_steps - 1)//grad_acc_steps)
if global_attn == 0:
    model_params = [p for n, p in model.named_parameters() if not (n.endswith('_global.bias') or n.endswith('_global.weight'))]
else:
    model_params = [p for p in model.parameters()]
    
    
scaler = t.cuda.amp.GradScaler(init_scale=65536.0/grad_acc_steps)
for step_, batch in enumerate(train_dataset):
    if step_ % grad_acc_steps == 0:
        for ix in range(len(opt.param_groups)):
            # opt.param_groups[ix]['lr'] = lr_schedule[step]
            # change_ scheduler
            opt.param_groups[ix]['lr'] = lr_schedule[step_]
        opt.zero_grad()

    tokens, mask, label, label_mask = (x.cuda() for x in batch)
    with t.cuda.amp.autocast():
        outs = model(tokens, mask)
    ce = -(((outs * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
    if rce_weight == 0:
        l = ce
    else:
        rce = -(((t.exp(outs) * t.log_softmax(label, -1)).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
        l = ce * ce_weight + rce * rce_weight    
    scaler.scale(l).backward()
    ls.append(l.detach())
    
    if step_ % grad_acc_steps == grad_acc_steps - 1:
        if max_grad_norm is not None:
            scaler.unscale_(opt)
            norm = t.norm(t.stack([t.norm(p.grad.detach()) for p in model_params]))
            # norms.append(norm)
            if t.isfinite(norm):
                grad_mult = (max_grad_norm / (norm + 1e-6)).clamp(max=1.)
                for p in model_params:
                    p.grad.detach().mul_(grad_mult)
        scaler.step(opt)#.step()
        scaler.update()
        
        with t.no_grad():
            current_averaged_params = {x: y.clone().double() + current_averaged_params[x] for x, y in model.state_dict().items()}
            match_updates = calc_acc(outs, label, label_mask)
            train_matches += match_updates[0]
            train_labels += match_updates[1]

        # 여기    
        # print(step, eval_interval)
        if step % eval_interval == (eval_interval - 1):
            # norms = t.stack(norms)
            log_dict = {}
            # 'norm_mean': t.mean(norms).item(),
            #             'norm_max': t.max(norms).item(),
            #             'norm_.9': t.quantile(norms, .9).item(),
            #             'norm_.95': t.quantile(norms, .95).item(),}
            
            norms = []
            t.autograd.set_grad_enabled(False)
            params_backup = {x: y.cpu().clone() for x, y in model.state_dict().items()}

            train_accs = (train_matches / train_labels).cpu().numpy()
            train_labels = train_labels.cpu().numpy()
            log_dict.update(make_match_dict(t.stack(ls).mean().item(), 'Train'))
            ls = []
            train_matches = t.zeros(16)
            train_labels = t.zeros(16)

            if len(past_averaged_params) == 25:
                past_averaged_params = past_averaged_params[-24:]
            past_averaged_params.append({x: (y * (1 / eval_interval)).cpu().half() for x, y in current_averaged_params.items()})
            if step > start_eval_at:
                evaluation_params = None
                for params_ix in range(1, len(past_averaged_params) + 1):
                    if evaluation_params is None:
                        evaluation_params = {x: y.double() for x, y in past_averaged_params[-params_ix].items()}
                    else:
                        evaluation_params = {x: y.double() + evaluation_params[x] 
                                             for x, y in past_averaged_params[-params_ix].items()}
                    
                    if params_ix >= 10 and params_ix % 5 == 0:
                        model.cpu()
                        model.load_state_dict({x: (y / params_ix).float() for x, y in evaluation_params.items()})
                        model.cuda()
                        
                        all_f1s = validate(model, val_dataset)
                        for val_fn_ix, val_fn_name in enumerate(('bugged', 'clean', 'extra')):
                            val_score = all_f1s[val_fn_ix]

                            if val_score >= best_val_scores[val_fn_ix]:
                                best_val_scores[val_fn_ix] = val_score
                                t.save(model.state_dict(), f'checkpoints/{val_fn_name}_{run.name}')
                                print(f"save model {val_fn_name} {best_val_scores[val_fn_ix]} -> {val_score}")
                        
                            log_dict.update(make_match_dict(None, f'{val_fn_name}_SWA{params_ix*eval_interval}', [all_f1s[val_fn_ix]]))
                            
                print(best_val_scores)
            
            model.cpu()
            model.load_state_dict(params_backup)
            model.cuda()
            current_averaged_params = {x: y.clone() for x, y in model.state_dict().items()}
    
            t.autograd.set_grad_enabled(True)
            wandb.log(log_dict)   

        step += 1
        pbar.update(1)

