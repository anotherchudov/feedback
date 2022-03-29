import wandb
run = wandb.init(entity=, project=)
import sys
sys.path.insert(0, PATH_TO_TRANSFORMERS_WITH_FAST_TOKENIZER_FOR_DEBERTA_V2)

gpun = sys.argv[1]
val_fold = sys.argv[2]

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
use_mixup = False
mixup_beta = 1.
start_eval_at = 3000
lr = 24e-6
min_lr = 24e-6
warmup_steps = 500
d1, d2, d3 = (0,0,0)
rce_weight = .1
max_grad_norm = 1.
rce_weight = .1
extend_pos_embs = False
if gpun == '0,1':
    val_fold = 0
    lr = 16e-6
    min_lr = lr / 2
    label_smoothing = .2
if gpun == '2,3':
    val_fold = 1
    lr = 16e-6
    min_lr = lr / 2
    label_smoothing = .2

ce_weight = 1 - rce_weight
decay_bias = False
eval_interval = 200
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpun
import torch as t
from transformers import DebertaV2Model
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import h5py
import random
import numpy as np
import dill as pickle
from random import shuffle
from tqdm import tqdm
from glob import glob
import pandas as pd
import re

with open('token_counts.pickle', 'rb') as f:
    groupped_token_counts, ungroupped_token_counts = pickle.load(f)
    
if use_groupped_weights:
    counts = groupped_token_counts
else:
    counts = ungroupped_token_counts

token_weights = (counts.mean() / counts) ** weights_pow


run.name = f'fold{val_fold}_minlr{min_lr}_maxlr{lr}_wd{wd}_warmup{warmup_steps}_gradnorm{max_grad_norm}_ls{label_smoothing}_wp{weights_pow}_mixup{use_mixup}_beta{mixup_beta}_d1{d1}_d2{d2}_d3{d3}_rce{rce_weight}_extpos{extend_pos_embs}'

gpu1 = 'cuda:0'
gpu2 = 'cuda:1'

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

all_texts = {}
for f in glob('train/*.txt'):
    with open(f) as x:
        all_texts[f.split('/')[-1].split('.')[0]] = x.read()

seed_everything(seed)

class TrainDataset(t.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.data = h5py.File(f'debertav2_data.h5py')
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ix):
        x = self.ids[ix]
        tokens = self.data['tokens'][x]
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
        self.data = h5py.File(f'debertav2_data.h5py')
        self.csv = pd.read_csv('train.csv')
        self.space_regex = re.compile('[\s\n]')
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
    

with open('id_to_ix_map.pickle', 'rb') as f:
    id_to_ix_map = {x.split('/')[-1].split('.')[0]: y for x, y in pickle.load(f).items()}
with open('data_splits.pickle', 'rb') as f:
    data_splits = pickle.load(f)

train_ids = [id_to_ix_map[x] for fold in range(5) if fold != val_fold for x in data_splits[seed][250]['normed'][fold]]
val_files = data_splits[seed][250]['normed'][val_fold]
val_ids = [id_to_ix_map[x] for x in val_files]

if use_mixup:
    epochs *= 2

all_train_ids = []
for epoch in range(epochs):
    epoch_train_ids = [x for x in train_ids]
    for _ in range(3):
        shuffle(epoch_train_ids)
    all_train_ids.extend(epoch_train_ids)
    
class TvmLongformer(t.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.feats = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')
        self.feats.pooler = None
        if grad_checkpt:
            self.gradient_checkpointing = True
        self.feats.train();
        if extra_dense:
            self.class_projector = t.nn.Sequential(
                t.nn.LayerNorm(1536),
                t.nn.Linear(1536, 384),
                t.nn.GELU(),
                t.nn.Linear(384, 15)
            )
        else:
            # self.convs = t.nn.ModuleList([
            #     t.nn.
            # ])
            self.class_projector = t.nn.Sequential(
                t.nn.LayerNorm(1536),
                t.nn.Linear(1536, 15)
            )
            
        if extend_pos_embs:
            self.replace_rel_embedding()
            
    def replace_rel_embedding(self):
        """tailored for 2048 seq length pos bucket, which the size is 384"""
        # create a new embedding layer
        new_rel_embeddings = t.nn.Embedding(768, 1536)

        # overwrite the pretrained weights to new relation embedding
        with t.no_grad():
            new_rel_embeddings.weight[:512, :] = self.feats.encoder.rel_embeddings.weight

        # overwrite
        self.feats.encoder.rel_embeddings = new_rel_embeddings
            
    # def extend_pos_embs(self):
    #     new_rel_embeddings = t.nn.Embedding(384 * 2, 1536)
    #     with t.no_grad():
    #         new_rel_embeddings.weight[128:128+512, :] = self.feats.encoder.rel_embeddings.weight
    #     self.feats.encoder.rel_embeddings = new_rel_embeddings
    #     attn_class = type(self.feats.encoder.layer[0].attention.self)
    #     for m in self.feats.encoder.modules():
    #         if isinstance(m, attn_class):
    #             m.pos_ebd_size = 384
            
    def to_gpus(self):
        self.to(gpu1);
        for x in self.feats.encoder.layer[len(self.feats.encoder.layer) // 2 - 2:]:
            x.to(gpu2)
        self.class_projector.to(gpu2)
        
            
    def forward(self, embeddings, attention_mask):
        hidden_states = self.feats.embeddings(
            mask=attention_mask,
            inputs_embeds=embeddings,
        )
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.feats.encoder.get_attention_mask(attention_mask)
        relative_pos = self.feats.encoder.get_rel_pos(hidden_states)
        
        rel_embeddings = self.feats.encoder.get_rel_embedding()
        next_kv = hidden_states
        
        for i, layer_module in enumerate(self.feats.encoder.layer):
            if i == len(self.feats.encoder.layer) // 2 - 2:  
                next_kv, attention_mask, relative_pos, rel_embeddings = (x.to(gpu2) for x in (next_kv, attention_mask, relative_pos, rel_embeddings))

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, False)

                    return custom_forward

                output_states = grad_checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    None,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=None,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=False,
                )

            if i == 0 and self.feats.encoder.conv is not None:
                output_states = self.feats.encoder.conv(hidden_states, output_states, input_mask)
                
            next_kv = output_states

        return t.log_softmax(self.class_projector(next_kv), -1)
    
train_bs = batch_size//grad_acc_steps
if use_mixup:
    train_bs *= 2
train_dataset = t.utils.data.DataLoader(TrainDataset(all_train_ids), collate_fn=train_collate_fn, 
                                        batch_size=train_bs,
                                        num_workers=8)
val_dataset = t.utils.data.DataLoader(ValDataset(val_ids), collate_fn=val_collate_fn, batch_size=1, num_workers=8,
                                      persistent_workers=True)

model = TvmLongformer()
model.to_gpus();
dropout_class = type(model.feats.embeddings.dropout)
for m in model.modules():
    if isinstance(m, dropout_class):
        m.drop_prob = 0
for l in model.feats.encoder.layer:
    l.attention.self.dropout.drop_prob = d1
    l.attention.output.dropout.drop_prob = d2
    l.output.dropout.drop_prob = d3
        
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
            tokens, mask, label, label_mask = (x.to(gpu2) for x in (tokens, mask, labels, labels_mask))
            num_tokens = num_tokens[0]
            with t.cuda.amp.autocast():
                outs = model(model.feats.embeddings.word_embeddings(tokens), mask)
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

lr_schedule = np.r_[np.linspace(0, lr, warmup_steps),
                    (np.cos(np.linspace(0, np.pi, len(train_dataset) - warmup_steps)) * .5 + .5) * (lr - min_lr) + min_lr]


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

gpu1_params = [p for p in model.parameters() if p.device.index == 0]
gpu2_params = [p for p in model.parameters() if p.device.index == 1]    
    
scaler = t.cuda.amp.GradScaler(init_scale=65536.0/grad_acc_steps)
for step_, batch in enumerate(train_dataset):
    if step_ % grad_acc_steps == 0:
        for ix in range(len(opt.param_groups)):
            opt.param_groups[ix]['lr'] = lr_schedule[step_]
        opt.zero_grad()
    tokens, mask = (x.to(gpu1) for x in batch[:2])
    label, label_mask = (x.to(gpu2) for x in batch[2:])
    if use_mixup:
        rand = t.tensor(np.random.beta(mixup_beta, mixup_beta), dtype=t.float, device='cuda:0').expand(1,1,1)
        emb_a, emb_b = model.feats.embeddings.word_embeddings(tokens).chunk(2, 0)
        lbl_a, lbl_b = label.chunk(2, 0)
        lm_a, lm_b = label_mask.chunk(2, 0)
        mask = (mask.sum(0, keepdim=True) > 0).float()
        embeddings = emb_a * rand.expand(1,1,1) + emb_b * (1 - rand)
        label = lbl_a * rand + lbl_b * (1 - rand)
        label_mask = lm_a * rand + lm_b * (1 - rand)
    else:
        embeddings = model.feats.embeddings.word_embeddings(tokens.to(gpu1))
    with t.cuda.amp.autocast():
        outs = model(embeddings, mask.to(gpu1))
    outs = outs.float()
    ce = -(((outs * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
    if rce_weight == 0:
        l = ce
    else:
        rce = -(((t.exp(outs) * t.log(label)).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean()
        l = ce * ce_weight + rce * rce_weight
    scaler.scale(l).backward()
    ls.append(l.detach())
    if step_ % grad_acc_steps == grad_acc_steps - 1:
        if max_grad_norm is not None:
            scaler.unscale_(opt)
            norm = t.norm(t.cat([t.stack([t.norm(p.grad.detach()) for p in gpu1_params]), 
                                 t.stack([t.norm(p.grad.detach()) for p in gpu2_params]).to(gpu1)]))
            
            if t.isfinite(norm):
                grad_mult_gpu1 = (max_grad_norm / (norm + 1e-6)).clamp(max=1.)
                grad_mult_gpu2 = grad_mult_gpu1.to(gpu2)
                for p in gpu1_params:
                    p.grad.detach().mul_(grad_mult_gpu1)
                for p in gpu2_params:
                    p.grad.detach().mul_(grad_mult_gpu2)
        scaler.step(opt)
        scaler.update()
        with t.no_grad():
            current_averaged_params = {x: y.clone().double() + current_averaged_params[x] for x, y in model.state_dict().items()}
        if step % eval_interval == (eval_interval - 1):
            log_dict = {}
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
                        model.to(gpu2)
                        all_f1s = validate(model, val_dataset)
                        
                        for val_fn_ix, val_fn_name in enumerate(('bugged', 'clean', 'extra')):
                            val_score = all_f1s[val_fn_ix]
                            if val_score >= best_val_scores[val_fn_ix]:
                                best_val_scores[val_fn_ix] = val_score
                                t.save(model.state_dict(), f'checkpoints_xlarge_v2/{val_fn_name}_{run.name}')
                            log_dict.update(make_match_dict(None, f'{val_fn_name}_SWA{params_ix*eval_interval}', [all_f1s[val_fn_ix]]))

            model.cpu()
            model.load_state_dict(params_backup)
            model.to_gpus()
            current_averaged_params = {x: y.clone() for x, y in model.state_dict().items()}
    
            t.autograd.set_grad_enabled(True)
            wandb.log(log_dict)        
        step += 1
        

