import wandb
run = wandb.init(entity=, project=)

import sys
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
max_grad_norm = 25 * batch_size
use_mixup = False
mixup_beta = 1.
start_eval_at = 3000
lr = 32e-6
min_lr = 32e-6
dataset_version = 1
warmup_steps = 500
d1, d2, d3 = (0,0,0)
rce_weight = .1
max_grad_norm = 1.
rce_weight = .1

ce_weight = 1 - rce_weight
decay_bias = False
eval_interval = 200
    
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpun
import torch as t
from transformers import RobertaTokenizer, DebertaModel
import h5py
import random
import numpy as np
import dill as pickle
from random import shuffle
from tqdm import tqdm
from glob import glob
import pandas as pd
import wandb
import re

with open('token_counts.pickle', 'rb') as f:
    groupped_token_counts, ungroupped_token_counts = pickle.load(f)
    
if use_groupped_weights:
    counts = groupped_token_counts
else:
    counts = ungroupped_token_counts

token_weights = (counts.mean() / counts) ** weights_pow

run.name = f'fold{val_fold}_minlr{min_lr}_maxlr{lr}_wd{wd}_warmup{warmup_steps}_gradnorm{max_grad_norm}_biasdecay{decay_bias}_ls{label_smoothing}_wp{weights_pow}_data{dataset_version}_mixup{use_mixup}_beta{mixup_beta}_d1{d1}_d2{d2}_d3{d3}_rce{rce_weight}'
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
        self.data = h5py.File('debertav1_data.h5py')
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ix):
        x = self.ids[ix]
        tokens = self.data['tokens'][x]
        attention_mask = self.data['attention_masks'][x]
        num_tokens = self.data['num_tokens'][x, 0]
        if global_attn > 0:
            attention_mask[0] = 2
            attention_mask[num_tokens - 1] = 2
            if global_attn > 1:
                attention_mask[tokens==4] = 2
                attention_mask[tokens==116] = 2
                attention_mask[tokens==328] = 2
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
        self.data = h5py.File('debertav1_data.h5py')
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
        if global_attn:
            attention_mask[tokens==4] = 2
            attention_mask[0] = 2
            attention_mask[num_tokens - 1] = 2

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
            if self.space_regex.match(text[char_ix]) is not None:# in (' ', '\n', '\xa0', '\x85'):
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
        max_len = 1600
        first_batch = False
    else:
        max_len = min(1600, (max(x[-1] for x in ins) + 7) // 8 * 8)
    return tuple(t.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) 
                 for x in range(len(ins[0]) - 1))

def val_collate_fn(ins):
    max_len = (max(x[-1] for x in ins) + 7) // 8 * 8
    return tuple(t.from_numpy(np.concatenate([ins[z][x][None, :max_len] for z in range(len(ins))])) 
                 for x in range(len(ins[0]) - 3)) \
                 + ([x[-3] for x in ins], [x[-2] for x in ins], np.array([x[-1] for x in ins]),)

def extract_entities(ps, n):
    cat_ps = ps.argmax(-1).cpu().numpy()
    all_entities = {}
    current_cat = None
    current_start = None
    for ix in range(1, n - 1):
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
    
    for cat_ix, min_len in zip(range(1, 8), (2, 2, 5, 2, 4, 3, 2)):
        if cat_ix in all_entities:
            all_entities[cat_ix] = [x for x in all_entities[cat_ix] if x[1] - x[0] + 1 >= min_len]
            
    return all_entities


def process_sample(raw_ps, raw_gts, index_map, bounds, gt_spans, num_tokens, match_stats, min_len=0):
    
    #bounds[num_tokens - 2, 1] = min(len(index_map) - 1, bounds[num_tokens - 2, 1])
    predicted_spans = {x: [map_span_to_word_indices(span, index_map, bounds) for span in y] 
                       for x, y in extract_entities(raw_ps, num_tokens).items()}
    predicted_spans = {x: [z for z in y if z[1] - z[0] >= min_len] for x, y in predicted_spans.items()}
    
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
    def __init__(self):
        super().__init__()
        self.feats = DebertaModel.from_pretrained('microsoft/deberta-xlarge')
        self.feats.pooler = None
        if grad_checkpt:
            self.feats.gradient_checkpointing_enable()
        self.feats.train();
        if extra_dense:
            self.class_projector = t.nn.Sequential(
                t.nn.LayerNorm(1024),
                t.nn.Linear(1024, 256),
                t.nn.GELU(),
                t.nn.Linear(256, 15)
            )
        else:
            self.class_projector = t.nn.Sequential(
                t.nn.LayerNorm(1024),
                t.nn.Linear(1024, 15)
            )
    def forward(self, embeddings, mask):
        return t.log_softmax(self.class_projector(self.feats(inputs_embeds=embeddings, attention_mask=mask, return_dict=False)[0]), -1)
    
train_bs = batch_size//grad_acc_steps
if use_mixup:
    train_bs *= 2
train_dataset = t.utils.data.DataLoader(TrainDataset(all_train_ids), collate_fn=train_collate_fn, 
                                        batch_size=train_bs,
                                        num_workers=8)
val_dataset = t.utils.data.DataLoader(ValDataset(val_ids), collate_fn=val_collate_fn, batch_size=4, num_workers=8,
                                      persistent_workers=True)

model = TvmLongformer().cuda()
# the code below does not disable dropout
for m in model.modules():
    if isinstance(m, t.nn.Dropout):
        m.p = 0
for l in model.feats.encoder.layer:
    l.attention.self.dropout.p = d1
    l.attention.output.dropout.p = d2
    l.output.dropout.p = d3
        
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
    val_matches = t.zeros(16)
    val_labels = t.zeros(16)
    match_stats = {ix:  {'fp': 0, 'fn': 0, 'tp': 0} for ix in range(1, 8)}
    f1s = np.zeros(8)
    rec = np.zeros(7)
    prec = np.zeros(7)
    for tokens, mask, labels, labels_mask, bounds, gt_dicts, index_map, num_tokens in val_dataset:
        with t.no_grad():
            tokens, mask, label, label_mask = (x.cuda() for x in (tokens, mask, labels, labels_mask))
            with t.cuda.amp.autocast():
                outs = model(model.feats.embeddings.word_embeddings(tokens), mask)
            ls.append(-(((outs.float() * label).sum(-1) * label_mask).sum(-1) / label_mask.sum(-1)).mean())
            match_updates = calc_acc(outs, label, label_mask)
            val_matches += match_updates[0]
            val_labels += match_updates[1]
            for sample_ix, num in enumerate(num_tokens):
                match_stats = process_sample(outs[sample_ix], labels[sample_ix], 
                                                   index_map[sample_ix], bounds[sample_ix],
                                                   gt_dicts[sample_ix],
                                                   num, match_stats, min_len=min_len)
    for ix in range(1, 8):
        f1s[ix] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + .5 * (match_stats[ix]['fp'] + match_stats[ix]['fn']))
        rec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fn'])
        prec[ix - 1] = match_stats[ix]['tp'] / (1e-7 + match_stats[ix]['tp'] + match_stats[ix]['fp'])
    f1s[0] = np.mean(f1s[1:])

    model.train()
    val_accs = (val_matches / val_labels).cpu().numpy()
    val_labels = val_labels.cpu().numpy()
    return t.stack(ls).mean().item(), val_accs, val_labels, f1s, rec, prec

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

def make_match_dict(ce, accs, labels, prefix, extras=None):
    log_dict = {}
    log_dict.update({f'{prefix}_ce': ce})
    log_dict.update({f'{prefix}_ACC#' + label_names[(x + 1)  // 2] + ('_B' if x % 2 == 1 else '_I'): accs[x]
                    for x in range(1, 15)})
    log_dict.update({f'{prefix}_ACC#' + 'None': accs[0], f'{prefix}_ACC#' + 'ALL': accs[-1]})
    log_dict.update({f'{prefix}_A_' + 'B': accs[1:-1:2].mean(), f'{prefix}_A_' + 'I': accs[:-1:2].mean(),
                    f'{prefix}_A_' + 'MEAN': accs[:-1].mean()})
    if extras is not None:
        f1s, rec, prec = extras
        log_dict.update({f'{prefix}_F1_{label_names[(ix + 1) // 2]}': f1s[ix]
                         for ix in range(1, 8)})
        log_dict.update({f'{prefix}_MacroF1': f1s[0]})
        log_dict.update({f'{prefix}_Rec_{label_names[(ix + 1) // 2]}': rec[ix - 1]
                         for ix in range(1, 8)})
        log_dict.update({f'{prefix}_Prec_{label_names[(ix + 1) // 2]}': prec[ix - 1]
                         for ix in range(1, 8)})
    return log_dict

lr_schedule = np.r_[np.linspace(0, lr, warmup_steps),
                    (np.cos(np.linspace(0, np.pi, len(train_dataset) - warmup_steps)) * .5 + .5) * (lr - min_lr) + min_lr]


ls = []
norms = []
train_matches = t.zeros(16)
train_labels = t.zeros(16)
log_dict = {}
best_val_score = 0

past_averaged_params = []
current_averaged_params = {x: y.clone().double() for x, y in model.state_dict().items()}
from tqdm import tqdm
step = 0
pbar = tqdm(total=(len(train_dataset) + grad_acc_steps - 1)//grad_acc_steps)

model_params = [p for n, p in model.named_parameters() if n != 'feats.embeddings.word_embeddings.weight']
    
    
scaler = t.cuda.amp.GradScaler(init_scale=65536.0/grad_acc_steps)
for step_, batch in enumerate(train_dataset):
    if step_ % grad_acc_steps == 0:
        for ix in range(len(opt.param_groups)):
            opt.param_groups[ix]['lr'] = lr_schedule[step]
        opt.zero_grad()
    tokens, mask, label, label_mask = (x.cuda() for x in batch)
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
        embeddings = model.feats.embeddings.word_embeddings(tokens)
    with t.cuda.amp.autocast():
        outs = model(embeddings, mask)
    outs = outs.float()
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
        if step % eval_interval == (eval_interval - 1):
            log_dict = {}
            norms = []
            t.autograd.set_grad_enabled(False)
            params_backup = {x: y.cpu().clone() for x, y in model.state_dict().items()}

            train_accs = (train_matches / train_labels).cpu().numpy()
            train_labels = train_labels.cpu().numpy()
            log_dict.update(make_match_dict(t.stack(ls).mean().item(), train_accs, train_labels, 'Train'))
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
                        val_ce, val_accs, val_labels, f1s, rec, prec = validate(model, val_dataset)
                        val_score = f1s[0]
                        if val_score >= best_val_score:
                            best_val_score = val_score
                            t.save(model.state_dict(), f'checkpoints_large_v1/{run.name}')
                        log_dict.update(make_match_dict(val_ce, val_accs, val_labels, f'ValSWA{params_ix*eval_interval}', (f1s, rec, prec)))

            
            model.cpu()
            model.load_state_dict(params_backup)
            model.cuda()
            current_averaged_params = {x: y.clone() for x, y in model.state_dict().items()}
    
            t.autograd.set_grad_enabled(True)
            wandb.log(log_dict)        
        step += 1
        pbar.update(1)

