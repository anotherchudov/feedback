
import os.path as osp
import pickle
import h5py
from glob import glob
import torch
import numpy as np

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
            all_texts[f.split('/')[-1].split('.')[0]] = f.read()
    
    return all_texts

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
    if current_cat is not None:
        if current_cat not in all_entities:
            all_entities[current_cat] = []
        all_entities[current_cat].append((current_start, ix))
    return all_entities

def map_span_to_word_indices(span, index_map, bounds):
    return (index_map[bounds[span[0], 0]], index_map[bounds[span[1], 1] - 1])

def split_predstring(x):
    vals = x.split()
    return int(vals[0]), int(vals[-1])


def process_sample(raw_ps, raw_gts, index_map, bounds, gt_spans, num_tokens, match_stats, min_len=0):
    # process_sample(outs[sample_ix], labels[sample_ix], 
    #                                                index_map[sample_ix], bounds[sample_ix],
    #                                                gt_dicts[sample_ix],
    #                                                num, match_stats, min_len=min_len)
    # bounds[num_tokens - 2, 1] = min(len(index_map) - 1, bounds[num_tokens - 2, 1])
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

def make_match_dict(ce, accs, labels, label_names, prefix, extras=None):
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

def calc_acc(raw_ps, raw_labels, valid_mask):
    valid_mask = (valid_mask > 0).float()
    all_matches = torch.zeros(16)
    all_labels = torch.zeros(16)
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