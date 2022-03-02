

import os
import os.path as osp
from glob import glob

import pickle
import h5py

import numpy as np
import pandas as pd

import torch


def extract_entities(preds, n):
    cat_preds = preds.argmax(-1).cpu().numpy()
    all_entities = {}
    current_cat = None
    current_start = None

    for ix in range(1, n - 1):
        if cat_preds[ix] % 2 == 1:
            if current_cat is not None:
                if current_cat not in all_entities:
                    all_entities[current_cat] = []
                all_entities[current_cat].append((current_start, ix - 1))
            current_cat = (cat_preds[ix] + 1) // 2
            current_start = ix
        elif cat_preds[ix] == 0:
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

def make_match_dict(ce, accs, labels, class_names, prefix, extras=None):
    log_dict = {}
    log_dict.update({f'{prefix}_ce': ce})
    log_dict.update({f'{prefix}_ACC#' + class_names[(x + 1)  // 2] + ('_B' if x % 2 == 1 else '_I'): accs[x]
                    for x in range(1, 15)})
    log_dict.update({f'{prefix}_ACC#' + 'None': accs[0], f'{prefix}_ACC#' + 'ALL': accs[-1]})
    log_dict.update({f'{prefix}_A_' + 'B': accs[1:-1:2].mean(), f'{prefix}_A_' + 'I': accs[:-1:2].mean(),
                    f'{prefix}_A_' + 'MEAN': accs[:-1].mean()})

    if extras is not None:
        f1s, rec, prec = extras
        log_dict.update({f'{prefix}_F1_{class_names[(ix + 1) // 2]}': f1s[ix]
                         for ix in range(1, 8)})
        log_dict.update({f'{prefix}_MacroF1': f1s[0]})
        log_dict.update({f'{prefix}_Rec_{class_names[(ix + 1) // 2]}': rec[ix - 1]
                         for ix in range(1, 8)})
        log_dict.update({f'{prefix}_Prec_{class_names[(ix + 1) // 2]}': prec[ix - 1]
                         for ix in range(1, 8)})
    return log_dict

def calc_acc(raw_preds, raw_labels, valid_mask):
    valid_mask = (valid_mask > 0).float()
    all_matches = torch.zeros(16)
    all_labels = torch.zeros(16)

    preds = raw_preds.argmax(-1)
    labels = raw_labels.argmax(-1) - (1 - valid_mask)

    matched = (preds == labels)
    for class_i in range(15):
        class_mask = labels == class_i
        class_labels = (class_mask).sum()
        class_matches = (matched[class_mask].sum())
        all_matches[class_i] = class_matches
        all_labels[class_i] = class_labels

    all_matches[-1] = matched.sum()
    all_labels[-1] = valid_mask.sum()

    return all_matches, all_labels