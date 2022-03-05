

import os
import os.path as osp
from glob import glob

import pickle
import h5py

import numpy as np
import pandas as pd

import torch


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


def extract_entities_wonho(ps, n):
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

def map_span_to_word_indices(span, index_map, bounds):
    return (index_map[bounds[span[0], 0]], index_map[bounds[span[1], 1] - 1])

def split_predstring(x):
    vals = x.split()
    return int(vals[0]), int(vals[-1])


def process_sample(raw_ps, raw_gts, index_map, bounds, gt_spans, num_tokens, match_stats, version='wonho'):
    if version == 'bug':
        extract_entities = extract_entities_bugged
    elif version == 'clean':
        extract_entities = extract_entities_clean
    elif version == 'wonho':
        extract_entities = extract_entities_wonho

    raw_ps = raw_ps.cpu().numpy()
    predicted_spans = {x: [map_span_to_word_indices(span, index_map, bounds) for span in y] 
                       for x, y in extract_entities(raw_ps, num_tokens).items()}
    
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

def init_match_dict(losses, accs, class_names):
    log_dict = {}
    log_dict.update({f'valid_loss': losses})
    log_dict.update({f'valid_ACC#' + class_names[(x + 1) // 2] + ('_B' if x % 2 == 1 else '_I'): accs[x]
                    for x in range(1, 15)})
    log_dict.update({f'valid_ACC#' + 'None': accs[0], f'valid_ACC#' + 'ALL': accs[-1]})
    log_dict.update({f'valid_A_' + 'B': accs[1:-1:2].mean(), f'valid_A_' + 'I': accs[:-1:2].mean(),
                    f'valid_A_' + 'MEAN': accs[:-1].mean()})
                
    return log_dict

def make_match_dict(log_dict, class_names, prefix, extras=None):
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