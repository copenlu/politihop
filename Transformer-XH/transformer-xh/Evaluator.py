# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from data import HotpotDataset, FEVERDataset, TransformerXHDataset

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from data import batcher_hotpot, batcher_fever, load_data
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from tqdm import tqdm
import logging
import torch.nn.functional as F
from utils import find_start_end_before_tokenized
import json


def evidence_prec(gold, pred):
    # assumes preds and gold are sets of sentence labels
    return len(pred.intersection(gold)) / len(pred)


def evidence_recall(gold, pred):
    # assumes preds and gold are sets of sentence labels
    return len(pred.intersection(gold)) / len(gold)


def evidence_f1(gold, pred):
    # assumes preds and gold are sets of sentence labels
    precision = evidence_prec(gold, pred)
    recall = evidence_recall(gold, pred)
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

'''
Evaluation for Hotpot QA task
'''

def evaluation_hotpot(model, eval_file, config, args):

    dataset = HotpotDataset(config["system"]['test_data'], config["model"], False, args.tokenizer)
    device = args.device
    dataloader = DataLoader(dataset=dataset,
                              batch_size=config['training']['test_batch_size'],
                              collate_fn=batcher_hotpot(device),
                              shuffle=False,
                              num_workers=0)
    logging.info("=================================================================================")
    total = 0
    pred_dict = dict()
    span_correct = 0
    label_correct = 0
    for batch in tqdm(dataloader):    
        
        logits, mrc_logits = model.network(batch, device)
        
        values, index = logits.topk(1)
        
        label = batch[1]
        g = batch[0]

        B_start = g.ndata['B_start'][index.item()]
                
        start_logits, end_logits = mrc_logits.split(1, dim=-1)
        ### find the span, where the end position is within 10 distance of start position

        start_values, indices = start_logits.squeeze(-1)[index.item(), B_start:].topk(1)
        start_index = indices[0]
        start = start_index + B_start
        ending = start + 10
        end_values, end_idx = end_logits.squeeze(-1)[index.item(), start:ending].topk(1)
        end = end_idx[0] + start


        start_pred = start - B_start
        start_pred = start_pred.item()
        end_pred = end - B_start
        end_pred = end_pred.item()
        total += 1

        if batch[2][index.item()] == start.item() and batch[3][index.item()] == end.item():
            span_correct += 1
        
        if label[index.item()].item() == 1:
            label_correct += 1

        pred_dict[batch[5][0]]={'node': index.item(), 'span': [start_pred, end_pred]}
    
    #### Generate prediction file for official eval
    graphs = dataset.data
    final_answer = dict()
    for graph in graphs:
        qid = graph['qid']
        pred = pred_dict[qid]

        for node in graph['node']:
            if node['node_id'] == pred['node']:
                ctx_str = ''.join(node['full_name'])
                context_tokens = node['context']
                span = pred['span']
                pred_str = context_tokens[span[0]: span[1] + 1]
                res = find_start_end_before_tokenized(ctx_str, [pred_str])
                if res[0] == (0,0):
                    tok_text = " ".join(pred_str)
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    tok_text = tok_text.strip()
                    final_answer[qid] = " ".join(tok_text.split())

                else:
                    final_answer[qid] = ctx_str[res[0][0]: res[0][1] + 1]
    final_pred = {'answer': final_answer, 'sp': dict()}


    ## the test file does not have correct labels, therefore we don't count the accuracy.
    accu = label_correct / total
    logging.info("********* Node accuracy ************{}".format(accu))
    accu = span_correct / total
    logging.info("********* Span accuracy ************{}".format(accu))
    return final_pred

    


'''
Evaluation for FEVER task
'''

def evaluation_fever(model, eval_file, config, args):
    
    dataset = FEVERDataset(eval_file, config["model"], False, args.tokenizer)
    device = args.device
    dataloader = DataLoader(dataset=dataset,
                              batch_size=config['training']['test_batch_size'],
                              collate_fn=batcher_fever(device),
                              shuffle=False,
                              num_workers=0)
    logging.info("=================================================================================")
    total = 0
    count = 0 
    preds, labels = [], []
    pred_dict = dict()
    
    for batch in tqdm(dataloader):  
          
        
        logits_score, logits_pred, attentions = model.network(batch, device)
        
        logits_score = F.softmax(logits_score)
        logits_pred =  F.softmax(logits_pred, dim=1)
        final_score = torch.mm(logits_score.unsqueeze(0), logits_pred).squeeze(0)

        values, index = final_score.topk(1)
        label = batch[2]
        
        if index[0].item() == label[0].item():
            count += 1
        total += 1
        
        preds.append(index[0].item())
        labels.append(label[0].item())

        pred_dict[batch[3][0]] = (index[0].item(), logits_score.tolist(), final_score.tolist())

    accu = count / total
    f1_mac = f1_score(labels, preds, average='macro')
    conf_matrix = confusion_matrix(labels, preds)

    logging.info("********* Label accuracy {}, f1 macro {} ************".format(accu, f1_mac))

    return accu, pred_dict, f1_mac




