# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import argparse
import os
import json
import numpy as np

from model import Model_Hotpot, Model_FEVER
import data
import logging
import random
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch.nn.functional as F
from Trainer import train_hotpot, train_fever
from Evaluator import evaluation_hotpot, evaluation_fever




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)



def parse_args():
    parser = argparse.ArgumentParser("Transformer-XH")
    parser.add_argument("--config-file", "--cf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
    parser.add_argument('--out_model', default=None, help="Name of the trained model (will be saved with that name). Used for training only")
    parser.add_argument('--in_model', default=None, help="Path to the model to load for training/eval. Required for eval. Optional for training.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                    "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
    parser.add_argument('--epochs',
                    type=int,
                    default=2,
                    help="number of training epochs")
    parser.add_argument('--checkpoint',
                    type=int,
                    default=1000)
    parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--test',
                    default=False,
                    action='store_true',
                    help="Whether on test mode")
    parser.add_argument('--arch', default='trans_xh', help="Model architecture. Either trans_xh or bert")
    parser.add_argument('--dataset', default='politihop', 
                    help="Dataset to use in training/eval. One of: politihop, liar, fever, politihop_all, liar_all, adv")
    parser.add_argument('--hops', type=int, default=3, help="Number of hops")
    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()
    config = json.load(open(args.config_file, 'r', encoding="utf-8"))
    base_dir = config['system']['base_dir']
    os.makedirs(os.path.join(base_dir, config['name']), exist_ok=True)
    os.makedirs(os.path.join(base_dir, config['name'], "saved_models/"), exist_ok=True)
    
    logging.info("********* Model Configuration ************")
    args.config = config
    args.task = config['task']
    config['model']['gnn_layer'] = 12 - args.hops

    if args.dataset == 'politihop':
        config['system']['train_data'] = 'data/politihop_train_xh.json'
        config['system']['validation_data'] = 'data/politihop_valid_xh.json'
        config['system']['test_data'] = 'data/politihop_test_xh.json'
    elif args.dataset == 'adv':
        config['system']['train_data'] = 'data/politihop_train_xh_adv.json'
        config['system']['validation_data'] = 'data/politihop_valid_xh_adv.json'
        config['system']['test_data'] = 'data/politihop_test_xh_adv.json'
    elif args.dataset == 'liar':
        config['system']['train_data'] = 'data/liar_train_xh.json'
        config['system']['validation_data'] = 'data/liar_valid_xh.json'
        config['system']['test_data'] = 'data/liar_test_xh.json'
    elif args.dataset == 'politihop_all':
        config['system']['train_data'] = 'data/politihop_train_xh_all.json'
        config['system']['validation_data'] = 'data/politihop_valid_xh_all.json'
        config['system']['test_data'] = 'data/politihop_test_xh_all.json'
    elif args.dataset == 'liar_all':
        config['system']['train_data'] = 'data/liar_train_xh_all.json'
        config['system']['validation_data'] = 'data/liar_valid_xh_all.json'
        config['system']['test_data'] = 'data/liar_test_xh_all.json'
    elif args.dataset == 'fever':
        config['system']['train_data'] = 'data/fever_train_graph.json'
        config['system']['validation_data'] = 'data/fever_dev_graph.json'
        config['system']['test_data'] = 'data/fever_test_graph.json'
    else:
        print('INCORRECT DATASET!')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
    else:
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            args.fp16 = True 
    #### here we only support single GPU training
    n_gpu = 1
    
    logging.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        config["training"]["train_batch_size"] / args.gradient_accumulation_steps)
    args.max_seq_length = config["model"]["bert_max_len"]
    logging.info('batch size: {}'.format(args.train_batch_size))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set device
    args.device = device
    args.n_gpu = n_gpu

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    if config['task'] == 'hotpotqa':
        model = Model_Hotpot(args, config)
    elif config['task'] == 'fever':
        model = Model_FEVER(args, config)
    
    if args.fp16:
        model.half()
    model.network.to(device)
    
    
    ### Model Evaluation
    if args.test:
        model.load(os.path.join(base_dir, args.in_model))
        model.eval()
        if os.path.exists('atts.txt'):
            os.remove('atts.txt')
        for eval_file in [config["system"]['train_data'], config["system"]['validation_data'], config["system"]['test_data']]:
            if config['task'] == 'hotpotqa':
                final_pred = evaluation_hotpot(model, eval_file, config, args)
                json.dump(final_pred, open("out_dev.json", "w"))
            elif config['task'] == 'fever':
                auc, pred_dict, f1_mac = evaluation_fever(model, eval_file, config, args)
            
            with open('outputs/preds_' + args.in_model.split('.')[0] + '_' + eval_file.split('/')[-1], 'w+') as f:
                f.write(json.dumps(pred_dict))
    
    ### Model Training
    else:
        if args.in_model:
            model.load(os.path.join(base_dir, args.in_model))

        # final layers fine-tuning
        """for param in model.network.parameters():
            param.requires_grad = False
        
        model.network.final_layer.weight.requires_grad = True
        model.network.final_layer.bias.requires_grad = True
        model.network.pred_final_layer.weight.requires_grad = True
        model.network.pred_final_layer.bias.requires_grad = True"""

        # Prepare Optimizer
        param_optimizer = list(model.network.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        #param_optimizer = [n for n in param_optimizer if n[1].requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config["training"]["learning_rate"])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["training"]["warmup_proportion"], t_total=config["training"]["total_training_steps"])

        if args.fp16:
            try:
                from apex.optimizers import FusedAdam, fp16_optimizer
            except:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=config["training"]["learning_rate"],
                                bias_correction=False,
                                max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = fp16_optimizer.FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = fp16_optimizer.FP16_Optimizer(
                    optimizer, static_loss_scale=args.loss_scale)

        best_score = -1
        if config['task'] == 'hotpotqa':
            for index in range(config['training']['epochs']):
                best_score = train_hotpot(model, index, config, args, best_score, optimizer, scheduler)
        
        elif config['task'] == 'fever':
            for index in range(args.epochs):
                best_score = train_fever(model, index, config, args, best_score, optimizer, scheduler)
