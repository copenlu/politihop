from __future__ import print_function
import argparse
import json
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from itertools import chain
from collections import Counter


def print(*args):
    __builtins__.print(*("%.3f" % a if isinstance(a, float) else a
                         for a in args))


def get_pred_gold_sets(gold, pred):
    try:
        gold = list(chain.from_iterable([x.split(',') for x in chain.from_iterable(gold.values())]))
        gold = set([int(x) for x in gold])
        pred = set(pred)
        return pred, gold
    except ValueError:
        print('Dataset Error: Could not parse sentence ids')
        return None, None


def get_chain_hits(gold, pred):
    """
    The function iterates over the gold chain, checking how many sentences
    can be found among the predicted sentences.
    Gold chain entry is a string of comma separated ids of semantically equivalent sentence.
    This means it's enough if at least one of these ids is found in the predicted chain. 
    """
    gold = [[int(y) for y in x.split(',')] for x in gold]
    return len([True for x in gold if len(set(x).intersection(pred)) > 0])


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


def plot_confusion_matrix(gold, pred, im_path):
    conf_matrix = confusion_matrix(gold, pred)
    
    label_names = ['false', 'half-true', 'true']
    if conf_matrix.shape[0] == 2:
        label_names = ['false', 'true']
    
    ax = sn.heatmap(conf_matrix, annot=True, cbar=False, fmt='d', xticklabels=label_names, yticklabels=label_names)
    ax.set(xlabel="predictions", ylabel="true labels",)

    plt.show()
    plt.savefig('imgs/' + im_path)


def max_chain_recall(gold, pred):
    """
    check which gold chain had the highest recall among the predicted sentences
    :param gold: dict of the annotated chains of sentences
    :param pred: set of predicted sentence ids
    """
    max_chain = '0'
    max_recall = 0
    for k in gold:
        recall = get_chain_hits(gold[k], pred) / len(gold[k])
        if recall >= recall:
            max_recall = recall
            max_chain = k
    return max_chain, max_recall


def fever_score(evi_gold, evi_pred, lab_gold, lab_pred):
    """
    fever accuracy measure: a prediction is correct iff at least one full chain retrieved and label correct
    """
    fever_counter = 0
    for i in range(len(lab_gold)):
        _, max_recall = max_chain_recall(evi_gold[i], set(evi_pred[i]))
        if lab_gold[i] == lab_pred[i] and max_recall == 1:
            fever_counter += 1

    return fever_counter / len(lab_gold)


def joint_f1(evi_precision, evi_recall, lab_precision, lab_recall):
    """
    Computes the joint f1 score (like in HotPotQA).
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    where joint_precision = evi_precision * lab_precision
    and joint_recall = evi_reca;; * lab_recall
    """
    joint_precision = evi_precision * lab_precision
    joint_recall = evi_recall * lab_recall
    if joint_precision == 0 and joint_recall == 0:
        return 0
    return 2 * joint_precision * joint_recall / (joint_precision + joint_recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, help='path to the tsv with predictions')
    parser.add_argument('--dataset', type=str, help='path to the data tsv with the annotated labels and evidence')
    parser.add_argument('--liar', default=False, action='store_true', help='For evaluating on the LIAR-PLUS dataset')
    parser.add_argument('--output_table', default=False, action='store_true', help='print results in a latex table friendly format')
    args = parser.parse_args()
    
    df_pred = pd.read_csv(args.preds, sep='\t')
    df_gold = pd.read_csv(args.dataset, sep='\t')

    df_pred.set_index('article_id', inplace=True)
    df_gold.set_index('article_id', inplace=True)
    df_gold = df_gold[df_gold.index.isin(df_pred.index)]

    assert len(df_pred) == len(df_gold)

    df_pred.sort_index(inplace=True)
    df_gold.sort_index(inplace=True)
    assert np.all(df_pred.index == df_gold.index)

    evi_gold = []
    if args.liar:
        evi_gold = [{'0': [str(y) for y in json.loads(x)]} for x in df_gold['summary_ids']]
    else:
        evi_gold = [json.loads(x) for x in df_gold['annotated_evidence']]

    evi_pred = [json.loads(x) for x in df_pred['pred_evidence']]

    lab_gold = []
    if args.liar:
        lab_gold = ['false' if x == 'pants-fire' else x for x in df_gold['gold_label'].values]
    else:
        lab_gold = df_gold['annotated_label'].values
    lab_pred = df_pred['pred_label'].apply(str).apply(lambda x: x.lower()).values

    ev_f1s, ev_recs, ev_precs = [], [], []

    for i in range(len(lab_gold)):
        evi_pred_set, evi_gold_set = get_pred_gold_sets(evi_gold[i], evi_pred[i])
        if not evi_gold_set:
            exit()

        ev_f1s.append(evidence_f1(evi_gold_set, evi_pred_set))
        ev_precs.append(evidence_prec(evi_gold_set, evi_pred_set))
        ev_recs.append(evidence_recall(evi_gold_set, evi_pred_set))

    plot_confusion_matrix(lab_gold, lab_pred, args.preds.split('/')[-1].split('.')[0] + '.png')

    label_acc = accuracy_score(lab_gold, lab_pred)
    label_f1 = f1_score(lab_gold, lab_pred, average='macro')
    label_precision = precision_score(lab_gold, lab_pred, average='macro')
    label_recall = recall_score(lab_gold, lab_pred, average='macro')

    evi_precision = np.mean(ev_precs)
    evi_recall = np.mean(ev_recs)
    evi_f1 = np.mean(ev_f1s)

    j_f1 = joint_f1(evi_precision, evi_recall, label_precision, label_recall)
    fever = fever_score(evi_gold, evi_pred, lab_gold, lab_pred)

    print('========LABEL PREDICTION========')
    print('LABEL ACCURACY: ', label_acc)
    print('LABEL PRECISION: ', label_precision)
    print('LABEL RECALL: ', label_recall)
    print('LABEL F1: ', label_f1)

    print('\n========EVIDENCE PREDICTION========')
    print('EVIDENCE PRECISION: ', evi_precision)
    print('EVIDENCE RECALL: ', evi_recall)
    print('EVIDENCE F1: ', evi_f1)

    print('\n========JOINT METRICS========')
    print('FEVER SCORE: ', fever)
    print('JOINT F1: ', j_f1)

    if args.output_table:
        print('\n')
        print(' & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f}'.format(label_f1, label_acc, evi_f1, evi_precision, fever))
        print('\n')
