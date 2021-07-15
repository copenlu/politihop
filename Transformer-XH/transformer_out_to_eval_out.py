import argparse
import json
import numpy as np
import pandas as pd


label_dict = {
	0: 'false',
	1: 'half-true',
	2: 'true'
}


label_dict_nee = {
	0: 'not enough evidence',
	1: 'false',
	2: 'half-true',
	3: 'true'
}


def xh_list_to_dict(jsons):
	xh_dict = {}
	for xh_json in jsons:
		xh_json = json.loads(xh_json)
		xh_dict[str(xh_json['qid'])] = xh_json

	return xh_dict


def get_ind_dict(xh_claim):
	ind_dict = {}
	for i in range(len(xh_claim['node'])):
		ind_dict[i] = xh_claim['node'][i]['sent_num']

	return ind_dict


def node_to_sent_ids(ids, ind_dict):
	return [ind_dict[x] for x in ids]


def top_ids(scores, k):
	return list(np.argsort(scores)[-k:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, help='path to the json with predictions')
    parser.add_argument('--xh_dataset', type=str, help='path to the xh format dataset')
    parser.add_argument('--output', type=str, help='path to the output file')
    parser.add_argument('--k', type=int, default=6, help="number of top sentences to retrieve as evidence")
    args = parser.parse_args()

    preds = json.load(open(args.preds))
    xh_data = open(args.xh_dataset).readlines()
    xh_data = xh_list_to_dict(xh_data)
    
    df_out = pd.DataFrame(columns=['article_id', 'pred_evidence', 'pred_label'])
    for art_id in preds:
    	label = preds[art_id][0]

    	ind_dict = get_ind_dict(xh_data[art_id])
    	evidence = node_to_sent_ids(top_ids(preds[art_id][1], args.k), ind_dict)
    	df_out = df_out.append({
    		'article_id': art_id,
    		'pred_evidence': json.dumps(evidence),
    		'pred_label': label_dict[label]}, ignore_index=True)
    df_out.to_csv(args.output, index=False, sep='\t')
