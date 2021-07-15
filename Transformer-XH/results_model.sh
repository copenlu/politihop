#!/bin/bash
#USAGE: bash results_model.sh model dataset

all=""
if test "$#" -eq 3; then
    all="_$3"
fi

arg_liar=""
if test $2 = "liar"; then
    arg_liar="--liar"
fi

python transformer_out_to_eval_out.py --preds outputs/preds_$1_$2_test_xh${all}.json --xh_dataset data/$2_test_xh${all}.json --output results/$1_$2_test${all}.tsv
python transformer_out_to_eval_out.py --preds outputs/preds_$1_$2_train_xh${all}.json --xh_dataset data/$2_train_xh${all}.json --output results/$1_$2_train${all}.tsv
python transformer_out_to_eval_out.py --preds outputs/preds_$1_$2_valid_xh${all}.json --xh_dataset data/$2_valid_xh${all}.json --output results/$1_$2_valid${all}.tsv

printf "\nTEST SET:\n"
python eval_model_outputs.py --preds results/$1_$2_test${all}.tsv --dataset data/$2_test.tsv --output_table ${arg_liar}
printf "\nTRAIN SET:\n"
python eval_model_outputs.py --preds results/$1_$2_train${all}.tsv --dataset data/$2_train.tsv --output_table ${arg_liar}
printf "\nDEV SET:\n"
python eval_model_outputs.py --preds results/$1_$2_valid${all}.tsv --dataset data/$2_valid.tsv --output_table ${arg_liar}
