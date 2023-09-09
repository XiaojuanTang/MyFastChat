#!/bin/bash
file_list=("/scratch/ml/xiaojuan/datasets/family-tree-counter/train.json" "/scratch/ml/xiaojuan/datasets/family-tree-sem/test.json" "/scratch/ml/xiaojuan/datasets/family-tree-sym/test.json" "/scratch/ml/xiaojuan/datasets/family-tree-counter/test.json" "/scratch/ml/xiaojuan/datasets/FOLIO/data/valid_ft.json")
log_list=("family-tree-counter-train.log" "family-tree-sem-test.log" "family-tree-sym-test.log" "family-tree-counter-test.log" "folio_valid.log")

model_path="/scratch/ml/xiaojuan/ft_models/llama2_7b_family_counter"
max_new_token=2048
num_gpus_per_model=4
num_gpus_total=4
for idx in "${!file_list[@]}"
do
    file="${file_list[$idx]}"
    log="${log_list[$idx]}"

    python evaluate.py \
        --model-path $model_path  \
        --bench-name $file \
        --output_log $log \
        --max-new-token $max_new_token \
        --num-gpus-per-model $num_gpus_per_model \
        --num-gpus-total $num_gpus_total
done


for i in 1 2 3 5
do
    python evaluate.py \
        --model-path $model_path \
        --bench-name /scratch/ml/xiaojuan/datasets/proofwriter/OWA/depth-$i/test_ft.json \
        --output_log pf-sem-d$i-test.log \
        --max-new-token $max_new_token \
        --num-gpus-per-model $num_gpus_per_model \
        --num-gpus-total $num_gpus_total

    python evaluate.py \
        --model-path $model_path \
        --bench-name /scratch/ml/xiaojuan/datasets/proofwriter_OWA_symbolic/depth-$i/test_ft.json \
        --output_log pf-sym-d$i-test.log \
        --max-new-token $max_new_token \
        --num-gpus-per-model $num_gpus_per_model \
        --num-gpus-total $num_gpus_total
done