#!/bin/bash
# In this example, we show how to train the model on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_multi-gpu_unsup_example.sh"
# about how to use PyTorch's distributed data parallel.

TRAIN_FILE_PATH=data/wiki1m_for_simcse.txt
EVAL_FILE_PATH=data/sts-dev.tsv

for seed in 42
do
model_name=bert
model_size=base
margin=0.44
alpha=2
uniform_t=6
temp=0.05
loss_lambda=0.5
loss_type=met
batch_size=128
learning_rate=1e-5
CUDA_DEVICES_NUM=0
MODEL_PATH="/home/LAB/niezj/pretrained_models/"${model_name}"-"${model_size}"-uncased"
SAVE_PATH=result/${seed}-temp${temp}-${loss_type}-lr${learning_rate}-bs${batch_size}-margin${margin}-alpha${alpha}-t${uniform_t}-loss_lambda${loss_lambda}-${model_name}-${model_size}-uncased

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python3 -u train.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE_PATH \
    --eval_path $EVAL_FILE_PATH \
    --output_dir $SAVE_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp ${temp} \
    --do_train \
    --do_eval \
    --seed ${seed} \
    --lambdas 0.2 \
    --margin ${margin} \
    --loss_lambda ${loss_lambda} \
    --loss_type ${loss_type} \
    --align_alpha ${alpha} \
    --uniform_t ${uniform_t} \
    --logger_dir ${MODEL_PATH} \
    "$@"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python3 simcse_to_huggingface.py --path=$SAVE_PATH
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python evaluation.py --model_name_or_path $SAVE_PATH --pooler cls_before_pooler --task_set full --mode test
done