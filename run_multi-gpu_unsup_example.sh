#!/bin/bash
# In this example, we show how to train the model on unsupervised Wikipedia data with PyTorch's distributed data parallel.
# If you want to train it with the single GPU cards, see "run_multi-gpu_unsup_example.sh"
# about how to use PyTorch's distributed data parallel.
for seed in 42
do
model_name=bert
model_size=base
model_path="/home/LAB/niezj/pretrained_models/"${model_name}"-"${model_size}"-uncased"
margin=0.14
alpha=2
uniform_t=6
temp=0.05
loss_lambda=0.5
loss_type=uniform
batch_size=128
learning_rate=1e-5
CUDA_DEVICES_NUM=0
NUM_GPU=2
MODEL_PATH=MODEL_PATH=result/${seed}-temp${temp}-loss_type-${loss_type}-lr${learning_rate}-bs${batch_size}-margin${margin}-alpha${alpha}-t${uniform_t}-loss_lambda${loss_lambda}-${model_name}-${model_size}-uncased
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path ${model_path} \
    --train_file data/wiki1m_for_simcse.txt \
    --eval_path data/sts-dev.tsv \
    --output_dir $MODEL_PATH \
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
    --align_alpha ${align_alpha} \
    --align_flood ${align_flood} \
    --uniform_t ${uniform_t} \
    --mma_top_k ${mma_top_k} \
    --mma_norm_p ${mma_norm_p} \
    --logger_dir ${MODEL_PATH} \
    --iter 1 \
    "$@"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python3 simcse_to_huggingface.py --path=$MODEL_PATH
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_NUM python evaluation.py --model_name_or_path $MODEL_PATH --pooler cls_before_pooler --task_set full --mode test
done