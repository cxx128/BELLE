#! /bin/bash
export CUDA_VISIBLE_DEVICES='2,3'
export MASTER_PORT="29556"
#export WANDB_PROJECT=...
#export WANDB_RUN_ID=...
export WANDB_RESUME=allow
export ABS_PATH=/mnt/afs/chenxiaoxuan
export PYTHONPATH="$ABS_PATH/BELLE/train"
export NCCL_P2P_DISABLE=1 # RTX4000系列显卡不支持NVLink或类似的高速互连技术
export NCCL_IB_DISABLE=1 # RTX4000系列显卡不支持NVLink或类似的高速互连技术
model_name_or_path=/mnt/afs/chenxiaoxuan/BELLE/train/work_dirs/sft/test_sft_origin_table_10epoch_current # or bloomz-7b1-mt

train_file=/mnt/afs/chenxiaoxuan/Projects_2024_cxx/20241118/data_belle/origin_table_current/origin_table_train_100epoch.json
validation_file=/mnt/afs/chenxiaoxuan/Projects_2024_cxx/20241118/data_belle/origin_table_val.json
#output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
output_dir="$ABS_PATH/BELLE/train/work_dirs/sft/test_sft_origin_table_110epoch_current"
mkdir -p ${output_dir}

cache_dir=/mnt/afs/chenxiaoxuan/hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

#FT
torchrun --master_port 29503 --nproc_per_node 2 src/entry_point/sft_train.py \
     --ddp_timeout 36000 \
     --model_name_or_path ${model_name_or_path} \
     --deepspeed /mnt/afs/chenxiaoxuan/BELLE/train/configs/deepspeed_config_stage3.json \
     --train_file ${train_file} \
     --validation_file ${validation_file} \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 1 \
     --gradient_accumulation_steps 1 \
     --num_train_epochs 1 \
     --model_max_length ${cutoff_len} \
     --save_strategy "steps" \
     --save_total_limit 1 \
     --learning_rate 8e-6 \
     --weight_decay 0.00001 \
     --warmup_ratio 0.05 \
     --lr_scheduler_type "cosine" \
     --logging_steps 10 \
     --evaluation_strategy "steps" \
     --torch_dtype "bfloat16" \
     --bf16 True \
     --seed 1234 \
     --gradient_checkpointing True \
     --cache_dir ${cache_dir} \
     --output_dir ${output_dir} \
     --qwen \
     --overwrite_output_dir \

     #--llama \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...


#LoRA with 8bit
# torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --use_int8_training \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...

# LoRA without 8bit
#torchrun --nproc_per_node 8 src/entry_point/sft_train.py \
#    --ddp_timeout 36000 \
#    --model_name_or_path ${model_name_or_path} \
#    --llama \
#    --use_lora \
#    --deepspeed configs/deepspeed_config_stage3.json \
#    --lora_config configs/lora_config_llama.json \
#    --train_file ${train_file} \
#    --validation_file ${validation_file} \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --num_train_epochs 10 \
#    --model_max_length ${cutoff_len} \
#    --save_strategy "steps" \
#    --save_total_limit 3 \
#    --learning_rate 3e-4 \
#    --weight_decay 0.00001 \
#    --warmup_ratio 0.01 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 10 \
#    --evaluation_strategy "steps" \
#    --torch_dtype "bfloat16" \
#    --bf16 \
#    --seed 1234 \
#    --gradient_checkpointing \
#    --cache_dir ${cache_dir} \
#    --output_dir ${output_dir} \
   # --use_flash_attention
   # --resume_from_checkpoint ...
