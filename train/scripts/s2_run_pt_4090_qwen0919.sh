#module switch compiler/dtk/24.04

JOB_NAME=${JOB_NAME:-"qwen_pretrain"}
GPUS=${1:-0} # 256正式跑，测试用16
model_name_or_path=${2:-"/mnt/afs/chenxiaoxuan/hf_home/Qwen1.5_7b"}  # /public/home/hpctest_xjtu/data/hf_home/bloom-7b1 # or bloomz-7b1-mt
output_dir=${3:-"/mnt/afs/chenxiaoxuan/BELLE/train/work_dirs/debug"}
batch_size=${4:-6}
gradient_accumulation_steps=${5:-1}
epochs=${6:-1}
learning_rate=${7:-1e-5} # 11.5之前默认的是2e-4
#train_file=${9:-'/mnt/afs/chenxiaoxuan/dataset/20240901pretrain/pretrain_all_0903.json'} # debug file
multiple_preprocessed_dataset_files_config=${8:-"/mnt/afs/chenxiaoxuan/BELLE/train/configs/multiple_dataset_files_config_debug.json"}
validation_file=${10:-'/mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/qwen_CLUE_len1024_val.json'}
negative=${11:-0.2}
deepspeed_config_file=${12:-"/mnt/afs/chenxiaoxuan/BELLE/train/configs/deepspeed_config_stage3_llama0830_test.json"}
#deepspeed_config_file=${12:-"/mnt/afs/chenxiaoxuan/BELLE/train/configs/deepspeed_config.json"}
PY_ARGS=${@:3}


# HOME_PATH=/work/home/ac3y91rcdl
HOME_PATH=/mnt/afs/chenxiaoxuan/
#export TORCH_EXTENSIONS_DIR=${HOME_PATH}/.cache/torch_extensions_env_pt2
export HF_HOME=${HOME_PATH}/hf_home

#train_file=${HOME_PATH}/belle/data_generate/data_train_and_val/$train_file
#validation_file=${HOME_PATH}/belle/data_generate/data_train_and_val/$validation_file
mkdir -p ${output_dir}

cache_dir=/mnt/afs/chenxiaoxuan/hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

set -x 

export MASTER_PORT=9912

partition=xahdnormal

SRUN_ARGS=''
OUTPUT_LOG_DIR=$output_dir
OUTPUT_TRAIN_DIR=$output_dir/train
mkdir -p ${OUTPUT_TRAIN_DIR}


# -o $OUTPUT/exp_logger-%j-$now.log
# -w c14r2n[00,02-08]  c13r4n05,c14r2n[06-08],c14r4n01
# c13r4n05 好像有问题 -w c13r4n05
torchrun --nproc_per_node 2 /mnt/afs/chenxiaoxuan/BELLE/train/src/entry_point/pt_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --deepspeed ${deepspeed_config_file} \
    --multiple_preprocessed_dataset_files_config ${multiple_preprocessed_dataset_files_config} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${epochs} \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate ${learning_rate} \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --bf16 True \
    --seed 1234 \
    --cache_dir ${cache_dir} \
    --output_dir ${OUTPUT_TRAIN_DIR} \
    --overwrite_output_dir \
    --qwen \
    --gradient_checkpointing True \
    #--bf16 \
    #--fp16 False \
    #--train_file ${train_file} \
    #--resume_from_checkpoint \




