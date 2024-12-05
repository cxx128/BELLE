module switch compiler/dtk/24.04

JOB_NAME=${JOB_NAME:-"chatglm3_pretrain0830_fp16_t21_len1024_l4_zero2"}
GPUS=${1:-32} # 256正式跑，测试用16
model_name_or_path=${2:-"/work/home/acehekbmzh/data/hf_home/ChatGLM3-6B/"}  # /public/home/hpctest_xjtu/data/hf_home/bloom-7b1 # or bloomz-7b1-mt
output_dir=${3:-"/work/home/acehekbmzh/cxx/BELLE/train/work_dirs/cxx_debug"}
batch_size=${4:-1}
gradient_accumulation_steps=${5:-1}
epochs=${6:-1}
learning_rate=${7:-1e-5} # 11.5之前默认的是2e-4
#train_file=${9:-'/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_0903.json'} # debug file
multiple_preprocessed_dataset_files_config=${8:-"/work/home/acehekbmzh/cxx/BELLE/train/configs/multiple_dataset_files_config_debug.json"}
validation_file=${10:-'/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_chatglm3_0915_val_00.json'}
negative=${11:-0.2}
deepspeed_config_file=${12:-"/work/home/acehekbmzh/cxx/BELLE/train/configs/deepspeed_config_zero2_fp16.json"}
PY_ARGS=${@:3}


GPUS_PER_NODE=${GPUS:-4}
if [ $GPUS_PER_NODE -ge 4 ]; then
  GPUS_PER_NODE=4
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

# HOME_PATH=/work/home/ac3y91rcdl
HOME_PATH=/work/home/acehekbmzh/cxx/
export TORCH_EXTENSIONS_DIR=${HOME_PATH}/.cache/torch_extensions_env_pt2
export HF_HOME=${HOME_PATH}/data/hf_home

#train_file=${HOME_PATH}/belle/data_generate/data_train_and_val/$train_file
#validation_file=${HOME_PATH}/belle/data_generate/data_train_and_val/$validation_file
mkdir -p ${output_dir}

cache_dir=/work/home/acehekbmzh/cxx/hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

set -x 

export MASTER_PORT=9912

partition=xahdnormal

SRUN_ARGS=''
OUTPUT_LOG_DIR=$output_dir
OUTPUT_TRAIN_DIR=$output_dir/train
mkdir -p ${OUTPUT_TRAIN_DIR}
now=$(date +"%Y%m%d_%H%M%S")

# -o $OUTPUT/exp_logger-%j-$now.log
# -w c14r2n[00,02-08]  c13r4n05,c14r2n[06-08],c14r4n01
# c13r4n05 好像有问题 -w c13r4n05
srun --partition=${partition} $SRUN_ARGS  \
    --job-name=${JOB_NAME} -n$GPUS --gres=dcu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1  --mem=55000 --cpus-per-task=${CPUS_PER_TASK} \
    -o $OUTPUT_LOG_DIR/exp_logger-%j-$now.log \
    python  /work/home/acehekbmzh/cxx/BELLE/train/src/entry_point/pt_train.py \
    --model_name_or_path ${model_name_or_path} \
    --ddp_timeout 36000 \
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
    --fp16 True \
    --seed 1234 \
    --cache_dir ${cache_dir} \
    --output_dir ${OUTPUT_TRAIN_DIR} \
    --overwrite_output_dir \
    --gradient_checkpointing True \
    #--bf16 \
    #--fp16 False \
    #--train_file ${train_file} \
    #--resume_from_checkpoint \




