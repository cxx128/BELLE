module switch compiler/dtk/24.04


JOB_NAME=${JOB_NAME:-"llama_preprocess_data"}
GPUS=${1:-64} # 256正式跑，测试用16
OUTPUT_LOG_DIR=${3:-"/work/home/acehekbmzh/cxx/BELLE/train/work_dirs/preprocess_data"}


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


cache_dir=/work/home/acehekbmzh/cxx/hf_power_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

set -x 

export MASTER_PORT=9912
export TQDM_PORT=9913

partition=xahdnormal

SRUN_ARGS=''


now=$(date +"%Y%m%d_%H%M%S")

# -o $OUTPUT/exp_logger-%j-$now.log
# -w c14r2n[00,02-08]  c13r4n05,c14r2n[06-08],c14r4n01
# c13r4n05 好像有问题 -w c13r4n05
srun --partition=${partition} $SRUN_ARGS  \
    --job-name=${JOB_NAME} -n$GPUS --gres=dcu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1  --mem=55000 --cpus-per-task=${CPUS_PER_TASK} \
    -o $OUTPUT_LOG_DIR/exp_logger-%j-$now.log \
    python  /work/home/acehekbmzh/cxx/BELLE/train/src/entry_point/preprocess_data.py \
    --model_name_or_path /work/home/acehekbmzh/data/hf_home/qwen1.5_7b \
    --model_max_length 1024 \
    --corpus_source_folder /work/home/acehekbmzh/cxx/dataset/20240901pretrain/corpus_folder \
    --output_folder  /work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split \
    --output_merge_dataset_path /work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_qwen_0927_len1024.json




