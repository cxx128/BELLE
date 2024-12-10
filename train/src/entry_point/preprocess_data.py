
# 预处理会在当前目录下生成临时文件，注意运行之前使用pwd检查是否当前目录有足够的权限和存储空间

# 这里只有llama的数据预处理
from multiprocessing import Pool
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

import json
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time

sys.path.append('/mnt/afs/chenxiaoxuan/BELLE/train')


import pickle
import subprocess
from tqdm import tqdm
from typing import Any, Callable, Dict, List
from itertools import chain
from transformers import PreTrainedTokenizer
from functools import partial
from datasets import load_dataset
import argparse
import glob

# 内存查看
# Importing the library
import psutil 
def show_memory_use(rank):
    # Getting % usage of virtual_memory ( 3rd field)
    print(f'rank : {rank} RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print(f'rank : {rank} RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

def batch_grouped_pretrain_generate(
    rank:int,
    world_size:int,
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    examples: Dict[str, List[str]],
) -> Dict[str, List[List[int]]]:
    example_list=examples['text']
    examples_list=tqdm(example_list,desc=f"rank {rank} tokenize : ",position=rank)#,disable=True)
    
    # build grouped texts with format `X1 X2 X3 ... <eos> X1 X2 X3 ... [<eos>]`
    tokenizer_method=partial(tokenizer,add_special_tokens=False)
    token_ids_list: List[List[int]] = list(map(tokenizer_method,examples_list))

    token_ids_list = [
        token_ids['input_ids'] + [tokenizer.eos_token_id] for token_ids in token_ids_list
    ]
    concatenated_ids = list(chain(*token_ids_list))
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (len(concatenated_ids) // model_max_length) * model_max_length
    
    
    result = [
        concatenated_ids[i : i + model_max_length]
        for i in tqdm(list(range(0, total_length, model_max_length)),desc=f"rank {rank} concatenated_ids : ",position=rank+world_size)#,disable=True)
    ]
    return {"input_ids": result,"labels":result.copy()}


def count_lines(file_path):
    lines_num = 0
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(2 ** 20)
            if not data:
                break
            lines_num += data.count(b'\n')
    return lines_num



def ddp_main(rank,world_size,model_name_or_path,model_max_length,corpus_path,output_folder_and_prefix):  

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    
    
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    
    #rank_all = dist.get_rank()
    #device_id = rank_all % torch.cuda.device_count()
    
    print("Starting %d workers for building datasets ... " % rank)
    
    #tokenizer =LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    max_num = count_lines(corpus_path)        
    start=rank*max_num//world_size
    end=(rank+1)*max_num//world_size
    if end >max_num:
        end=max_num
        
    #dataset_writer = open("dataset-tmp-" + str(rank) + ".pt", "wb")
    pos = 0
    examples=[]
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        while pos < start:
            f.readline()
            pos += 1
        while True:
            line = f.readline().strip()
            pos += 1
            
            line = json.loads(line)
            examples.append(line['text'])
            if pos >= end:
                break
            

    examples={'text':examples}
    
    #batch_grouped_pretrain_generate_partial=partial(batch_grouped_pretrain_generate,model_max_length=model_max_length,tokenizer=tokenizer)
    #tokenized_text_split= map(batch_grouped_pretrain_generate_partial,examples) #
    tokenized_text_split=batch_grouped_pretrain_generate(rank,world_size,model_max_length,tokenizer,examples) #
    
    

    save_path=output_folder_and_prefix+str(rank)+'.json'
    with open(save_path,'w') as jsonfile:
        for t in tokenized_text_split['input_ids']:
            json.dump({'input_ids':t,'labels':t},jsonfile,ensure_ascii=False)
            jsonfile.write('\n')
    #save_path=output_folder_and_prefix+str(rank)+'.bin'
    #with open(save_path,'wb') as binfile:
    #    for t in tokenized_text_split['input_ids']:
    #        data_x={'input_ids':t,'labels':t}
    #        pickle.dump(data_x,binfile)
    #time.sleep(10)
    #train_data = load_dataset(
    #    "json", data_files=save_path, cache_dir=cache_dir
    #)
    dist.barrier()
    
    dist.destroy_process_group()
   
    print(f"rank : {rank} saved dataset file. path : {save_path}")

def init_slurm_env():  # 初始化slurm，国超步骤？？？
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id==0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))

        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        node_list = os.environ['SLURM_STEP_NODELIST']
        # node_list = os.environ['SLURM_STEP_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        stepid = os.environ["SLURM_STEP_ID"]
       

        tcp_port = os.environ.get('MASTER_PORT', 9904)


        os.environ['MASTER_PORT'] =str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)

        print('rank: {} world size: {} addr: {}  port: {}'.format(proc_id, ntasks, addr, os.environ['MASTER_PORT']))

def calculate_total_lines(input_file):
    # 使用 wc -l 计算总行数
    result = subprocess.run(['wc', '-l', input_file], capture_output=True, text=True)
    total_lines = int(result.stdout.split()[0])
    return total_lines

def split_file(input_file, output_prefix, num_parts=10):
    # 计算总行数
    total_lines = calculate_total_lines(input_file)
    print(f"总行数: {total_lines}")

    # 每个文件的行数
    lines_per_file = (total_lines // num_parts) + 1
    print(f"每部分的行数: {lines_per_file}")

    # 使用 split 命令分割文件
    subprocess.run(['split', '-l', str(lines_per_file), '-d', input_file, output_prefix], check=True)

def rename_files(output_prefix):
    # 获取分割后的文件列表
    # 使用glob来匹配所有以output_prefix开头的文件
    #files=subprocess.run(['ls', '/data/chenxiaoxuan/LLM_pretrain/dedup_debug/pretrain_0830_web_paper_book_split_*'], capture_output=True, text=True)
    files = glob.glob(f'{output_prefix}*') 
    # 遍历文件列表，重命名文件
    for file in files:
        new_file = f"{file}.json"
        os.rename(file, new_file)
        file_size=os.path.getsize(new_file)
        print(f'get split result file : {new_file} file size : {file_size} file_size_h : {file_size/(1024*1024*1024)} GB')

def main():
    init_slurm_env()   # 初始化国超
    
    # 参数设置
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--model_name_or_path", type=str, default='???/work/home/acehekbmzh/data/hf_home/linly_chinese_llama_7b_hf',
                        help="tokenizer过程中使用的tokenizer路径")   
    parser.add_argument("--model_max_length", type=str, default=1024,
                        help="每一条数据转化为token_ids的长度") 
    parser.add_argument("--corpus_source_folder", type=str, default='???/work/home/acehekbmzh/cxx/dataset/20240901pretrain/corpus_folder',
                        help="待处理的语料库放在这个文件夹下，每个最好不要大于15G，不要超过100G/4=25G")     
    parser.add_argument("--output_folder", type=str, default='???/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split',# required=True,
                        help="存储的多个（例如worldsize=256，就会存储256*n个）数据集处理结果")   
    parser.add_argument("--output_merge_dataset_path", type=str, default='???/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_llama_0924_len1024.json',
                        help="合并后，整个数据集的存储路径")

    args = parser.parse_args()
    
    
    model_name_or_path=args.model_name_or_path#'/work/home/acehekbmzh/data/hf_home/linly_chinese_llama_7b_hf/'
    model_max_length=int(args.model_max_length)
    corpus_source_folder=args.corpus_source_folder
    output_folder=args.output_folder
    output_merge_dataset_path=args.output_merge_dataset_path
    
    
    corpus_path_list=[os.path.join(corpus_source_folder,i) for i in os.listdir(corpus_source_folder)]
    
    output_folder_and_prefix_list=[os.path.join(output_folder,i.split('.json')[0])+'_len'+str(model_max_length)+'_' for i in os.listdir(corpus_source_folder)]

   
    
    
    
    
    # 检测现有可用卡数
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    print(f"got n_gpus : {n_gpus}")
    
    
    # 启动并行进程
    #mp.spawn(ddp_main, # mp.spwan启动一个进程，进程 = ddp_main(rank,args[0],args[1],args[2], ... )
    #         args=(world_size,model_name_or_path,model_max_length,corpus_path),  # 这个是ddp_main的参数，因为函数内需要dist.init进行初始化
    #         nprocs=world_size,  # 这个是mp.spwan的参数，启动的进程数
    #         join=True)
    # 所以，mp.spwan会按照nprocs创建一定数量的进程，并为每一个进程添加一个初始参数rank，其余参数则按照args的顺序自动填在rank后面

    rank=int(os.environ['RANK'])
    assert len(corpus_path_list)==len(output_folder_and_prefix_list)
    for index in range(len(corpus_path_list)):
        corpus_path=corpus_path_list[index]
        output_folder_and_prefix=output_folder_and_prefix_list[index]
        ddp_main(rank,world_size,model_name_or_path,model_max_length,corpus_path,output_folder_and_prefix)
        print(f'finish corpus_path : {corpus_path}  output_folder_and_prefix : {output_folder_and_prefix}')
        time.sleep(10)

    print(f'finish all dataset split')
    print(f'begin merge')
    
    
    path_out=[]
    for ofapl in output_folder_and_prefix_list:
        path_out+=[ofapl+str(index)+'.json' for index in range(world_size)]
    output_data=[]
    
    if rank == 0 :
        with open(output_merge_dataset_path,'w') as jsonfile:
            for p in tqdm(path_out):
                with open(p,'r') as infile:
                    for line in tqdm(infile):
                        line_s=line.strip()
                        jsonfile.write(line_s)
                        jsonfile.write('\n')    
        file_size=os.path.getsize(output_merge_dataset_path)
        print(f'finish merge ! result file {output_merge_dataset_path} file size : {file_size} file_size_h : {file_size/(1024*1024*1024)} GB')

        # 使用示例
        input_file = output_merge_dataset_path
        output_prefix = output_merge_dataset_path.split('.json')[0]+'_split_'

        split_file(input_file, output_prefix)
        rename_files(output_prefix)

        print("文件分割完成，并已重命名。")


if __name__ == "__main__":
    main()