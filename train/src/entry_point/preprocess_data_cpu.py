
# 预处理会在当前目录下生成临时文件，注意运行之前使用pwd检查是否当前目录有足够的权限和存储空间

# 这里只有llama的数据预处理
from multiprocessing import Pool
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer

import json
import os
import sys
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

from multiprocessing import Pool

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

    tokenized_text_split=batch_grouped_pretrain_generate(rank,world_size,model_max_length,tokenizer,examples) #
    

    save_path=output_folder_and_prefix+str(rank)+'.json'
    with open(save_path,'w') as jsonfile:
        for t in tokenized_text_split['input_ids']:
            json.dump({'input_ids':t,'labels':t},jsonfile,ensure_ascii=False)
            jsonfile.write('\n')

    print(f"rank : {rank} saved dataset file. path : {save_path}")


def calculate_total_lines(input_file):
    # 使用 wc -l 计算总行数
    result = subprocess.run(['wc', '-l', input_file], capture_output=True, text=True)
    total_lines = int(result.stdout.split()[0])
    return total_lines

def split_file(input_file, output_prefix, file_size, max_size_GB):
    # 分割后的子文件数量
    num_parts=int(file_size/max_size_GB)+1
    
    # 计算总行数
    total_lines = calculate_total_lines(input_file)
    print(f"总行数: {total_lines}")

    # 每个文件的行数
    lines_per_file = (total_lines // num_parts) + 1
    print(f"每部分的行数: {lines_per_file}")

    # 使用 split 命令分割文件
    subprocess.run(['split', '-l', str(lines_per_file), '-d', input_file, output_prefix], check=True)

def rename_files(output_prefix):
    new_file_list=[]
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
        new_file_list.append(new_file)
    return new_file_list

def check_corpus_file_size(file_path,max_size_GB):
    file_size=os.path.getsize(file_path)
    file_size_h=file_size/(1024*1024*1024)
    if file_size_h > max_size_GB :
        raise ValueError(
            f"\nFile size exceed the max recommended size ! \n"
            f"Use 'split_big_json_file.py' to split the big file !\n"
            f"File path : {file_path} \n"
            f"File size : {file_size_h} GB \n"
            f"Max recommended size : {max_size_GB} GB \n"
            )
    else :
        print(
            f"File size is proper !\n"
            f"File path : {file_path} \n"
            f"File size : {file_size_h} GB \n"
        )

def log_success(result):
    print("Task completed successfully:", result)

def log_error(e):
    print("Task failed with error:", e)


def main():
    # 每个语料文件的大小上限（GB），超过则报错
    max_size_GB = 15

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--workers_num", type=int, default=-1,
                        help="并行线程数")   
    parser.add_argument("--model_name_or_path", type=str, default='???/mnt/afs/chenxiaoxuan/hf_home/Qwen2.5-7B-Instruct',
                        help="tokenizer过程中使用的tokenizer路径")   
    parser.add_argument("--model_max_length", type=str, default=1024,
                        help="每一条数据转化为token_ids的长度") 
    parser.add_argument("--corpus_source_folder", type=str, default='???/mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/corpus_folder',
                        help="待处理的语料库放在这个文件夹下，每个最好不要大于15G，不要超过100G/4=25G。（20250123-不超过10G）")     
    parser.add_argument("--output_folder", type=str, default='???/mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/tokenized_text_split',# required=True,
                        help="存储的多个（例如worldsize=256，就会存储256*n个）数据集处理结果")   
    parser.add_argument("--output_merge_dataset_path", type=str, default='???/mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/pretrain_all_len1024_testcpu.json',
                        help="合并后，整个数据集的存储路径")
    parser.add_argument("--output_multiple_dataset_config_path", type=str, default='???/home/ma-user/work/chenxiaoxuan/BELLE/train/configs/multiple_dataset_files_config.json',
                        help="生成预训练多数据集输入的config文件")

    args = parser.parse_args()
    
    world_size=args.workers_num
    model_name_or_path=args.model_name_or_path #'/work/home/acehekbmzh/data/hf_home/linly_chinese_llama_7b_hf/'
    model_max_length=int(args.model_max_length)
    corpus_source_folder=args.corpus_source_folder
    output_folder=args.output_folder
    output_merge_dataset_path=args.output_merge_dataset_path
    output_multiple_dataset_config_path=args.output_multiple_dataset_config_path
    
    corpus_path_list=[os.path.join(corpus_source_folder,path) for path in os.listdir(corpus_source_folder)]
    
    output_folder_and_prefix_list=[os.path.join(output_folder,path.split('.json')[0])+'_len'+str(model_max_length)+'_' for path in os.listdir(corpus_source_folder)]

    # 确保存储中间过程文件的目录是空的
    if len(os.listdir(args.output_folder)) > 0 :
        raise ValueError(
                f"Output directory ({args.output_folder}) already exists and is not empty. "
                "Please clear the directory and retry."
        )

    # 检查每个语料文件大小，确保每一个文件不会太大导致内存溢出
    for c in corpus_path_list:
        check_corpus_file_size(file_path=c,max_size_GB=max_size_GB)
    


    assert len(corpus_path_list)==len(output_folder_and_prefix_list)
    for index in range(len(corpus_path_list)):
        corpus_path=corpus_path_list[index]
        output_folder_and_prefix=output_folder_and_prefix_list[index]
        pool = Pool(world_size)
        for rank in range(world_size):
            pool.apply_async(func=ddp_main, args=[rank,world_size,model_name_or_path,model_max_length,corpus_path,output_folder_and_prefix],callback=log_success, error_callback=log_error) 
            #ddp_main(rank,world_size,model_name_or_path,model_max_length,corpus_path,output_folder_and_prefix)
        
        print(f'finish corpus_path : {corpus_path}  output_folder_and_prefix : {output_folder_and_prefix}')
        pool.close()
        pool.join()
        time.sleep(10)

    print(f'finish all dataset split')
    print(f'begin merge')
    
    
    path_out=[]
    for ofapl in output_folder_and_prefix_list:
        path_out+=[ofapl+str(index)+'.json' for index in range(world_size)]
   
    
    #将多线程产生的多个文件写入子文件
    with open(output_merge_dataset_path,'w') as jsonfile:
        for p in tqdm(path_out):
            with open(p,'r') as infile:
                for line in tqdm(infile):
                    line_s=line.strip()
                    jsonfile.write(line_s)
                    jsonfile.write('\n') 
            
            # 写入合并文件路径后，就删除多线程过程中产生的文件。（如果不删除，就需要两倍的空间才能完成整个数据集预处理）
            if os.path.exists(p):
                os.remove(p)
                print(f"'{p}' 文件已被删除")
            else:
                raise ValueError(f"'{p}' 文件不存在")   
            
    file_size=os.path.getsize(output_merge_dataset_path)
    file_size_h=file_size/(1024*1024*1024)
    print(f'finish merge ! result file {output_merge_dataset_path} file size : {file_size} file_size_h : {file_size_h} GB')





    # 使用示例
    input_file = output_merge_dataset_path
    output_prefix = output_merge_dataset_path.split('.json')[0]+'_split_'

    split_file(input_file, output_prefix,file_size_h,max_size_GB)
    split_dataset_path_list=rename_files(output_prefix)

    print("文件分割完成，并已重命名。")

    with open(output_multiple_dataset_config_path,'w') as jsonfile:
        path_list={
            "path": []
        }
        for s in split_dataset_path_list:
            path_list["path"].append(s)
        json.dump(path_list,jsonfile,ensure_ascii=False)

    print(f"已生成mutiple_dataset_config,路径为：{output_multiple_dataset_config_path}")


if __name__ == "__main__":
    main()