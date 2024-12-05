import json
import os
import pickle
from tqdm import tqdm
path=[]
world_size=256
path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/chatglm3_CLUE_'+str(index)+'.json' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/CLUE_'+str(index)+'.bin' for index in range(world_size)]
path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/chatglm3_web_paper_book_'+str(index)+'.json' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/web_paper_book_'+str(index)+'.bin' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/test/'+str(index)+'.json' for index in range(3)]






output_data=[]
split_num=10
split_length=len(path)//10+1
output_file_path_prefix='/work/home/acehekbmzh/cxx/dataset/20240901pretrain/chatglm3_pretrain_0915_10_'

'''
with open(output_file_path,'w') as jsonfile:
    for p in tqdm(path):
        pretrain_dataset_reader=open(p,'rb')
        while True:
            try:
                instance = pickle.load(pretrain_dataset_reader) # data_x={'input_ids':t,'labels':t}
                json.dump(instance,jsonfile,ensure_ascii=False)
                jsonfile.write('\n')
            except EOFError:
                break
'''
for i in range(split_num):
    output_file_path=output_file_path_prefix+str(i)+'.json'
    start=i*split_length
    end=split_length*i+split_length
    print(f'start {start} end {end} output_file_path {output_file_path}')
    if end>len(path):
        end=len(path)
    with open(output_file_path,'w') as jsonfile:
        for p in tqdm(path[start:end]):
            with open(p,'r') as infile:
                for line in tqdm(infile):
                    jsonfile.write(line.strip())





with open('/work/home/acehekbmzh/cxx/BELLE/train/configs/multiple_dataset_files_config.json','w') as jsonfile:
    json.dump({"path": [output_file_path]},jsonfile,ensure_ascii=False)