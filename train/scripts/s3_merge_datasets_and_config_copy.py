import json
import os
import pickle
from tqdm import tqdm
path=[]
world_size=128
path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/qwen_CLUE_len512_'+str(index)+'.json' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/CLUE_'+str(index)+'.bin' for index in range(world_size)]
path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/qwen_web_paper_book_len512_'+str(index)+'.json' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/20240901pretrain/tokenized_text_split/web_paper_book_'+str(index)+'.bin' for index in range(world_size)]
#path+=['/work/home/acehekbmzh/cxx/dataset/test/'+str(index)+'.json' for index in range(3)]






output_data=[]
output_file_path='/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_qwen_0919_len512.json'


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




with open(output_file_path,'w') as jsonfile:
    for p in tqdm(path):
        with open(p,'r') as infile:
            for line in tqdm(infile):
                line_s=line.strip()
                jsonfile.write(line_s)
                jsonfile.write('\n')






#with open('/work/home/acehekbmzh/cxx/BELLE/train/configs/multiple_dataset_files_config.json','w') as jsonfile:
#    json.dump({"path": [output_file_path]},jsonfile,ensure_ascii=False)