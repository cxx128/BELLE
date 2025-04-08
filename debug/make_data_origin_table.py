import json
import pandas as pd
import copy

a=[json.loads(l) for l  in open('/mnt/afs/chenxiaoxuan/Projects_2024_cxx/20241118/data_belle/feature_label.json')]

out_format={
    "conversations": [
        {
            "from": "human",
            "value": ""
        },
        {
            "from": "assistant",
            "value": ""
        }
    ],
    "history": []
}




out_x_all=[]

for seq_len,split_value in {"6k":80,"12k":40,"24k":20}.items():
    out_x_all=[]
    for ax in a:

        file_path=ax['file_path']
        label=ax['labels'][0][-1]
        if label=='SJ' or label=='F' or label=='B':
            continue

        # 读取Excel或csv文件
        sheet_name = 'Sheet1'  # 替换为你需要读取的工作表名称
        file_name=file_path
        if file_name.endswith('.xlsx')==True:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_name.endswith('.csv')==True:
            df = pd.read_csv(file_path)

        time_list=df.iloc[::split_value,0]
        #for i in range(21,22): # val data
        for i in range(1,21):   # train data
            x_list=df.iloc[::split_value,i]

            assert len(time_list)==len(x_list)
            out_x=copy.deepcopy(out_format)
            for j in range(len(time_list)):
                
                out_x["conversations"][0]["value"]+=f"({1000*time_list.iloc[j]:.2f},{(5*x_list.iloc[j]-12.5):.2f})\n"
            out_x["conversations"][1]["value"]=f"{label}"
            out_x_all.append(out_x)

    output_copy=copy.deepcopy(out_x_all)
    with open('/mnt/afs/chenxiaoxuan/BELLE/debug/long_seq_'+str(seq_len)+'.json','w') as jsonfile:
        for o in output_copy:
            json.dump(o,jsonfile,ensure_ascii=False)
            jsonfile.write('\n')
        
