import os
import json

#大文件一分为多，不是精确等分，按行等分，如果每行之间数据长短差太多，可能文件之间大小差很多
def split_large_json(input_file: str, output_folder : str, output_prefix=None, max_lines_per_file=None, num_chunk=None, max_size_GB=None):
    assert max_lines_per_file or num_chunk or max_size_GB
       
    if output_prefix==None:
        output_prefix=input_file.split('.json')
        assert len(output_prefix)==2
        output_prefix=output_prefix[0]
    
     # 首先计算文件中的行总数
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as file:
        for _ in file:
            total_lines += 1
    
    print(f"total_lines : {total_lines}")
    
    if max_lines_per_file==None and num_chunk and max_size_GB==None:
        max_lines_per_file = total_lines // num_chunk +1
        print(f"Get the value of max_lines_per_file={max_lines_per_file} from num_chunk={num_chunk}")
    elif max_size_GB and max_lines_per_file==None and num_chunk==None:
        file_size=os.path.getsize(input_file)
        file_size_h=file_size/(1024*1024*1024)
        num_chunk=int(file_size_h/max_size_GB)+1
        max_lines_per_file = total_lines // num_chunk +1
        print(f"Get the value of max_lines_per_file={max_lines_per_file} from max_size_GB={max_size_GB}")
    elif max_lines_per_file and num_chunk==None and max_size_GB==None:
        print(f"Get the value of max_lines_per_file={max_lines_per_file}")
    else:
        raise ValueError("There must be two 'None' in variable values : max_lines_per_file, max_lines_per_file, max_lines_per_file")


    # 用于跟踪当前文件的行数和当前数据
    current_batch = []
    batch_count = 0
    file_count = 0
    
    # 逐行读取原始文件
    with open(input_file, 'r') as infile:
        for line in infile:
            # 解析当前行的 JSON 对象
            data = json.loads(line.strip())
            current_batch.append(data)
            
            # 如果当前批次达到了最大行数，保存文件并重置批次
            if len(current_batch) >= max_lines_per_file:
                output_filename = os.path.join(output_folder,f"{output_prefix}_split_{file_count}.json")
                with open(output_filename, 'w') as outfile:
                    for c in current_batch:
                        json.dump(c, outfile, ensure_ascii=False)
                        outfile.write('\n')
                print(f"Saved {output_filename} with {len(current_batch)} items.")
                
                # 重置当前批次
                current_batch = []
                file_count += 1

        # 保存剩余的部分（如果有）
        if current_batch:
            output_filename = os.path.join(output_folder,f"{output_prefix}_split_{file_count}.json")
            with open(output_filename, 'w') as outfile:
                for c in current_batch:
                    json.dump(c, outfile, ensure_ascii=False)
                    outfile.write('\n')
            print(f"Saved {output_filename} with {len(current_batch)} items.")


# 使用函数分割你的json文件，请替换为你的json文件路径
# split_large_json_file('/mnt/afs/chenxiaoxuan/Dataset_folder/SkyPile-150B/output.json')

# 使用示例
split_large_json(
    input_file='/mnt/afs/chenxiaoxuan/Dataset_folder/MNBVC/output_split_14.json',
    output_folder='/mnt/afs/chenxiaoxuan/Dataset_folder/MNBVC',
    max_lines_per_file=None, 
    num_chunk=None, 
    max_size_GB=1,
    )


