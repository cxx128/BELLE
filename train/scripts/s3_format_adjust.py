import json
import os
import tempfile
from tqdm import tqdm

def convert_large_json_streaming(input_file, output_file, chunk_size=100000):
    # 用于存储中间结果的临时文件列表
    temp_files = []

    # 逐行读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 解析每一行 JSON 数据
            data = json.loads(line)
            # 将 input_ids 的值添加到临时列表
            temp_input_ids = [data['input_ids']]

            # 当列表长度达到 chunk_size 时，写入临时文件
            if len(temp_input_ids) >= chunk_size:
                # 创建临时文件
                temp_fd, temp_path = tempfile.mkstemp(prefix='temp_', suffix='.json')
                os.close(temp_fd)
                temp_files.append(temp_path)

                # 将当前列表写入临时文件
                with open(temp_path, 'w', encoding='utf-8') as temp_outfile:
                    json.dump({'input_ids': temp_input_ids}, temp_outfile, ensure_ascii=False, indent=4)

                # 清空列表
                temp_input_ids = []

        # 处理剩余的数据
        if temp_input_ids:
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(prefix='temp_', suffix='.json')
            os.close(temp_fd)
            temp_files.append(temp_path)

            # 将当前列表写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as temp_outfile:
                json.dump({'input_ids': temp_input_ids}, temp_outfile, ensure_ascii=False, indent=4)

    # 打开最终输出文件，初始化 JSON 结构
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('{"input_ids": [')

        # 逐个读取并写入临时文件的数据
        first = True
        for temp_path in temp_files:
            with open(temp_path, 'r', encoding='utf-8') as temp_infile:
                temp_data = json.load(temp_infile)
                if not first:
                    outfile.write(', ')
                else:
                    first = False
                outfile.write(json.dumps(temp_data['input_ids'], ensure_ascii=False))
            os.remove(temp_path)  # 删除临时文件

        # 结束 JSON 数组
        outfile.write(']}')


# 使用示例
#input_json_file = '/path/to/large_input.json'  # 替换为你的输入 JSON 文件路径
#output_json_file = '/path/to/large_output.json'  # 替换为你想要保存输出文件的路径

#convert_large_json_streaming(input_json_file, output_json_file, chunk_size=100000)

folder='/work/home/acehekbmzh/cxx/dataset/20240901pretrain/test/'
temp_files=os.listdir(folder)

output_file='/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_0903_format_adjust_test.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('{"input_ids": [')

        # 逐个读取并写入临时文件的数据
        first = True
        for temp_path in tqdm(temp_files):
            with open(folder+temp_path, 'r', encoding='utf-8') as temp_infile:
                temp_data = [json.loads(l) for l in temp_infile]
                if not first:
                    outfile.write(', ')
                else:
                    first = False
                for t in temp_data:
                    outfile.write(json.dumps(t['input_ids'], ensure_ascii=False))
            #os.remove(temp_path)  # 删除临时文件
            print(f'finish input_ids temp_path : {temp_path}')
        # 结束 JSON 数组
        outfile.write('],"labels":[')

                # 逐个读取并写入临时文件的数据
        first = True
        for temp_path in tqdm(temp_files):
            with open(folder+temp_path, 'r', encoding='utf-8') as temp_infile:
                temp_data = [json.loads(l) for l in temp_infile]
                if not first:
                    outfile.write(', ')
                else:
                    first = False
                for t in temp_data:
                    outfile.write(json.dumps(t['input_ids'], ensure_ascii=False))
            #os.remove(temp_path)  # 删除临时文件
            print(f'finish labels temp_path : {temp_path}')
        outfile.write(']}')