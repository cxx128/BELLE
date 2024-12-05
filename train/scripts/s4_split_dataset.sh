#!/bin/bash

# 输入文件路径
input_file="/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_llama_0924_len1024.json"
# 输出文件前缀
output_prefix="/work/home/acehekbmzh/cxx/dataset/20240901pretrain/pretrain_all_llama_0924_len1024_"

# 计算总行数
total_lines=$(wc -l < "$input_file")
echo $total_lines
# 每个文件的行数
lines_per_file=$((total_lines / 10+1))
echo $lines_per_file
# 使用 split 命令分割文件
split -l $lines_per_file -d "$input_file" "$output_prefix"

echo "文件分割完成。"

# 获取分割后的文件列表
files=($(ls $output_prefix*))

# 遍历文件列表，重命名文件
for file in "${files[@]}"; do
    mv "$file" "${file}.json"

done

echo "文件分割并重命名完成。"