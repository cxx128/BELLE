torchrun --nproc_per_node 8 /mnt/afs/chenxiaoxuan/BELLE/train/src/entry_point/preprocess_data.py \
  --model_name_or_path /mnt/afs/chenxiaoxuan/hf_home/Qwen1.5_7b \
  --model_max_length 1024 \
  --corpus_source_folder /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/corpus_folder \
  --output_folder  /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/tokenized_text_split \
  --output_merge_dataset_path /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/pretrain_all_qwen_0927_len1024.json




