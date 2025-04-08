python /mnt/afs/chenxiaoxuan/BELLE/train/src/entry_point/preprocess_data_cpu.py \
  --workers_num 4 \
  --model_name_or_path /mnt/afs/chenxiaoxuan/hf_home/GLM-4-9B-Chat-HF \
  --model_max_length 1024 \
  --corpus_source_folder /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/corpus_folder \
  --output_folder  /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/tokenized_text_split \
  --output_merge_dataset_path /mnt/afs/chenxiaoxuan/BELLE/data/dataset/20240901pretrain/pretrain_all_len1024_testcpu.json \
  --output_multiple_dataset_config_path /mnt/afs/chenxiaoxuan/BELLE/train/configs/multiple_dataset_files_config.json



