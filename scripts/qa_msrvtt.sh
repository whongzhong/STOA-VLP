#!/bin/bash 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
DATA_PATH=data/video_data
MODEL_PATH=
OUTPUT_DIR=
python3 -m torch.distributed.launch \
./tasks/eval_qa.py --do_train \
--num_thread_reader=4 \
--clip_archive modules/clip \
--labelmap_path dataloaders/label_map.json \
--epochs 5 \
--batch_size 6 \
--batch_size_val 32 \
--n_display 50 \
--datatype msrvtt_qa \
--expand_msrvtt_sentences \
--lr 1e-4 \
--msrvtt_train_csv ${DATA_PATH}/json_files/MSRVTT_train.9k.csv \
--msrvtt_val_csv /${DATA_PATH}/json_files/MSRVTT_JSFUSION_test.csv \
--msrvtt_train_json ${DATA_PATH}/json_files/MSRVTT_data.json \
--msrvtt_features_path ${DATA_PATH}/MSRVTT/MSRVTT_Videos \
--msrvtt_qa_train_json ${DATA_PATH}/MSRVTT/msrvtt_qa_annotation/train.jsonl \
--msrvtt_qa_val_json ${DATA_PATH}/MSRVTT/msrvtt_qa_annotation/val.jsonl \
--msrvtt_qa_test_json ${DATA_PATH}/MSRVTT/msrvtt_qa_annotation/test.jsonl \
--msrvtt_qa_anslabel_json ${DATA_PATH}/MSRVTT/msrvtt_qa_annotation/train_ans2label.json \
--output_dir ${OUTPUT_DIR} \
--init_model ${MODEL_PATH} \
--need_object \
--object_path data/object_feature/msr_vtt \
--object_count 10 \
--object_layer 12 \
--object_fuse fusion \
--use_action
