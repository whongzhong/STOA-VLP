#!/bin/bash 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

DATA_PATH=data/video_data
MODEL_PATH=
OUTPUT_DIR=
python3 -m torch.distributed.launch \
./tasks/eval_qa.py --do_train \
--num_thread_reader=2 \
--clip_archive modules/clip \
--labelmap_path dataloaders/label_map.json \
--epochs 5 \
--batch_size 8 \
--batch_size_val 32 \
--n_display 50 \
--datatype msvd_qa \
--expand_msrvtt_sentences \
--lr 1e-4 \
--msvd_qa_train_json ${DATA_PATH}/MSVD/MSVD-QA/train_qa.json \
--msvd_qa_val_json ${DATA_PATH}/MSVD/MSVD-QA/val_qa.json \
--msvd_qa_test_json ${DATA_PATH}/MSVD/MSVD-QA/test_qa.json \
--msvd_qa_anslabel_json ${DATA_PATH}/MSVD/MSVD-QA/train_ans2label.json \
--msvd_features_path ${DATA_PATH}/MSVD/MSVD_Videos \
--output_dir ${OUTPUT_DIR} \
--init_model ${MODEL_PATH} \
--need_object \
--object_path data/object_feature/msvd_videos \
--object_count 10 \
--object_layer 12 \
--object_fuse fusion \
--use_action
