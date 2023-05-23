#!/bin/bash 

DATA_PATH=data/video_data/MSVD
MODEL_PATH=
OUTPUT_DIR=
python3 -m torch.distributed.launch \
./tasks/eval_captioning.py --do_train --num_thread_reader=4 \
--epochs=10 --batch_size=6 --n_display=50 \
--clip_archive modules/clip \
--labelmap_path dataloaders/label_map.json \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/MSVD_Videos \
--output_dir ${OUTPUT_DIR} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msvd_caption --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model  ${MODEL_PATH} \
--need_object \
--object_path data/object_feature/msvd_videos \
--object_count 10 \
--object_layer 12 \
--object_fuse fusion \
--use_action
