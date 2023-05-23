#!/bin/bash

DATA_PATH=data/video_data
MODEL_PATH=
OUTPUT_DIR=
python3 -m torch.distributed.launch \
./tasks/eval_retrieval.py --do_train --num_thread_reader=2 \
--clip_archive modules/clip \
--labelmap_path dataloaders/label_map.json \
--epochs=5 --batch_size=8 --n_display=50 \
--data_path ${DATA_PATH}/DiDeMo/txt_db/didemo_retrieval \
--features_path ${DATA_PATH}/DiDeMo/vis_db/didemo \
--output_dir ${OUTPUT_DIR} \
--lr 1e-4 --max_words 64 --max_frames 12 --batch_size_val 8 \
--datatype didemo_retrieval --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${MODEL_PATH} \
 --need_object \
 --object_path data/object_feature/didemo_videos \
 --object_count 10 \
 --object_layer 12 \
 --object_fuse fusion \
 --use_action