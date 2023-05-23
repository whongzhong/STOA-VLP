# STOA-VLP: Spatial-Temporal Modeling of Object and Action for Video-Language Pre-training

This repository contains PyTorch implementation of our paper [STOA-VLP: Spatial-Temporal Modeling of Object and Action for Video-Language Pre-training](https://arxiv.org/abs/2302.09736) (AAAI 2023).

Although large-scale video-language pre-training models, which usually build a global alignment between the video and the text, have achieved remarkable progress on various downstream tasks, the idea of adopting fine-grained information during the pre-training stage is not well explored. In this work, we propose STOA-VLP, a pre-training framework that jointly models object and action information across spatial and temporal dimensions. More specifically, the model regards object trajectories across frames and multiple action features from the video as fine-grained features. Besides, We design two auxiliary tasks to better incorporate both kinds of information into the pre-training process of the video-language model. The first is the dynamic object-text alignment task, which builds a better connection between object trajectories and the relevant noun tokens. The second is the spatial-temporal action set prediction, which guides the model to generate consistent action features by predicting actions found in the text. Extensive experiments on three downstream tasks (video captioning, text-video retrieval, and video question answering) demonstrate the effectiveness of our proposed STOA-VLP (e.g. 3.7 Rouge-L improvements on MSR-VTT video captioning benchmark, 2.9% accuracy improvements on MSVD video question answering benchmark, compared to previous approaches).

## Requirement

```bash
conda create -n stoa-vlp python==3.6.8
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
pip install tensorflow==2.4.0
pip install lmdb
pip install pycocoevalcap
pip install scipy
yum install java java-1.8.0-openjdk-devel or apt-get install openjdk-8-jre
```

## Data Preparation

```bash
mkdir data
cd data
mkdir video_data
mkdir object_feature
mkdir models
```

## Model ckpts
Download CLIP (ViT-B/16) weight,
```bash
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```
put it under modules/clip.

### Webvid
Dataset are placed in data/video_data/${dataset_name}
Download webvid-2.5M dataset from  [link](https://github.com/m-bain/webvid). The downloaded video should be preprocessed using `utils/webvid_to_tfrecord.py`
### MSRVTT
Download the MSR-VTT data and video from [link](https://github.com/ArrowLuo/CLIP4Clip). 
### MSVD
Download the raw videos from [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/). The splits and raw captions can be found in the woderful job [collaborative-experts](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/msvd/README.md).
### LSMDC
Download the LSMDC dataset from [link](https://sites.google.com/site/describingmovies/download). The 1000 test clips data is in [here](https://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/LSMDC16_challenge_1000_publictect.csv). You should obtain permission from MPII to download and use the data.
### DiDeMo
Download the raw videos from [link](https://github.com/LisaAnne/LocalizingMoments). The splits can be found in [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo/README.md).

## Feature Extraction
### Object Feature extraction
Please follow [VinVL](https://github.com/pzzhang/VinVL) to extract object feature. For sampled frames from one video, extract objects for each frame. The resulted json file should be in following structure:

```json
[
	{
		'objects': [
			{
				'class': object_class,
				'rect': [x1, x2, y1, y2],
				'conf': class_confidence,
				}, % object_1
			...,
			{} % object_n
		]
		}, % frame_1
	{}, % frame_2
	...,
	{} % frame_n
]
```

### Object Feature Pre-process
Preprocessed object features are placed in data/object_feature/${dataset_name}.
It will process the given object feature when firstly run the dataloader with `modules/utils_object_tractor.py/object_loader_confident(vid_path_list, object_count, label_map)`. 
## Pre-training
Run the pre-training stage with the following script

```bash
python3 -m torch.distributed.launch \
./tasks/pre_training.py --do_train --num_thread_reader=4 \
--epochs=10 --batch_size=8 --n_display=50 \
--clip_archive data/models/clip \
--labelmap_path data/object_feature/label_map.json \
--train_csv ${DATA_PATH}/frozen_train_labels.json \
--val_csv ${DATA_PATH}/json_files/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/json_files/frozen_train_labels.json \
--output_dir ${OUTPUT_DIR} \
--lr 1e-5 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype pretrain --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${MODEL_PATH} \
--need_object \
--object_path data/object_feature/webvid \
--object_count 10 \
--object_layer 12 \
--object_fuse fusion \
--use_action
```
The pre-trained model is available [here](https://1drv.ms/u/s!Ag6-bOQBmqnBwa1aKEaFfE1bbRcaGw?e=QreZua).
## Fine-tuning and Evaluation for Downstream Tasks
Scripts for fine-tuning and evaluation for downstream tasks can be found under `scripts/{task}_{dataset}_.sh`
## Acknowledgements
Our code is built upon [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip), [CLIP (ViT-B/16)](https://github.com/openai/CLIP), [ORViT](https://github.com/eladb3/ORViT) and [UniVL](https://github.com/microsoft/UniVL).