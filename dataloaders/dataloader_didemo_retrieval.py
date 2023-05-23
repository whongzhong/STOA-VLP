from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
from dataloaders.rawvideo_util import RawVideoExtractor
import lmdb
import av
import random
import io
import torch
from modules.utils_object_tractor import object_loader_confident, get_attr_sent

class DiDeMo_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            labelmap_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            need_object=False,
            object_path = None,
            object_count = 10
    ):
        self.label_map = json.load(open(labelmap_path))
        lmdb_env = lmdb.open(features_path, lock=False)
        self.lmdb_txn = lmdb_env.begin()
        self.need_object = need_object
        self.object_count = object_count
        
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_rate = json.load(open('dataloaders/frame_rate.json','r'))
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
    
        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        
        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"]
            if video not in video_ids:
                continue

            # each video is split into 5-second temporal chunks
            # average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5

            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            # trick to save time on obtaining each video length
            # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
            # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        video_dict = {}
        for i in video_ids:
            video_dict[i] = i

        self.caption_dict = caption_dict
        self.video_dict = video_dict
  
        # Get all captions
        self.iter2video_pairs_dict = {}
        for video_id in self.caption_dict.keys():
            if video_id not in video_ids:
                continue
            caption = self.caption_dict[video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)
        if self.need_object:
            assert os.path.exists(object_path)
            obj_path = os.path.join(object_path, f"object_attr_{self.subset}_{object_count}.pth")
            if os.path.isfile(obj_path):
                print(f"load from {obj_path}")
                self.object_bbox_dict = torch.load(obj_path)
            else:
                vid_path_list = [(vid[0], os.path.join(object_path, self.subset, 'inference', 'vinvl_vg_x152c4','tsv_files', f'{vid[0]}_[0]_[31].json')) for vid in self.iter2video_pairs_dict.values()]
                
                self.object_bbox_dict = object_loader_confident(vid_path_list=vid_path_list, object_count=object_count, label_map=self.label_map)
                torch.save(self.object_bbox_dict, obj_path) 
            

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id, attr_sentence=None):
        if attr_sentence is None:
            caption = self.caption_dict[video_id]
        else:
            caption = attr_sentence

        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k, dtype=np.long)
        ends = np.zeros(k, dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            if attr_sentence is None:
                ind = r_ind[i]
                start_, end_ = caption['start'][ind], caption['end'][ind]
                
                words = self.tokenizer.tokenize(caption['text'][ind])
                starts[i], ends[i] = start_, end_
            else:
                words = self.tokenizer.tokenize(caption)
                

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, starts, ends

    def _get_rawvideo(self, idx, s, e):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)
        video_id = self.video_dict[idx][:self.video_dict[idx].find('.')]


        for i in range(len(s)):
            start_time = int(s[i])
            end_time = int(e[i])
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1

            cache_id = "{}_{}_{}".format(video_id, start_time, end_time)

            raw_video_data = self.lmdb_txn.get(video_id.encode())
            in_mem_bytes_io = io.BytesIO(raw_video_data)
            video_container = av.open(in_mem_bytes_io, mode='r', metadata_errors="ignore")
            images = []
            try:
                for frame in video_container.decode(video=0):
                    images.append(frame.to_image())
            except:
                print('Frame err',video_id)
            frame_rate = float(self.frame_rate[video_id])
            images = images[int(start_time*frame_rate):int(end_time*frame_rate)]
            for j in range(len(images)):
                images[j] = self.rawVideoExtractor.transform(images[j].convert("RGB"))
            raw_video_data = torch.tensor(np.stack(images))
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}, start: {}, end: {}".format(video_id, idx, start_time, end_time))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):

        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        if self.need_object:
            object_tensor = self.object_bbox_dict[video_id]['objects']
            attr_sentence = self.object_bbox_dict[video_id]['attr_sentence']
            object_cls_tensor = self.object_bbox_dict[video_id]['obj_cls']
            attr_pairs_text, attr_pairs_mask, attr_pairs_segment, starts, ends = self._get_text(video_id, sub_id, attr_sentence)
            
            conf_set = self.object_bbox_dict[video_id]['conf_set']
            conf_set = [(key, value) for key, value in conf_set.items()]
            conf_set = conf_set[:self.object_count*2]
            conf_set_len = len(conf_set)
            
            pad_len = self.object_count*2 - conf_set_len
            if pad_len > 0:
                conf_set.extend([(0, 0) for i in range(pad_len)])
            id_list, conf_set = [itm[0] for itm in conf_set], [itm[1] for itm in conf_set]
            
            id_list = torch.tensor(id_list).to(torch.int64)
            
            id_text = [self.label_map['idx_to_label'].get(str(idx.item()), 0) for idx in id_list]
            obj_vocab_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))[0] if text is not 0 else 0 for text in id_text]
            obj_vocab_idx = torch.Tensor(obj_vocab_idx).to(torch.int64)
        
        
        pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)
        video, video_mask = self._get_rawvideo(video_id, starts, ends)
       
        if self.need_object:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list
        else:  
            return pairs_text, pairs_mask, pairs_segment, video, video_mask