from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from dataloaders.rawvideo_util import RawVideoExtractor
from modules.utils_object_tractor import object_loader_confident
import torch
from tqdm import tqdm
    
class MSRVTT_QA_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            json_path,
            features_path,
            labelmap_path,
            qa_anslabel_json_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,  # 256
            frame_order=0,
            slice_framepos=0,
            need_object=False,
            object_path = None,
            object_count = 10,
            split="val",
    ):
        self.label_map = json.load(open(labelmap_path))
        vid_set = set()
        self.sentences_dict = {}
        with open(json_path, 'r') as f:
            for items in f:
                item = json.loads(items)
                target_dict = {
                    'video_id': item['video_id'],
                    'question': item['question'],
                    'answer': item['answer'], 
                    'answer_type': item['answer_type']
                }
                
                vid_set.add(item['video_id'])
                self.sentences_dict[len(self.sentences_dict)] = target_dict

        self.need_object = need_object
        self.object_count = object_count
        
        if self.need_object:
            assert os.path.exists(object_path)
            obj_dir = os.path.join(object_path, f"msrvtt_qa_object_attr_{split}_{object_count}.pth")
            if os.path.isfile(obj_dir):
                print(f"load obj bbox from {obj_dir}")
                self.object_bbox_dict = torch.load(obj_dir)
            else:
                vid_path_list = [(vid, os.path.join(object_path, 'tsv_files', f'{vid}.json')) for vid in vid_set]
                
                self.object_bbox_dict = object_loader_confident(vid_path_list=vid_path_list, object_count=object_count, label_map=self.label_map)
                print(f"save obj bbox to {obj_dir}")
                torch.save(self.object_bbox_dict, obj_dir)

        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.cls_ids = {}
        self.emb_ids = {}

        with open(qa_anslabel_json_path, 'r') as f:
            item = json.load(f)
            for key, value in item.items():
                text = key
                cls_id = value
                emb_id = tokenizer.convert_tokens_to_ids( tokenizer.tokenize(text) )
            
                # print(text, cls_id, emb_id)

                self.cls_ids[key] = cls_id
                self.emb_ids[key] = emb_id


    def __len__(self):
        return len(self.sentences_dict)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

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

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            #if os.path.exists(video_path) is False:
               # video_path = video_path.replace(".mp4", ".webm")
            
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
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
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.sentences_dict[idx]['video_id']
        question = self.sentences_dict[idx]['question']
        answer = self.sentences_dict[idx]['answer']
        
        
        if self.need_object:
            object_tensor = self.object_bbox_dict[video_id]['objects']
            attr_sentence = self.object_bbox_dict[video_id]['attr_sentence']
            object_cls_tensor = self.object_bbox_dict[video_id]['obj_cls']
            attr_pairs_text, attr_pairs_mask, attr_pairs_segment, choice_video_ids = self._get_text(video_id, attr_sentence)
            
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
        
        align_text, align_mask, pairs_segment, choice_video_ids = self._get_text(video_id, question+' '+answer)
        question_text, question_mask, pairs_segment, choice_video_ids = self._get_text(video_id, question)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        answer_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))[0]

        if self.need_object:
            return question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list, video_id
        else:
            return question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask