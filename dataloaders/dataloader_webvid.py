from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from math import floor, ceil

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from random import sample
from dataloaders.rawvideo_util import RawVideoExtractor
from copy import deepcopy
import torch
import json
import os
import sys
import time
import cv2

import tensorflow as tf
from collections import defaultdict
from datetime import datetime as dt
import struct
from torchkit.data import example_pb2
from PIL import Image
import torch
from tqdm import tqdm
from modules.utils_object_tractor import object_loader_confident, get_attr_sent

class Formatter:
    def __init__(self,
                 file_dir,
                 file_name,
                 examples_per_shard=1000):
        #self.input_path = input_path
        self.file_dir = file_dir
        self.file_name = file_name
        self.index_path = os.path.join(self.file_dir,
                                       self.file_name + '.index')
        self.examples_per_shard = examples_per_shard
        self.vid2rs = self._read_index_file(self.index_path)
    def read_file_name(self, sid):
        if len(self.vid2rs[sid]) != 2:
            record, offset = self.vid2rs[sid][:2]
        else:
            record, offset = self.vid2rs[sid]
        return record.split('.')[-2]
        
    

    def _read_index_file(self, index_file):
        vid2rs = defaultdict(list)
        with open(index_file, 'r') as ifs:
            for line in ifs:
                vid, record_name, tf_record_offset = line.rstrip().split('\t')
                # vid might be duplicated?
                #if vid in vid2rs:
                #    import ipdb; ipdb.set_trace()
                vid2rs[vid].append(os.path.join(record_name))
                vid2rs[vid].append(int(tf_record_offset))
            return vid2rs

    def _parser(self, feature_list):
        for key, feature in feature_list:
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = np.fromstring(image_raw, dtype=np.uint8)
                image = image.reshape(-1, 256, 256, 3)
                #image = Image.fromarray(np.uint8(image[1])).convert('RGB')
        return image

    def read_video(self,sid):
        if len(self.vid2rs[sid]) != 2:
            print(self.vid2rs[sid])
            record, offset = self.vid2rs[sid][:2]
        else:
            record, offset = self.vid2rs[sid]
        with open(record, 'rb') as ifs:
            ifs.seek(offset)
            byte_len_crc = ifs.read(12)
            proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
            pb_data = ifs.read(proto_len)
            if len(pb_data) < proto_len:
                print(f'### Read pb_data error, '
                        f'proto_len: {proto_len}, '
                        f'pb_data len: {len(pb_data)}')
            example = example_pb2.Example()
            example.ParseFromString(pb_data)
            # keep key value in order
            feature = sorted(example.features.feature.items())
            record = self._parser(feature)
            return record

    def has_video(self,sid):
        if sid in self.vid2rs.keys():
            return True
        else:
            return False

class webvid_TrainDataLoader(Dataset):
    """WebVid train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            labelmap_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            need_object=False,
            object_path = None,
            object_count = 10,
            split=0,
    ):
        self.label_map = json.load(open(labelmap_path))
        file_name = 'path'
        self.tfrecord = Formatter(csv_path,file_name)
        self.data = json.load(open(json_path))
        self.need_object = need_object
        self.object_count = object_count
        split -= 1
        split_num = 300
        vid_num = len(self.data)
        per_count = ceil(vid_num / split_num)
        self.object_bbox_dict = {}
        captions_list = []
        #for i in tqdm(range(1, 300)):
        #    captions_list.extend(torch.load(os.path.join(object_path, "webvidpos", f"pos_{object_count}_{i}_{split_num}.pth")))
        #    break
        #data_split = self.data[split*per_count:(split+1)*per_count]
        id_caption_pos = torch.load(os.path.join(object_path, "webvidpos", f"postagging_ids_{object_count}_new.pth"))

        for i in tqdm(range(300)):
            temp_object_bbox_dict = torch.load(os.path.join(object_path, f"object_attr_{object_count}_{i}_{split_num}.pth"))
            self.object_bbox_dict.update(temp_object_bbox_dict)
           
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

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        
        self.sentences_dict = {}
        for itm in self.data:
            # print(itm)
            self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
        self.sample_len = len(self.sentences_dict)
        
        #id_caption_pos = id_caption_pos[1*8293:(1+1)*8293]
        self.id_caption_pos_dict = {}
        for itm in id_caption_pos:
            self.id_caption_pos_dict[len(self.id_caption_pos_dict)] = itm
            

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len
    
    def get_pretrain_mask(self, pairs_text, pairs_mask, pairs_segment, video, video_mask):
        max_video_len = video_mask.shape[-1]
        max_text_len = pairs_mask.shape[-1]
        fusion_labels = np.concatenate((video_mask[:,0:1],video_mask,pairs_mask),axis=-1)[0]
        sep_idx = np.expand_dims(np.concatenate((video_mask.sum(axis=-1),max_video_len+pairs_mask.sum(axis=-1)),axis=-1),axis=0)
        
       #仅MLM任务
            
        mlm_mask = np.array([i for i in range(max_video_len+2,sep_idx[0][1])])
        mlm_idx = np.random.binomial(1, 0.15, len(mlm_mask))

        mask = mlm_mask[mlm_idx==1]

        
        try:
            if len(mask) == 0:
                mask = sample(mlm_mask.tolist(),1)
            fusion_labels[mask] = -1
        except:
            fusion_labels[max_video_len+2] = -1

        return fusion_labels
        
    def _get_text(self, video_id, caption=None, pos_tagging=None):
        k = 1
        choice_video_ids = [video_id]

        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]


            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            if 0 in input_ids:
                input_ids = [item for item in input_ids if item > 0]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            
            if pos_tagging is not None:
                
                pos_tagging = ["PAD"] + pos_tagging + ["PAD"]
                pos_tagging = pos_tagging[:self.max_words]
            else:
                pos_tagging = [0] * len(input_ids)
            
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            while len(pos_tagging) < self.max_words:
                pos_tagging.append(0)
                
            assert len(pos_tagging) == self.max_words
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        if pos_tagging is not None:
            return pairs_text, pairs_mask, pairs_segment, choice_video_ids, np.array(pos_tagging)
        else:
            return pairs_text, pairs_mask, pairs_segment, choice_video_ids
            

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        # 1 x 12 x 1 x 3 x 256 x 256
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            #video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            #if os.path.exists(video_path) is False:
               # video_path = video_path.replace(".mp4", ".webm")
            
            raw_video_data = self.tfrecord.read_video(video_id)
            #raw_video_data[:,[0,1,2],:,:] = raw_video_data[:,[2,1,0],:,:]
            images = []
            for frame in raw_video_data:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.rawVideoExtractor.transform(Image.fromarray(frame_rgb).convert("RGB")))
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
                print("video path: {} error. video id: {}".format(video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        if self.need_object:
            object_tensor = self.object_bbox_dict[video_id]['objects']
            object_cls_tensor = self.object_bbox_dict[video_id]['obj_cls']
            conf_set = self.object_bbox_dict[video_id]['conf_set']
            conf_set = [(key, value) for key, value in conf_set.items()]
            tract_len = self.object_count * 2
            conf_set = conf_set[:tract_len]
            conf_set_len = len(conf_set)
            
            
            obj_prd_idx = np.random.binomial(1, 0.15, conf_set_len)
            
            pad_len = tract_len - conf_set_len
            if pad_len > 0:
                conf_set.extend([(0, 0) for i in range(pad_len)])
                obj_prd_idx = np.append(obj_prd_idx, [0 for i in range(pad_len)])
            id_list, conf_set = [itm[0] for itm in conf_set], [itm[1] for itm in conf_set]
            #id_select = torch.multinomial(torch.tensor(conf_set,dtype=torch.float), num_samples=self.object_count*2)
            #fusion_obj_id = id_list
            # select for obj predict 
            # length object_count * 2
            #fusion_obj_id.extend([0 for i in range(self.object_count*2 - len(fusion_obj_id))])
            id_list = torch.tensor(id_list).to(torch.int64)
            
            prd_id_list = id_list.clone()
            prd_id_list[obj_prd_idx!=1] = 0 # no mask place, remove obj id 
            
            
            
            id_text = [self.label_map['idx_to_label'].get(str(idx.item()), 0) for idx in id_list]
            obj_vocab_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))[0] if text is not 0 else 0 for text in id_text]
            obj_vocab_idx = torch.Tensor(obj_vocab_idx).to(torch.int64)
            attr_sentence = self.object_bbox_dict[video_id]['attr_sentence']
            attr_pairs_text, attr_pairs_mask, attr_pairs_segment, choice_video_ids, _ = self._get_text(video_id, attr_sentence)
        
            postag = self.id_caption_pos_dict[idx][1]
            pairs_text, pairs_mask, pairs_segment, choice_video_ids, postag = self._get_text(video_id, caption, postag)
            verb_pos_tag = id_list.new_zeros(postag.shape)
            noun_pos_tag = id_list.new_zeros(postag.shape)
            verb_pos_tag[postag == "VERB"] = 1
            noun_pos_tag[postag == "NOUN"] = 1
        else:   
            pairs_text, pairs_mask, pairs_segment, choice_video_ids, _ = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        fusion_labels = self.get_pretrain_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)
        
        if self.need_object:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list, obj_vocab_idx, prd_id_list, verb_pos_tag, noun_pos_tag
        else:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels



