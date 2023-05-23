from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataloaders.rawvideo_util import RawVideoExtractor
from modules.utils_object_tractor import object_loader_confident
import torch
from tqdm import tqdm

import json
class MSVD_DataLoader(Dataset):
    """MSVD dataset loader."""
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
        self.object_count = object_count
        self.data_path = data_path
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

        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        self.need_object = need_object
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        if self.need_object:
            assert os.path.exists(object_path)
            obj_path = os.path.join(object_path, f"object_attr_{self.subset}_{object_count}.pth")
            if os.path.isfile(obj_path):
                print(f"load from {obj_path}")
                self.object_bbox_dict = torch.load(obj_path)
            else:
                vid_path_list = [(vid, os.path.join(object_path, 'tsv_files', f'{vid}.json')) for vid in video_ids]
                
                self.object_bbox_dict = object_loader_confident(vid_path_list=vid_path_list, object_count=object_count, label_map=self.label_map)
                torch.save(self.object_bbox_dict, obj_path)
        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0

        if self.subset == 'train':
            self.sentences_dict = {}
            self.cut_off_points = []
            for video_id in video_ids:
                assert video_id in captions
                for cap in captions[video_id]:
                    cap_txt = " ".join(cap)
                    self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
                self.cut_off_points.append(len(self.sentences_dict))

        elif self.subset == 'test' or self.subset == 'val':
            self.sentences_dict = {}
            self.cut_off_points = []
            for video_id in video_ids:
                assert video_id in captions
                for cap in captions[video_id]:
                    cap_txt = " ".join(cap)
                    if video_id in self.sentences_dict:
                        self.sentences_dict[video_id][1].append(cap_txt)               
                    else:
                        self.sentences_dict[video_id] = [video_id, [cap_txt]]
                self.cut_off_points.append(len(self.sentences_dict))
        
        self.sentences = []
        self.sample_len = len(self.sentences_dict)
        for i in  self.sentences_dict:
            self.sentences.append(self.sentences_dict[i])

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences)))

        self.sample_len = len(self.sentences)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
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

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

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

    def get_caption_mask(self, pairs_text, pairs_mask, pairs_segment, video, video_mask):
        max_video_len = video_mask.shape[-1]
        fusion_labels = np.concatenate((video_mask[:,0:1],video_mask,pairs_mask),axis=-1)[0]
        sep_idx = np.expand_dims(np.concatenate((video_mask.sum(axis=-1),max_video_len+pairs_mask.sum(axis=-1)),axis=-1),axis=0)
        return fusion_labels,sep_idx

    def __getitem__(self, idx):
        
        video_id, caption = self.sentences[idx]
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

        if self.subset == 'train':

            pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
            video, video_mask = self._get_rawvideo(choice_video_ids)
            fusion_labels,sep_idx = self.get_caption_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)
            
            if self.need_object:
                return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list,  video_id
            else:
                return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels

        elif self.subset == 'val' or self.subset == 'test':

            captions = caption
            #video_id, captions = self.sentences[idx]
            pairs_texts = []
            pairs_masks = []
            pairs_segments = []
            choice_video_ids = []
            
            
            for caption in captions:
                pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
                pairs_texts.append(pairs_text)
                pairs_masks.append(pairs_mask)
                pairs_segments.append(pairs_segment)

            pairs_texts = np.concatenate(pairs_texts,axis=0)
            if pairs_texts.shape[0] < 100:
                pad_len = 100 - pairs_texts.shape[0]
                pairs_texts = np.concatenate((pairs_texts,np.zeros((pad_len,pairs_texts.shape[1]))),axis=0)
            pairs_masks = np.concatenate(pairs_masks,axis=0)
            if pairs_masks.shape[0] < 100:
                pad_len = 100 - pairs_masks.shape[0]
                pairs_masks = np.concatenate((pairs_masks,np.zeros((pad_len,pairs_masks.shape[1]))),axis=0)
            pairs_segments = np.concatenate(pairs_segments,axis=0)
            if pairs_segments.shape[0] < 100:
                pad_len = 100 - pairs_segments.shape[0]
                pairs_segments = np.concatenate((pairs_segments,np.zeros((pad_len,pairs_segments.shape[1]))),axis=0)



            video, video_mask = self._get_rawvideo(choice_video_ids)
            fusion_labels,sep_idx = self.get_caption_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)
            
            if self.need_object:
                return pairs_texts, pairs_masks, pairs_segments, video, video_mask, fusion_labels, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list,  video_id
            else:
                return pairs_texts, pairs_masks, pairs_segments, video, video_mask, fusion_labels, video_id
