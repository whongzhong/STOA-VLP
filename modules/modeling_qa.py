import json
import numpy as np

import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
from torch.nn import functional as F

import logging

from modules.module_clip import CLIP, convert_weights, LayerNorm
from modules.module_cross import CrossConfig, CrossModel, Transformer as TransformerClip
from modules.until_module import PreTrainedModel, CrossEn, AllGather
logger = logging.getLogger(__name__)
allgather = AllGather.apply


class CLIP4ClipPreTrainedModel(PreTrainedModel, torch.nn.Module):
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2,object_count=10, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        task_config.object_count = object_count

        if state_dict is None:
            print("state_dict is None, init_preweight from CLIP")
            state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16", clip_archive=task_config.clip_archive)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)
        
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)


        return model


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)

        self.task_config = task_config
        self.ignore_video_index = -1
        self.NLGLoss = nn.CrossEntropyLoss()
        self.dropout =nn.Dropout(0.5)
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            linear_patch=self.linear_patch,
            object_num=task_config.object_count
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        scale = vision_width ** -0.5
        
        self.obj_ln = LayerNorm(vision_width)
        self.obj_proj = nn.Parameter(scale * torch.randn(vision_width, transformer_width))
        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()

        self.apply(self.init_weights)
    
    def forward(self, question_ids, question_mask, pairs_segment,video, video_mask, answer_ids, align_text, align_mask, objects = None, object_layer=[], attr_triple = None, attr_sim=False, insert_attr=False, replace_cls=False, phrase_fuse="None", object_fuse="attention", obj_cls_tuple=None, pos_tag=None, use_action = False, action_only = False, use_sim=False):

        question_ids = question_ids.view(-1, question_ids.shape[-1]) 
        question_mask = question_mask.view(-1, question_mask.shape[-1]) 
        align_text = align_text.view(-1, align_text.shape[-1]) 
        align_mask = align_mask.view(-1, align_mask.shape[-1]) 
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w) 
        video_frame = bs * ts
        
        object_output=None
        object_mask=None
        attr_text_tokens=None
        attr_pairs_mask=None
        
        if attr_sim or insert_attr or replace_cls or phrase_fuse == 'fusion':
            attr_pairs_text, attr_pairs_label, attr_pairs_segment = attr_triple
            attr_pairs_text = attr_pairs_text.view(-1, attr_pairs_text.shape[-1])
            attr_pairs_segment = attr_pairs_segment.view(-1, attr_pairs_segment.shape[-1])
            max_text_len = attr_pairs_label.shape[-1]
            attr_pairs_label = attr_pairs_label.view(-1, attr_pairs_label.shape[-1])
            attr_sequence_output, attr_text_tokens = self.get_sequence_output(attr_pairs_text, None)
            
            if phrase_fuse == 'fusion':
                attr_pairs_mask = torch.zeros(attr_pairs_label.shape).to(attr_pairs_label.device)
                attr_pairs_mask[attr_pairs_label==0] = 0
                attr_pairs_mask[attr_pairs_label==1] = 1
            
        if object_fuse == "fusion":
            sequence_output, visual_output, object_tensors = self.get_sequence_visual_output(question_ids, None, video, video_mask, video_frame=video_frame, objects=objects, object_layer=object_layer, need_object_token=True, object_fuse=object_fuse)
            (object_output, object_label, patch_token) = object_tensors
            object_output = object_output.permute(1, 0, 2)
            object_output = object_output + self.clip.object_frame_pos_emb.unsqueeze(1).repeat((b, object_output.shape[1], 1))
            
            object_cls_tensor, id_list = obj_cls_tuple
            object_cls_tensor = object_cls_tensor.reshape((b, -1)) 
            object_output = object_output.reshape((b, -1, object_output.shape[-1]))
            
            if use_action:
                act_object_output = object_output.clone()
                act_object_output = act_object_output.reshape(b, video_frame, object_label.shape[-1], object_output.shape[-1])
                act_object_label = object_label.clone()
                act_object_label = act_object_label.reshape(b, video_frame, object_label.shape[-1])
                # (B, num, dim)
                act_output = self.get_act_output(act_object_output, act_object_label, patch_token)
            bch_obj_mask_list = []
            for bch_idx in range(b):
                obj_mask_list = []
                for cls_id in id_list[bch_idx]:
                    obj_mask_list.append((object_cls_tensor[bch_idx] != cls_id).unsqueeze(0)) 
                bch_obj_mask_list.append(torch.cat(obj_mask_list, dim=0).unsqueeze(0)) 
            bch_obj_mask_list = torch.cat(bch_obj_mask_list, dim=0) 
            object_output = object_output.unsqueeze(1).repeat((1, id_list.shape[1], 1, 1)) 
            object_cls_embedding = self.clip.object_cls_embedding.unsqueeze(0).unsqueeze(0).repeat((b, id_list.shape[1], 1, 1))
            
            obj_cls_mask = bch_obj_mask_list.new_zeros((b, id_list.shape[1], 1))
            bch_obj_mask_list = torch.cat((obj_cls_mask, bch_obj_mask_list), dim=2)
            object_output = torch.cat((object_cls_embedding, object_output), dim=2)
            
            bch_obj_mask_list = bch_obj_mask_list.reshape((b*id_list.shape[1], -1))
            object_output = object_output.reshape((bch_obj_mask_list.shape[0], bch_obj_mask_list.shape[1], -1))
            
            # input for transformer
            object_output = object_output.permute(1, 0, 2) 
            object_output = self.clip.object_fusion.transformer(object_output.half(), key_padding_mask=bch_obj_mask_list)
            object_output = object_output.permute(1, 0, 2)
            object_output = object_output[:, 0, :] 
            
            object_output = object_output.reshape((b, id_list.shape[1], -1)) 
            object_mask = torch.zeros(id_list.shape).to(id_list.device) 
            object_mask[id_list==0] = 0
            object_mask[id_list!=0] = 1
            
            object_output = self.clip.object_fusion.ln_post(object_output) @ self.clip.object_fusion.proj
            if use_action:
                if action_only:
                    object_output = act_output
                    action_mask = object_mask.new_ones((b, self.clip.action_num))
                    object_mask = action_mask
                else:
                    object_output = torch.cat((object_output, act_output), dim=1)
                    action_mask = object_mask.new_ones((b, self.clip.action_num))
                    object_mask = torch.cat((object_mask, action_mask), dim=-1)
        else:
            sequence_output, visual_output = self.get_sequence_visual_output(question_ids, None, video, video_mask, video_frame=video_frame, objects=objects, object_layer=object_layer)
        
        if replace_cls:
            sequence_output[:,:1, :] = attr_sequence_output
        fusion_output, fusion_mask = self.get_fusion_output(sequence_output, question_mask, visual_output, video_mask, object_output=object_output,object_mask=object_mask, attr_text_tokens=attr_text_tokens, attr_pairs_mask=attr_pairs_mask)
        answer_token = fusion_output[:, -1, :]
        logit_weight = torch.matmul(self.clip.fusion_proj, self.clip.token_embedding.weight.T)
        score = F.linear(self.dropout(answer_token).half(), logit_weight.T.half(), self.clip.fusion_logit_bias.half())
        if self.training:
            loss = 0.    
            ce_loss = F.cross_entropy(score, answer_ids)
            sequence_output, _ = self.get_sequence_output(align_text,None)
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, align_mask, video_mask,shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            if use_sim:
                loss += sim_loss
            loss += ce_loss
            return loss
        else:
            correct_or_not = torch.eq(score.argmax(dim=-1), answer_ids)
            
            predict_ids = score.argmax(dim=-1)

            return int(correct_or_not.sum()), correct_or_not.shape[0], predict_ids

    def get_sequence_visual_output(self, question_ids, question_mask, video, video_mask, video_frame=-1, objects=None, object_layer=[], need_object_token=False, object_fuse="attention"):
        
        bs_pair = question_ids.size(0)
        sequence_hidden, text = self.clip.encode_text(question_ids, return_hidden=True)
        text = text.float()
        text = text.view(bs_pair, -1, text.size(-1))

        sequence_output = text

        bs_pair = video_mask.size(0)
        visual_hidden, object_token = self.clip.encode_image(video, video_frame=video_frame, objects=objects, object_layer=object_layer, object_fuse=object_fuse)
        
        visual_hidden = visual_hidden.float()
        (object_token, object_mask, patch_output) = object_token
        if object_token is not None:
            object_token = object_token.float()
            
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        visual_output = visual_hidden
        
        if need_object_token:
            return sequence_output, visual_output, (object_token, object_mask, patch_output)
        else:
            return sequence_output, visual_output

    def get_fusion_output(self, sequence_output, sequence_mask, visual_output, video_mask, object_output=None,object_mask=None, attr_text_tokens=None, attr_pairs_mask=None):

        max_video_len = video_mask.shape[-1]
        bs = sequence_output.shape[0]
        # mean pooling
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_tokens = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        v_sep_token = (torch.sum(video_tokens, dim=1) / video_mask_un_sum).unsqueeze(1)
        


        answer_token = torch.zeros_like(sequence_output[:, 0:1, :]).to(visual_output.device) 
        fusion_tokens = torch.cat((visual_output, v_sep_token, sequence_output, answer_token), dim=1) 

            
            
        fusion_list = [fusion_tokens[:,:video_mask.shape[-1] + 1, :]]
        if object_output is not None:
            fusion_list.append(object_output)
            max_object_len = object_mask.shape[-1]
            max_video_len = max_video_len + max_object_len
        if attr_text_tokens is not None:
            fusion_list.append(attr_text_tokens)
            max_video_len = max_video_len + attr_pairs_mask.shape[-1]
        fusion_list.append(fusion_tokens[:,video_mask.shape[-1] + 1:, :])
        fusion_tokens = torch.cat(fusion_list, dim=1)
        

        fusion_pos_emd = self.clip.fusion_positional_embedding[:fusion_tokens.size(1), :]  
        fusion_tokens = fusion_tokens + fusion_pos_emd 
        #去除SEP的Positional Embedding
        sep_idx = torch.cat(((video_mask.sum(dim=-1)).unsqueeze(1), (max_video_len + sequence_mask.sum(dim=-1)).unsqueeze(1)), dim=-1)  # [bs, 2]
        for i in range(bs):
            fusion_tokens[i,sep_idx[i,0]] = v_sep_token[i]
            fusion_tokens[i,sep_idx[i,1]] = sequence_output[i,sep_idx[i,1]-max_video_len-1]
            
        fusion_tokens = fusion_tokens.half()

        sep_mask = torch.ones(bs, 1).to(visual_output.device)
        answer_mask = torch.ones(bs, 1).to(visual_output.device)

        fusion_mask_list = [video_mask, sep_mask]
        
        if object_output is not None:
            fusion_mask_list.append(object_mask)
        if attr_text_tokens is not None:
            fusion_mask_list.append(attr_pairs_mask)
            
        fusion_mask_list.append(sequence_mask)
        fusion_mask_list.append(answer_mask)
        
        fusion_mask = torch.cat(fusion_mask_list, dim=1)
        fusion_mask[fusion_mask==0] = float('-inf')
        fusion_mask[fusion_mask==1] = 0.
        
        fusion_mask = fusion_mask.unsqueeze(1).repeat(1,fusion_mask.shape[-1],1)
        


        fusion_tokens = fusion_tokens.permute(1, 0, 2)  # NLD -> LND
        fusion_output = self.clip.fusion_transformer(fusion_tokens, mask=fusion_mask, task='fusion')
        fusion_output = fusion_output.permute(1, 0, 2)  # LND -> NLD

        return fusion_output, fusion_mask    
    def get_act_output(self, object_token, object_label, patch_token):
        _, patch_num, dim = patch_token.shape
        patch_token = patch_token.reshape(-1, 12, patch_num, dim)
        feature_head = torch.mean(patch_token[:, :, 0, :], dim=1)
        patch_feature = patch_token[:, :, 1:, :]
        attention_mask = patch_token.new_zeros(patch_feature.shape[:-1]).float()
        patch_obj_feature = torch.cat((patch_feature, object_token), dim=2) 
        obj_mask = patch_token.new_zeros(object_token.shape[:-1]).float()
        obj_mask[object_label == True] = float('-inf')
        obj_mask[object_label == False] = 0
        attention_mask = torch.cat((attention_mask, obj_mask), dim=2)
        attention_mask = attention_mask.unsqueeze(2).repeat(1, 1, self.clip.action_num, 1)
        act_token = self.clip.object_action.repeat(patch_obj_feature.shape[0], patch_obj_feature.shape[1], 1, 1) 
        
        weight = torch.matmul(act_token.float(), patch_obj_feature.float().permute(0, 1, 3, 2))
        weight = weight + attention_mask
        weight = F.softmax(weight, -1)
        
        act_fusion = torch.matmul(weight, patch_obj_feature.float())
        
        act_fusion = act_fusion.permute(0, 2, 1, 3)
        feature_head = feature_head.unsqueeze(1).repeat(1, act_fusion.shape[1], 1).unsqueeze(2)
        action_fusion = torch.cat((feature_head, act_fusion), dim=2)
        
        bz, action_num, frame_num, dim = action_fusion.shape
        
        action_fusion = action_fusion.reshape(-1, frame_num, dim)

        action_fusion = action_fusion.permute(1, 0, 2)
        encoded_action = self.clip.object_act_fusion.transformer(action_fusion.half())
        encoded_action = encoded_action.permute(1, 0, 2)

        encoded_action = encoded_action.reshape(bz, action_num, frame_num, dim)

        action_token = encoded_action[:, :, 0, :] 
        action_token = self.clip.object_act_fusion.ln_post(action_token) @ self.clip.object_act_fusion.proj
        
        return action_token    

    def get_sequence_output(self, input_ids,attention_mask):
        
        bs_pair = input_ids.size(0)
        sequence_hidden , text = self.clip.encode_text(input_ids,attention_mask)
        sequence_hidden = sequence_hidden.float()
        text = text.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        text = text.view(bs_pair, -1, text.size(-1))
        return sequence_hidden , text
    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        
        video_out = torch.sum(visual_output, dim=1)
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()

        retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)

        return retrieve_logits, contrastive_direction

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config
