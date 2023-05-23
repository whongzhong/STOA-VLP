from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from pyparsing import col
import torch
from torch import nn
import numpy as np
from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
import random
from modules.module_clip import CLIP, convert_weights, LayerNorm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import copy
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, restore_path=None, object_count=10, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        task_config.object_count = object_count
        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16", clip_archive=task_config.clip_archive)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
            
            if not os.path.exists(restore_path):
                        
                if key.startswith('transformer'):
                    copyed_key = "clip." + key.replace("transformer", "fusion_transformer")
                    state_dict[copyed_key] = val.clone()
                if key.startswith('positional_embedding'):
                    fusion_positional_embedding = nn.Parameter(torch.empty(90 + object_count * 12, 512))
                    nn.init.normal_(fusion_positional_embedding, std=0.01)
                    fusion_positional_embedding[:77] = val[:77].clone()
                    copyed_key = "clip." + key.replace("positional_embedding", "fusion_positional_embedding")
                    state_dict[copyed_key] = fusion_positional_embedding.clone()       
                    
        
        for key in list(state_dict.keys()):
            if key.startswith('clip.visual') and not key.startswith('clip.visual.transformer'):

                new_key = key.replace('visual', 'object_fusion')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
                    
                new_key = key.replace('visual', 'object_act_fusion')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
            elif key.startswith('clip.visual.transformer.resblocks.10'):
                new_key = key.replace('clip.visual.transformer.resblocks.10', 'clip.object_fusion.transformer.resblocks.0')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
                new_key = key.replace('clip.visual.transformer.resblocks.10', 'clip.object_act_fusion.transformer.resblocks.0')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
            elif key.startswith('clip.visual.transformer.resblocks.11'):
                new_key = key.replace('clip.visual.transformer.resblocks.11', 'clip.object_fusion.transformer.resblocks.1')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
                new_key = key.replace('clip.visual.transformer.resblocks.11', 'clip.object_act_fusion.transformer.resblocks.1')
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
         
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        if task_config.do_train:
            ## ===> Initialization trick [HARD CODE]
            if model.linear_patch == "3d":
                contain_conv2 = False
                for key in state_dict.keys():
                    if key.find("visual.conv2.weight") > -1:
                        contain_conv2 = True
                        break
                if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                    cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                    kernel_size = model.clip.visual.conv2.weight.size(2)
                    conv2_size = model.clip.visual.conv2.weight.size()
                    conv2_size = list(conv2_size)

                    left_conv2_size = conv2_size.copy()
                    right_conv2_size = conv2_size.copy()
                    left_conv2_size[2] = (kernel_size - 1) // 2
                    right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                    left_zeros, right_zeros = None, None
                    if left_conv2_size[2] > 0:
                        left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                    if right_conv2_size[2] > 0:
                        right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                    cat_list = []
                    if left_zeros != None: cat_list.append(left_zeros)
                    cat_list.append(cp_weight.unsqueeze(2))
                    if right_zeros != None: cat_list.append(right_zeros)
                    cp_weight = torch.cat(cat_list, dim=2)

                    state_dict["clip.visual.conv2.weight"] = cp_weight

            if model.sim_header == 'tightTransf':
                contain_cross = False
                for key in state_dict.keys():
                    if key.find("cross.transformer") > -1:
                        contain_cross = True
                        break
                if contain_cross is False:
                    for key, val in clip_state_dict.items():
                        if key == "positional_embedding":
                            state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                            continue
                        if key.find("transformer.resblocks") == 0:
                            num_layer = int(key.split(".")[2])

                            # cut from beginning
                            if num_layer < task_config.cross_num_hidden_layers:
                                state_dict["cross."+key] = val.clone()
                                continue

            if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
                contain_frame_position = False
                for key in state_dict.keys():
                    if key.find("frame_position_embeddings") > -1:
                        contain_frame_position = True
                        break
                if contain_frame_position is False:
                    for key, val in clip_state_dict.items():
                        if key == "positional_embedding":
                            state_dict["frame_position_embeddings.weight"] = val.clone()
                            continue
                        if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                            num_layer = int(key.split(".")[2])
                            # cut from beginning
                            if num_layer < task_config.cross_num_hidden_layers:
                                state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                                continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.mask_ratio = self.task_config.mask_ratio
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
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
            vision_patch_size = 16
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = 16
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

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
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
        self.simloss = nn.MSELoss(reduction='mean')
        self.apply(self.init_weights)


    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None, fusion_labels=None, objects = None, object_layer=[], attr_triple = None, attr_sim=False, insert_attr=False, replace_cls=False, phrase_fuse="None", object_fuse="attention", obj_cls_tuple=None, pos_tag=None, tokenizer=None, use_action = False, use_sim_loss = True, action_only = False):
        max_video_len = video_mask.shape[-1]
        ori_max_video_len = max_video_len
        max_text_len = attention_mask.shape[-1]
        fusion_len = fusion_labels.shape[-1]
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        video_mask_ori = video_mask
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        
        if attr_sim or insert_attr or replace_cls or phrase_fuse == 'fusion':
            attr_pairs_text, attr_pairs_label, attr_pairs_segment = attr_triple
            attr_pairs_text = attr_pairs_text.view(-1, attr_pairs_text.shape[-1])
            attr_pairs_segment = attr_pairs_segment.view(-1, attr_pairs_segment.shape[-1])
            attr_pairs_label = attr_pairs_label.view(-1, attr_pairs_label.shape[-1])
            attr_sequence_output, attr_text_tokens = self.get_sequence_output(attr_pairs_text,attr_pairs_segment,None,shaped=True)
            attr_sequence_output = attr_sequence_output.detach()
            attr_text_tokens = attr_text_tokens.detach()
            
        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
       
        mlm_ids = input_ids.clone()
        mlm_ids[fusion_labels[:,max_video_len+1:]==-1] = 0
        
        if object_fuse == "fusion":
            if action_only: 
                sequence_output, visual_output, text_tokens, object_tensors = self.get_sequence_visual_output(mlm_ids, token_type_ids, None ,video, video_mask, shaped=True, video_frame=video_frame, objects=objects, object_layer=object_layer, need_object_token=True, object_fuse="attention")
            else:
                sequence_output, visual_output, text_tokens, object_tensors = self.get_sequence_visual_output(mlm_ids, token_type_ids, None ,video, video_mask, shaped=True, video_frame=video_frame, objects=objects, object_layer=object_layer, need_object_token=True, object_fuse=object_fuse)
                
            (object_output, object_label, patch_token) = object_tensors
            object_output = object_output.permute(1, 0, 2)
            object_output = object_output + self.clip.object_frame_pos_emb.unsqueeze(1).repeat((b, object_output.shape[1], 1))
            object_output = object_output.reshape((b, -1, object_output.shape[-1]))
            
            if use_action:
                act_object_output = object_output.clone()
                act_object_output = act_object_output.reshape(b, video_frame, object_label.shape[-1], object_output.shape[-1])
                act_object_label = object_label.clone()
                act_object_label = act_object_label.reshape(b, video_frame, object_label.shape[-1])
               
                act_output = self.get_act_output(act_object_output, act_object_label, patch_token)
            
            
            object_cls_tensor, id_list, obj_vocab_idx, prd_id_list = obj_cls_tuple
            object_cls_tensor = object_cls_tensor.reshape((b, -1)) 
            
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
            
            object_output = object_output.permute(1, 0, 2) 
            object_output = self.clip.object_fusion.transformer(object_output.half(), key_padding_mask=bch_obj_mask_list)
            object_output = object_output.permute(1, 0, 2) 
            object_output = object_output[:, 0, :] 
            
            object_output = object_output.reshape((b, id_list.shape[1], -1)) 
            object_mask = fusion_labels.new_zeros(id_list.shape) 
            
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
                
            #### bipairy matching
            object_output_biparties = object_output.clone()
                
            fusion_labels = torch.cat((fusion_labels[:, :max_video_len+1], object_mask, fusion_labels[:, max_video_len+1:]), dim=1)
            max_object_len = object_mask.shape[-1]
            
            obj_pos_emd = self.clip.fusion_positional_embedding[max_video_len + 1:max_video_len + 1 + max_object_len, :]
            max_video_len = max_video_len + max_object_len
            fusion_len = fusion_len + max_object_len
        else:
            sequence_output, visual_output, text_tokens = self.get_sequence_visual_output(mlm_ids, token_type_ids, None ,video, video_mask, shaped=True, video_frame=video_frame, objects=objects, object_layer=object_layer)
        
        if phrase_fuse == 'fusion':
            attr_pairs_mask = fusion_labels.new_zeros(attr_pairs_label.shape)
            attr_pairs_mask[attr_pairs_label==0] = 0
            attr_pairs_mask[attr_pairs_label==1] = 1
            fusion_labels = torch.cat((fusion_labels[:, :max_video_len+1], attr_pairs_mask, fusion_labels[:, max_video_len+1:]), dim=1)
            
            attr_pos_emd = self.clip.fusion_positional_embedding[max_video_len+1:max_video_len+max_text_len+1, :]
            max_video_len = max_video_len + max_text_len
            fusion_len = fusion_len + max_text_len
       
            
            
            
            
        sep_idx = torch.cat((video_mask_ori.sum(dim=-1),max_video_len+attention_mask.sum(dim=-1)),dim=-1)
        
        # MLM
        sequence_output = sequence_output.detach()
        visual_output = visual_output.detach()
        text_tokens = text_tokens.detach()

        # mean pooling
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_tokens = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        v_sep_token = (torch.sum(video_tokens, dim=1) / video_mask_un_sum).unsqueeze(1)
        
        video_tokens = torch.cat((visual_output,v_sep_token),dim=-2)
        

        if object_fuse == 'fusion' or phrase_fuse == 'fusion':
            
            cat_tensor = [video_tokens]
            if object_fuse == 'fusion':
                cat_tensor.append(object_output)
            if phrase_fuse == 'fusion':
                cat_tensor.append(attr_text_tokens)
            
            cat_tensor.append(text_tokens)
            fusion_input = torch.cat(cat_tensor,dim=-2)
        else:
            fusion_input = torch.cat((video_tokens,text_tokens),dim=-2)
            
        pos_emd = self.clip.fusion_positional_embedding[:fusion_input.size(1), :]
        
        fusion_input[fusion_labels==-1] = torch.zeros(512).to(fusion_labels.device)
        fusion_mask = torch.zeros(fusion_labels.shape).to(fusion_labels.device)

        fusion_mask[fusion_labels==0] = float('-inf')
        fusion_mask = fusion_mask.unsqueeze(1).repeat(1,fusion_len,1)
        
        
        fusion_input = fusion_input + pos_emd
        #去除SEP的Positional Embedding
        for i in range(b):
            fusion_input[i,sep_idx[i,0]] = video_tokens[i,sep_idx[i,0]]
            fusion_input[i,sep_idx[i,1]] = text_tokens[i,sep_idx[i,1]-max_video_len-1]

        fusion_input = fusion_input.permute(1, 0, 2)  # NLD -> LND
        fusion_output = self.clip.fusion_transformer(fusion_input.half(),mask=fusion_mask.half(),task='fusion')
        fusion_output = fusion_output.permute(1, 0, 2)  # LND -> NLD

        
        predicted_output = fusion_output[fusion_labels==-1]  
        logit_weight = torch.matmul(self.clip.fusion_proj, self.clip.token_embedding.weight.T)
        scores = F.linear(self.dropout(predicted_output).float(), logit_weight.T.float(), self.clip.fusion_logit_bias.float())
        target_ids = input_ids[fusion_labels[:,max_video_len+1:]==-1]
        mlm_loss = self.CrossEntropyLoss(scores,target_ids)
        mlm_loss = mlm_loss.mean()

        logpt = F.log_softmax(scores, dim=-1)
        predicted_ids = logpt.argmax(-1)
        mlm_acc = predicted_ids.eq(target_ids).sum().view(-1) / target_ids.shape[0]
        
        mlm_acc = allgather(mlm_acc, self.task_config)
        torch.distributed.barrier()

        if self.task_config.rank == 0:
            print('MLM ACC:',mlm_acc.mean().item())

        ### obj_flow matching
        matching_loss = 0
        if object_fuse == "fusion":
            sequence_output, text_tokens = self.get_sequence_output(input_ids,token_type_ids,None,shaped=True)
            text_tokens = text_tokens.detach()
            
            verb_pos_tag, noun_pos_tag = pos_tag
            verb_tokens = text_tokens.clone()
            noun_tokens = text_tokens.clone()
            text_sep = attention_mask.sum(dim=-1)
            for bch in range(b):
                verb_pos_tag[bch][text_sep[bch] - 1] = 1 
                noun_pos_tag[bch][text_sep[bch] - 1] = 1
            
            if use_action:
                action_biparties = object_output_biparties[:, -self.clip.action_num:, ]
                obj_biparties = object_output_biparties[:, :-self.clip.action_num, ]
                
                
            else:
                obj_biparties = object_output_biparties
            verb_tokens[verb_pos_tag == 0] = torch.zeros(512).to(verb_tokens.device)
            noun_tokens[noun_pos_tag == 0] = torch.zeros(512).to(verb_tokens.device)
            obj_biparties[id_list==0] = torch.zeros(512).half().to(verb_tokens.device)
            
            verb_logit_weight = verb_tokens
            noun_logit_weight = noun_tokens
            
            obj_token = []
            text_token = []
            for bch in range(b):
                if not action_only:
                    obj_row = obj_biparties[bch][id_list[bch] != 0]
                    noun_row = noun_logit_weight[bch][noun_pos_tag[bch] != 0]
                    if obj_row.shape[0] == 0 or noun_row.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        result_mat = torch.matmul(obj_row.float(), noun_row.T)
                        result_mat = result_mat.softmax(dim=-1)
                        row_ind, col_ind = linear_sum_assignment(result_mat.detach().cpu().numpy(), maximize=True)
                    obj_token.append(obj_row[row_ind])
                    text_token.append(noun_row[col_ind])
                
                if use_action:
                    act_row = action_biparties[bch]
                    verb_row = verb_logit_weight[bch][verb_pos_tag[bch] != 0]
                    if verb_row.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        verb_result_mat = torch.matmul(act_row.float(), verb_row.T)
                        verb_result_mat = verb_result_mat.softmax(dim=-1)
                        row_ind, col_ind = linear_sum_assignment(verb_result_mat.detach().cpu().numpy(), maximize=True)
                    obj_token.append(act_row[row_ind])
                    text_token.append(verb_row[col_ind])
                    
            if len(obj_token) > 0:
                obj_token = torch.cat(obj_token, dim=0)
                text_token = torch.cat(text_token, dim=0)
            
                matching_loss = self.simloss(obj_token.float(), text_token)    
          
        if object_fuse == "fusionuseless":
            sequence_output, text_tokens = self.get_sequence_output(input_ids,token_type_ids,None,shaped=True)
            text_tokens = text_tokens.detach()
            
            if object_fuse == 'fusion' or phrase_fuse == 'fusion':
                
                cat_tensor = [video_tokens]
                if object_fuse == 'fusion':
                    cat_tensor.append(object_output)
                if phrase_fuse == 'fusion':
                    cat_tensor.append(attr_text_tokens)
                
                cat_tensor.append(text_tokens)
                fusion_input = torch.cat(cat_tensor,dim=-2)
            else:
                fusion_input = torch.cat((video_tokens,text_tokens),dim=-2)
                
            pos_emd = self.clip.fusion_positional_embedding[:fusion_input.size(1), :]
            
            fusion_input[fusion_labels==-1] = torch.zeros(512).to(fusion_labels.device)
            fusion_mask = torch.zeros(fusion_labels.shape).to(fusion_labels.device)

            fusion_mask[fusion_labels==0] = float('-inf')
            fusion_mask = fusion_mask.unsqueeze(1).repeat(1,fusion_len,1)
            
            
            fusion_input = fusion_input + pos_emd
            #去除SEP的Positional Embedding
            for i in range(b):
                fusion_input[i,sep_idx[i,0]] = video_tokens[i,sep_idx[i,0]]
                fusion_input[i,sep_idx[i,1]] = text_tokens[i,sep_idx[i,1]-max_video_len-1]

            fusion_input = fusion_input.permute(1, 0, 2)  # NLD -> LND
            fusion_output = self.clip.fusion_transformer(fusion_input.half(),mask=fusion_mask.half(),task='fusion')
            fusion_output = fusion_output.permute(1, 0, 2).float()  # LND -> NLD
            
            
            obj_flow_output = fusion_output[:, video_tokens.shape[1]:video_tokens.shape[1] + id_list.shape[1], :]
            obj_flow_output = obj_flow_output[prd_id_list!=0]
            
            logit_weight = torch.matmul(self.clip.fusion_proj, self.clip.token_embedding.weight.T)
            logit_weight = logit_weight.detach()
            obj_predict_scores = F.linear(self.dropout(obj_flow_output).half(), logit_weight.T.half(), self.clip.fusion_logit_bias.half())
            target_obj_idx = obj_vocab_idx[prd_id_list!=0]
            
            obj_predict_loss = self.CrossEntropyLoss(obj_predict_scores,target_obj_idx)
            obj_predict_loss = obj_predict_loss.mean()

            logpt = F.log_softmax(obj_predict_scores, dim=-1)
            obj_predicted_ids = logpt.argmax(-1)
        
            obj_predict_acc = obj_predicted_ids.eq(target_obj_idx).sum().view(-1) / target_obj_idx.shape[0]
            
            obj_predict_acc = allgather(obj_predict_acc, self.task_config)
            torch.distributed.barrier()

            if self.task_config.rank == 0:
                print('obj predict ACC:',obj_predict_acc.mean().item())
                
        # PLM
        
        fusion_mask = torch.zeros(fusion_labels.shape).to(fusion_labels.device)
        fusion_mask[fusion_labels==0] = float('-inf')
        fusion_mask[fusion_labels==1] = 0
        fusion_mask = fusion_mask.unsqueeze(1).repeat(1,fusion_len,1)
        fusion_mask[:,:max_video_len+1,max_video_len+1:] = float('-inf')
        cap_mask = torch.triu(torch.ones(fusion_mask[:,max_video_len+1:,max_video_len+1:].shape),diagonal=1).to(fusion_mask.device)
        cap_mask[cap_mask==1] = float('-inf')
        fusion_mask[:,max_video_len+1:,max_video_len+1:] = cap_mask


        text_mask = attention_mask.view(-1, attention_mask.shape[-1])




        text_input = torch.zeros((b,fusion_len-max_video_len-1,512)).to(visual_output.device)

        # mean pooling
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_tokens_plm = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        v_sep_token = (torch.sum(video_tokens_plm, dim=1) / video_mask_un_sum).unsqueeze(1)
        
        video_ori_tokens = torch.cat((visual_output,v_sep_token),dim=-2)

        video_pos_emd = self.clip.fusion_positional_embedding[:video_ori_tokens.size(1), :]
        
        video_tokens_plm = video_ori_tokens + video_pos_emd
        
        

        #去除SEP的Positional Embedding
        for i in range(b):
            video_tokens_plm[i,sep_idx[i,0]] = video_ori_tokens[i,sep_idx[i,0]]
        
        
        text_pos_emd = self.clip.fusion_positional_embedding[max_video_len+1:max_video_len+max_text_len+1, :]
        text_input = text_input + text_pos_emd

        video_tokens_plm = video_tokens_plm.detach()
        scores = []
        for i in range(input_ids.shape[1]-1):
            sequence_output, text_tokens = self.get_sequence_output(input_ids,token_type_ids,None,shaped=True)

            text_tokens = text_tokens.detach()
            text_input[:,:i+1,:] = text_tokens[:,:i+1,:] + text_pos_emd[:i+1,:]
            if object_fuse == 'fusion' or phrase_fuse == 'fusion':
                
                cat_tensor = [video_tokens]
                if object_fuse == 'fusion':
                    cat_tensor.append(object_output + obj_pos_emd)
                if phrase_fuse == 'fusion':
                    cat_tensor.append(attr_text_tokens + attr_pos_emd)
                
                cat_tensor.append(text_input)
                fusion_input = torch.cat(cat_tensor,dim=-2)
            else:
                fusion_input = torch.cat((video_tokens_plm,text_input),dim=-2)
                
            fusion_input = fusion_input.permute(1, 0, 2)  # NLD -> LND
            fusion_output = self.clip.fusion_transformer(fusion_input.half(),mask=fusion_mask.half(),task='fusion')
            fusion_output = fusion_output.permute(1, 0, 2).float()  # LND -> NLD
            predicted_next = fusion_output[:,max_video_len+2+i,:] 
            logit_weight = torch.matmul(self.clip.fusion_proj, self.clip.token_embedding.weight.T)
            score = F.linear(self.dropout(predicted_next).float(), logit_weight.T.float(), self.clip.fusion_logit_bias.float())
            
            scores.append(score.unsqueeze(1))

        scores = torch.cat(scores,dim=1)
        scores_plm = []
        cap_targets = []
        fusion_labels_plm = fusion_labels.clone()
        fusion_labels_plm[fusion_labels==-1] = 1
        for i in range(fusion_labels_plm.shape[0]):
            text_len = fusion_labels_plm[i][max_video_len+2:].sum().item()
            start_idx = random.randint(1,text_len)
            cap_targets.append(input_ids[i][start_idx:text_len+1])
            scores_plm.append(scores[i][start_idx-1:text_len])

        cap_targets = torch.cat(cap_targets,dim=-1)
        scores_plm = torch.cat(scores_plm,dim=-2)

        plm_loss = self.CrossEntropyLoss(scores_plm,cap_targets)
        plm_loss = plm_loss.mean()

        logpt = F.log_softmax(scores_plm, dim=-1)
        predicted_ids = logpt.argmax(-1)
        plm_acc = predicted_ids.eq(cap_targets).sum().view(-1) / cap_targets.shape[0]
        
        plm_acc = allgather(plm_acc, self.task_config)
        torch.distributed.barrier()

        if self.task_config.rank == 0:
            print('PLM ACC:',plm_acc.mean().item())

        
        # VTM
        
        sequence_output, text_tokens = self.get_sequence_output(input_ids, token_type_ids, None, shaped=True)
        sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,shaped=True, loose_type=self.loose_type)
        sim_loss = 0
        if use_sim_loss:
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
        
        attention_mask = attention_mask.float()
        attention_mask[attention_mask==0] = float('-inf')
        attention_mask[attention_mask==1] = 0
        video_mask = torch.cat((video_mask[:,0:1],video_mask),dim=-1).float()
        video_mask[video_mask==0] = float('-inf')
        video_mask[video_mask==1] = 0

        video_mask = video_mask.unsqueeze(1)

        if object_fuse == 'fusion':
            object_mask = object_mask.float()
            object_mask[object_mask==0] = float('-inf')
            object_mask[object_mask==1] = 0
            object_mask = object_mask.unsqueeze(1)
        if phrase_fuse == 'fusion':
            attr_pairs_mask = attr_pairs_mask.float()
            attr_pairs_mask[attr_pairs_label==0] = float('-inf')
            attr_pairs_mask[attr_pairs_label==1] = 0
            attr_pairs_mask = attr_pairs_mask.unsqueeze(1)

        video_tokens = video_tokens.detach()
        text_tokens = text_tokens.detach()
        attention_mask = attention_mask.detach()
        video_mask = video_mask.detach()
        if object_fuse == 'fusion':
            all_object_output = allgather(object_output, self.task_config)
            all_object_mask = allgather(object_mask, self.task_config)
        if phrase_fuse == 'fusion':
            all_attr_text_tokens = allgather(attr_text_tokens, self.task_config)
            all_attr_pairs_mask = allgather(attr_pairs_mask, self.task_config)
        all_video_tokens = allgather(video_tokens,self.task_config)
        all_text_tokens = allgather(text_tokens,self.task_config)
        all_text_mask = allgather(attention_mask,self.task_config)
        all_video_mask = allgather(video_mask,self.task_config)
            

        torch.distributed.barrier()
        

        diag_mask = torch.ones(sim_matrix.shape[0])
        diag_mask = torch.diag(diag_mask)

        t2v_scores = F.softmax(sim_matrix, dim=-1)
        v2t_scores = F.softmax(sim_matrix.T, dim=-1)
        t2v_scores[diag_mask==1] = 0.
        v2t_scores[diag_mask==1] = 0.
        
        t2v_hard_idx = torch.multinomial(t2v_scores,1,replacement=False)
        v2t_hard_idx = torch.multinomial(v2t_scores,1,replacement=False)

        t2v_hard_neg = all_video_tokens[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
        v2t_hard_neg = all_text_tokens[v2t_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
        t2v_hard_mask = all_video_mask[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
        v2t_hard_mask = all_text_mask[v2t_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
        
        
        fusion_cat_list = []  
        fusion_mask_cat_list = []      
        if object_fuse == 'fusion':
            t2v_hard_object_output_neg = all_object_output[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            t2v_hard_object_mask_neg = all_object_mask[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            fusion_cat_list.append(t2v_hard_object_output_neg)
            fusion_mask_cat_list.append(t2v_hard_object_mask_neg)
        if phrase_fuse == 'fusion':
            t2v_hard_attr_text_tokens_neg = all_attr_text_tokens[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            t2v_hard_attr_pairs_mask_neg = all_attr_pairs_mask[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            fusion_cat_list.append(t2v_hard_attr_text_tokens_neg)
            fusion_mask_cat_list.append(t2v_hard_attr_pairs_mask_neg)

        t2v_hard_neg_list = [t2v_hard_neg]
        t2v_hard_neg_list.extend(fusion_cat_list)
        t2v_hard_neg_list.append(text_tokens.unsqueeze(1).repeat(1,t2v_hard_neg.shape[1],1,1))
        t2v_hard_neg = torch.cat(t2v_hard_neg_list, dim=-2)
        
        v2t_hard_neg_list = [video_tokens.unsqueeze(1).repeat(1,v2t_hard_neg.shape[1],1,1)]
        if object_fuse == 'fusion':
            v2t_hard_neg_list.append(object_output.unsqueeze(1).repeat(1,v2t_hard_neg.shape[1],1,1))
        if phrase_fuse == 'fusion':
            v2t_hard_neg_list.append(attr_text_tokens.unsqueeze(1).repeat(1,v2t_hard_neg.shape[1],1,1))
        v2t_hard_neg_list.append(v2t_hard_neg)
        v2t_hard_neg = torch.cat(v2t_hard_neg_list, dim=-2)
        
        t2v_hard_mask_list = [t2v_hard_mask]
        t2v_hard_mask_list.extend(fusion_mask_cat_list)
        t2v_hard_mask_list.append(attention_mask.unsqueeze(1).repeat(1,t2v_hard_mask.shape[1],1,1))
        t2v_hard_mask = torch.cat(t2v_hard_mask_list, dim=-1)

        t2v_hard_mask = t2v_hard_mask.repeat(1,1,t2v_hard_mask.shape[-1],1)
        
        
        v2t_hard_mask_list = [video_mask.unsqueeze(1).repeat(1,v2t_hard_mask.shape[1],1,1)]
        if object_fuse == 'fusion':
            v2t_hard_mask_list.append(object_mask.unsqueeze(1).repeat(1,v2t_hard_mask.shape[1],1,1))
        if phrase_fuse == 'fusion':
            v2t_hard_mask_list.append(attr_pairs_mask.unsqueeze(1).repeat(1,v2t_hard_mask.shape[1],1,1))
        v2t_hard_mask_list.append(v2t_hard_mask)
        v2t_hard_mask = torch.cat(v2t_hard_mask_list, dim=-1)
        
        v2t_hard_mask = v2t_hard_mask.repeat(1,1,v2t_hard_mask.shape[-1],1)
        
        if object_fuse == 'fusion' or phrase_fuse == 'fusion':
                
                cat_tensor = [video_tokens]
                cat_mask_tensor = [video_mask]
                if object_fuse == 'fusion':
                    cat_tensor.append(object_output)
                    cat_mask_tensor.append(object_mask)
                if phrase_fuse == 'fusion':
                    cat_tensor.append(attr_text_tokens)
                    cat_mask_tensor.append(attr_pairs_mask)
                
                cat_tensor.append(text_tokens)
                cat_mask_tensor.append(attention_mask)
                pos_input = torch.cat(cat_tensor,dim=-2).unsqueeze(1)
                pos_mask = torch.cat(cat_mask_tensor, dim=-1).unsqueeze(1)
        else:
            pos_input = torch.cat((video_tokens,text_tokens),dim=-2).unsqueeze(1)
            pos_mask = torch.cat((video_mask,attention_mask),dim=-1).unsqueeze(1)
        pos_mask = pos_mask.repeat(1,1,pos_mask.shape[-1],1)
        
        vtm_input = torch.cat((pos_input,t2v_hard_neg,v2t_hard_neg),dim=1)

        vtm_mask = torch.cat((pos_mask,t2v_hard_mask,v2t_hard_mask),dim=1)

        vtm_input = vtm_input.view(-1,vtm_input.shape[-2],vtm_input.shape[-1])
        vtm_mask = vtm_mask.view(-1,vtm_mask.shape[-2],vtm_mask.shape[-1])
        fusion_input = vtm_input.clone()
        fusion_labels = vtm_mask.clone()
        fusion_labels[fusion_labels==0.] = 1
        fusion_labels[fusion_labels==float('-inf')] = 0
        fusion_labels = fusion_labels.long()
        sep_idx = torch.cat((fusion_labels[:,0:1,:ori_max_video_len].sum(dim=-1),max_video_len+fusion_labels[:,0:1,max_video_len+1:].sum(dim=-1)),dim=-1)

        targets = torch.cat((torch.ones(pos_input.shape[0],pos_input.shape[1]),torch.zeros(t2v_hard_neg.shape[0],t2v_hard_neg.shape[1]),torch.zeros(v2t_hard_neg.shape[0],v2t_hard_neg.shape[1])),dim=1).long().to(pos_input.device)
        targets = targets.view(-1)
        
        vtm_input = vtm_input + pos_emd
        #去除SEP的Positional Embedding
        for i in range(vtm_input.shape[0]):
            vtm_input[i,sep_idx[i,0]] = fusion_input[i,sep_idx[i,0]]
            vtm_input[i,sep_idx[i,1]] = fusion_input[i,sep_idx[i,1]]
            
        vtm_input = vtm_input.permute(1, 0, 2)  # NLD -> LND
        vtm_output = self.clip.fusion_transformer(vtm_input.half(),mask=vtm_mask.half(),task='fusion')
        vtm_output = vtm_output.permute(1, 0, 2) # LND -> NLD
        vtm_v_sep = vtm_output[:,ori_max_video_len,:]
        vtm_t_sep = torch.cat([vtm_output[i][vtm_mask[i][0]==0][-1:] for i in range(vtm_output.shape[0])],dim=0)

        match_scores = self.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ self.clip.fusion_match_matrix.half() @ self.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
        unmatch_scores = self.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ self.clip.fusion_unmatch_matrix.half() @ self.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))

        diag_mask = torch.ones(match_scores.shape[0])
        diag_mask = torch.diag(diag_mask)

        match = match_scores[diag_mask==1].view(-1,1)
        unmatch = unmatch_scores[diag_mask==1].view(-1,1)
        vtm_predicted = torch.cat((unmatch,match),dim=-1)
        vtm_loss = self.CrossEntropyLoss(vtm_predicted,targets)
        
        vtm_loss = vtm_loss.mean()

        return vtm_loss,mlm_loss,plm_loss, matching_loss, sim_loss

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
        
        

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden , text = self.clip.encode_text(input_ids,attention_mask)
        sequence_hidden = sequence_hidden.float()
        text = text.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        text = text.view(bs_pair, -1, text.size(-1))
        return sequence_hidden , text

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, objects=None, object_layer=[], need_object_token=False, object_fuse="attention"):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden, object_token = self.clip.encode_image(video, video_frame=video_frame, objects=objects, object_layer=object_layer, object_fuse=object_fuse)
        visual_hidden = visual_hidden.float()
        (object_token, object_mask, patch_output) = object_token
        if object_token is not None:
            object_token = object_token.float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        if need_object_token:
            return visual_hidden, (object_token, object_mask, patch_output)
        else:
            return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1, objects=None, object_layer=[],need_object_token=False, object_fuse="attention"):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output,text = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        if need_object_token:
            visual_output, object_tokens = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame, objects=objects, object_layer=object_layer, need_object_token=True, object_fuse=object_fuse)
            return sequence_output, visual_output, text, object_tokens
        else:
            visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame, objects=objects, object_layer=object_layer, need_object_token=False)
            return sequence_output, visual_output, text

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
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
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
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction