"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from requests import patch
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
import random
import math
from .utils_object_tractor import ObjectEncoder
torch.set_printoptions(profile="full")


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# =============================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attn_mask=None, key_padding_mask = None):
        attn_mask_ = attn_mask if attn_mask is not None else None
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__') and attn_mask_ is None:
            attn_mask_ = self.attn_mask(x.size(0))
        
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        if attn_mask_ is None:
            # can be apply to check attention weights
            ret = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_, key_padding_mask = key_padding_mask)
            return ret[0]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple, key_padding_mask=None):
        x, video_frame, attn_mask = x_tuple
        x = x + self.attention(self.ln_1(x), attn_mask, key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, task="text", object_num=10, insert_layer=0, patch_size=14):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resolution = 8
        self.num_tokens = 65
        self.num_frame = 12
        scale = width ** -0.5
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        if task == 'vision':
            self.object_encoder = ObjectEncoder(width, (224, 224), object_num, self.num_frame, patch_size=patch_size)


    def forward(self, x: torch.Tensor, video_frame=-1, mask=None, task=None, objects=None, object_layer = [], object_fuse='attention', key_padding_mask = None):
        if mask is not None:
            bz, length, dim = mask.shape
            mask = mask.unsqueeze(1).expand(bz, self.heads, length, dim)
            mask = mask.reshape(-1, length, dim)
            mask = mask.contiguous()
        if task == 'vision':
            object_tokens = None
            fusion_flag = False
            attn_mask = None
            object_mask = None
            pathch_output = None
            patch_num, frame_num, dim = x.shape
            for i in range(len(self.resblocks)):
                if objects is not None and i in object_layer:
                    if i == 11:
                        x = x.permute(1, 0, 2)
                        x = x.reshape(-1, patch_num, dim)
                        x = x.permute(1, 0, 2)
                        fusion_flag = True
                    object_tokens, object_mask = self.object_encoder(x, objects) # [H*W+1+10, B * T, d]
                    if object_fuse == 'attention':
                        x_mask = torch.zeros((x.shape[1], x.shape[0]), dtype=torch.bool).to(object_tokens.device)
                        attn_mask = torch.cat([x_mask, object_mask], dim=1)
                        x = torch.cat([x, object_tokens], dim = 0)
                if i == 10 or (fusion_flag and i == 11):
                    x = x.permute(1, 0, 2)
                    if i == 10:
                        pathch_output = x.detach()
                    if objects is not None and i in object_layer and object_fuse == 'attention':
                        x = x.reshape(-1, (patch_num+object_tokens.shape[0]) * 12, dim)
                        attn_mask = attn_mask.reshape(-1, (patch_num+object_tokens.shape[0]) * 12)
                    else:
                        x = x.reshape(-1, patch_num*12, dim)
                    x = x.permute(1, 0, 2)
                (x, video_frame, mask) = self.resblocks[i]((x, video_frame, mask), key_padding_mask=attn_mask)
                if objects is not None and i in object_layer and object_fuse == 'attention':
                    attn_mask = None
                    if i == 10 or i == 11:
                        x = x.permute(1, 0, 2)
                        x = x.reshape(-1, patch_num+object_tokens.shape[0], dim)
                        x = x[:,:patch_num,:].contiguous()
                        x = x.reshape(-1, patch_num*12, dim)
                        x = x.permute(1, 0, 2)
                    else:
                        x = x[:patch_num,].contiguous() # [H*W+1+10, B * T, d]
              

            x = x.permute(1, 0, 2)
            x = x.reshape(-1, patch_num, dim)
            x = x.permute(1, 0, 2)
            return x, (object_tokens, object_mask, pathch_output)

        elif task == 'fusion':
             for i in range(0,6):
                 (x, video_frame, mask) = self.resblocks[i]((x, video_frame, mask))
             return x
             
        if key_padding_mask is not None:
            for i in range(0,1):
                (x, video_frame, mask) = self.resblocks[i]((x, video_frame, mask), key_padding_mask=key_padding_mask)
            return x
        else:
            return self.resblocks((x, video_frame, mask))[0]


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d', object_num=10):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, object_num=object_num, task="vision")

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor,  video_frame=-1, flag=None, objects=None, object_layer= [], object_fuse='attention'):
        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d) 
            x_3d = x_3d.permute(0, 2, 1, 3, 4) 
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() 
        else:
            x = self.conv1(x)

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, object_tokens = self.transformer(x, video_frame=video_frame, task='vision', objects=objects, object_layer=object_layer, object_fuse=object_fuse)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x, object_tokens


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 object_num: int = 10,
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                object_num=object_num
            )
            self.object_fusion = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=2,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                object_num=object_num
            )
            self.object_act_fusion = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=2,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                object_num=object_num
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.fusion_transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.fusion_positional_embedding = nn.Parameter(torch.empty(90 + object_num * 12, transformer_width))
        self.object_frame_pos_emb = nn.Parameter(torch.empty(12, vision_width))
        self.object_cls_embedding = nn.Parameter(512 ** -0.5 * torch.randn(1, vision_width))
        self.object_fusion_positional_embedding = nn.Parameter(torch.empty(object_num * 12, transformer_width))
        
        self.object_text_fusion_positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.action_num = 4
        self.object_action = nn.Parameter(vision_width ** -0.5 * torch.randn(1, 1, self.action_num, vision_width))
        
        
        self.object_alignment_proj = nn.Parameter(512 ** -0.5 * torch.randn(512, 512))
        self.object_logit_bias = nn.Parameter(torch.zeros((32,)), requires_grad=True)
        
        self.fusion_proj = nn.Parameter(512 ** -0.5 * torch.randn(512, 512))
        self.fusion_logit_bias = nn.Parameter(torch.zeros((self.token_embedding.weight.shape[0],)), requires_grad=True)
        self.fusion_match_matrix = nn.Parameter(512 ** -0.5*torch.randn(512,512))
        self.fusion_unmatch_matrix = nn.Parameter(512 ** -0.5*torch.randn(512,512))
        self.ln_final = LayerNorm(transformer_width)
        self.lm_head = nn.Linear(transformer_width, embed_dim)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.fusion_positional_embedding, std=0.01)
        nn.init.normal_(self.object_fusion_positional_embedding, std=0.01)
        nn.init.normal_(self.object_text_fusion_positional_embedding, std=0.01)
        nn.init.normal_(self.object_frame_pos_emb, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/16", clip_archive=None):
        model_path = os.path.join(clip_archive, "ViT-B-16.pt")
        if pretrained_clip_name == "ViT-B/16" and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1, flag=None, objects=None, object_layer = [], object_fuse="attention"):
        hidden, object_tokens = self.visual(image.type(self.dtype),video_frame=video_frame, flag=flag, objects=objects, object_layer=object_layer, object_fuse=object_fuse)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        x = hidden[:, 0, :]

        if return_hidden:
            return x, hidden, object_tokens
        return x, object_tokens

    def encode_text(self, text, mask=None, return_hidden=False, pretext=0):
        x = self.token_embedding(text).type(self.dtype)
        if pretext > 0:
            if mask is not None:
                text_emd = self.positional_embedding[:x.size(1) - pretext, :].type(self.dtype)
                attr_emd = self.positional_embedding[x.size(1) - pretext:x.size(1), :].type(self.dtype)
                pos_emd = torch.cat((attr_emd, text_emd), dim=-2)
            else:
                attr_emd = self.positional_embedding[pretext:x.size(1) + pretext, :].type(self.dtype)
                pos_emd = attr_emd
        else:
            pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x,mask=mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        return x , hidden
    

    def forward(self, image, text, flag=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
