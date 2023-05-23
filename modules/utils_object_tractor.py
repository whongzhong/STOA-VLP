from torch import nn
import torch
from torchvision.ops import roi_align
import json
import itertools
from tqdm import tqdm
def get_attr_dict():
    attr_dict = {}
    attr_dict['color'] = ["white", "black", "blue", "green", "red", "brown", "yellow", "gray", "orange", "dark", "pink", "tan", "purple", "blond", "light blue", "bright", "blond", "light brown", "light blue", "dark brown", "black and white", "dark blue", "golden", "light colored", "dark colored", "lime green", "cream colored", "rainbow colored", "golden brown", "browned"]
    
    reverse_dict = {}
    for key, value in attr_dict.items():
        for attr in value:
            reverse_dict[attr] = key
            
    return reverse_dict
    
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
     
    Parameters
    --------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner


    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

from math import ceil

def get_attr_sent(attrs, thres_hold=10):
    reverse_attr_class_dict = get_attr_dict()
    attrs = list(itertools.chain(*attrs))
    occurance_counting = {}
    object_top_conf = {}
    object_top_size = {}
    for sample in attrs:
        object_class = sample['class']
        object_top_conf[object_class] = max(object_top_conf.get(object_class, 0), round(sample['class_conf'] * 10))
        object_top_size[object_class] = max(object_top_size.get(object_class, 0), sample['size'])
        if object_class in occurance_counting:
            occurance_counting[object_class] += 1
        else:
            occurance_counting[object_class] = 1
    occurance_counting_list = [(key, value, object_top_conf.get(key, 0), object_top_size.get(key, 0)) for key, value in occurance_counting.items()]
    if ceil(thres_hold * 3) < len(occurance_counting_list):
        occurance_counting_list.sort(key=lambda x:(x[1], x[2], x[3]), reverse=True)
        occurance_counting_list = occurance_counting_list[:thres_hold * 3]
        
    if ceil(thres_hold * 2) < len(occurance_counting_list):
        occurance_counting_list.sort(key=lambda x:(x[2], x[3], x[1]), reverse=True)
        occurance_counting_list = occurance_counting_list[:thres_hold * 2 ]
    occurance_counting_list.sort(key=lambda x:(x[3], x[2], x[1]), reverse=True)
    occurance_counting_list = occurance_counting_list[:thres_hold]
    occurance_filter_counting_list = [key for (key, count, conf, size) in occurance_counting_list]
    
    object_attr_list = {}
    object_attr_dict = {}
    for single_object in occurance_filter_counting_list:
            object_attr_list[single_object] = []
            object_attr_dict[single_object] = {}
        
    for sample in attrs:
        object_class = sample['class']
        if object_class not in occurance_filter_counting_list:
            continue
        for attr in sample['attrs']:
            count, conf = object_attr_dict[object_class].get(attr['attribute'], (0, 0))
            object_attr_dict[object_class][attr['attribute']] = (count + 1, max(conf, attr['conf']))
            
    for object_class, object_attrs in object_attr_dict.items():
        for attr_name, (count, conf) in object_attrs.items():
            object_attr_list[object_class].append({'attribute': attr_name, 'conf': conf, 'count': count})
            
    for single_object in occurance_filter_counting_list:
        object_attr_list[single_object].sort(key=lambda x:(x['count'], x['conf']), reverse=True)
    attr_str_list = []
    for single_object_class in occurance_filter_counting_list:
        single_attrs = object_attr_list[single_object_class]
        single_attr_list = []
        attr_class_dict = set()
        for single_attr in single_attrs:
            if len(single_attr_list) > 1:
                break
            if single_attr['attribute'] not in single_attr_list \
                and reverse_attr_class_dict.get(single_attr['attribute'], None) not in attr_class_dict:
                single_attr_list.append(single_attr['attribute'])
                if reverse_attr_class_dict.get(single_attr['attribute'], None):
                    attr_class_dict.add(reverse_attr_class_dict[single_attr['attribute']])
        single_attr_list.append(single_object_class)
        attr_str_list.append(" ".join(single_attr_list))
    return ", ".join(attr_str_list)

def get_size(bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    assert x1 >= x0
    assert y1 >= y0
    return (x1 - x0) * (y1 - y0)

def object_loader_confident(vid_path_list, object_count, label_map):
    cls_to_idx = label_map["label_to_idx"]
    idx_to_cls = label_map['idx_to_label']
    total = 0
    count = 0
    objects_per_frame = {}
    for vid, vid_path in tqdm(vid_path_list):
        #print(vid)
        with open(vid_path, 'r') as f:
            frame_lists = json.load(f)
        vid_object_list = []
        vid_object_cls_list = []
        vid_class_attribute_list = [] 
        conf_set = {}
        for frame_result in frame_lists:
            frame_objects = frame_result['objects']
            frame_object_list = []
            frame_object_cls_list = []
            frame_class_attribute_list = []
            rect_list = []
            object_set = {}
            for single_object in frame_objects:
                # IOU threshold = 0.5
                total += 1
                if single_object['class'] in object_set:
                    iouflag = False
                    for rect in  object_set[single_object['class']]:
                        if get_iou(single_object['rect'], rect) >= 0.2:
                            
                            count += 1
                            #print(count * 1.0 / total)
                            iouflag = True
                            break
                    if iouflag:
                        continue
                else:
                    flag = False
                    for cached_rect in rect_list:
                        if get_iou(single_object['rect'], cached_rect) >= 0.5:
                            #print(count * 1.0 / total)
                            flag = True
                            break
                    if flag:
                        count += 1
                        continue
                    rect = single_object['rect']
                    obj_class =single_object['class']
                    if obj_class not in object_set:
                        object_set[obj_class] = [rect]
                    else:
                        object_set[obj_class].append(rect)
                    
                    attributes = []
                    rect_list.append(rect)
                    ### attribute selection
                    for attribute, attr_conf in zip(single_object['attributes'], single_object['attr_scores']):
                        if attr_conf > 0.2:
                            attributes.append({"attribute": attribute, "conf": attr_conf})
                        if len(attributes) > 1:
                            break
                    frame_class_attribute_list.append({'class': obj_class, 'attrs': attributes, 'class_conf': single_object['conf'], 'size': get_size(rect), 'rect': rect})
                    if len(frame_class_attribute_list) == object_count * 2:
                        break
            frame_class_attribute_list.sort(key=lambda x: x['class_conf'], reverse=True)
            frame_class_attribute_list = frame_class_attribute_list[:object_count*2]
            frame_class_attribute_list.sort(key=lambda x: x['size'], reverse=True)
            frame_class_attribute_list = frame_class_attribute_list[:object_count]
            for class_attribute in frame_class_attribute_list:
                frame_object_list.append(class_attribute['rect'])
                frame_object_cls_list.append(cls_to_idx[class_attribute['class']])
                
                if cls_to_idx[class_attribute['class']] not in conf_set:
                    conf_set[cls_to_idx[class_attribute['class']]] = class_attribute['class_conf']
                else:
                    conf_set[cls_to_idx[class_attribute['class']]] += class_attribute['class_conf']
                    
            if len(frame_object_list) < object_count:
                more = object_count - len(frame_object_list)
                frame_object_list.extend([[-1, -1, -1, -1] for i in range(more)])
                frame_object_cls_list.extend([0 for i in range(more)])
            assert len(frame_object_list) == object_count
            vid_object_list.append(frame_object_list)
            vid_object_cls_list.append(frame_object_cls_list)
            vid_class_attribute_list.append(frame_class_attribute_list)
            
        attr_sentence = get_attr_sent(vid_class_attribute_list, thres_hold=10)
        objects_per_frame[vid] = {"objects": torch.tensor(vid_object_list, dtype=torch.float),"obj_cls": torch.tensor(vid_object_cls_list, dtype=torch.int64), "attr_sentence": attr_sentence, "conf_set": conf_set}
    return objects_per_frame
                    

class ObjectEncoder(nn.Module):
    def __init__(self, in_dim, video_shape, object_num, nb_frames, patch_size=16):
        """
        video_size: original boxes scale
        spatial_slots: output spatial size
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.aligned = True
        self.sampling_ratio = -1
        self.video_hw = (video_shape[0], video_shape[1])
        self.nb_frames = nb_frames
        self.patch_to_d = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )
        print(f"object number count: {object_num}")
        
        self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, object_num, self.in_dim))
        
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

    def prepare_outdim(self, outdim):
        if isinstance(outdim, (int, float)):
            outdim = [outdim, outdim]
        assert len(outdim) == 2
        return tuple(outdim)

    def forward(self, features, boxes):
        """
        boxes: List[Tensor[T, N_OBJ, 4]]
        features: [BS, d, T, H=14, W=14]
        """
        #features = features.detach()
        cls_token, patch_tokens_ori = features[0:1,:], features[1:, :]
        patch_tokens = patch_tokens_ori.permute(1, 0, 2).reshape((-1, self.patch_size, self.patch_size, self.in_dim))
        
        bs_t, H, W, d = patch_tokens.shape # [BS * T, H, W, d]
        Horig, Worig = self.video_hw
        O = boxes.size(2)
        output_size = (H, W)
        spatial_scale = [H/Horig, W/Worig][0]
        patch_tokens = patch_tokens.permute(0, 3, 1, 2) # [BS * T, d, H, W]
        boxes = boxes.contiguous().flatten(0, 1) # [BS * T, O, 4]
        
        attn_mask = (boxes[:,:,0] == -1)
        
        boxes = boxes.type(patch_tokens.dtype)
        ret = roi_align(
            patch_tokens,
            list(boxes),
            output_size,
            spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )  # [BS * T * O, d, H, W]
        # T = T // 2
        ret = ret.reshape(bs_t, O, d, H, W) # [BS * T, O, d, H, W]
        object_tokens = ret.permute(0, 1, 3, 4, 2).contiguous() # [BS * T, O, H, W, d]
        
        object_tokens = self.patch_to_d(object_tokens) # [BS * T, O, H, W, d]
        object_tokens =torch.amax(object_tokens, dim=(-3,-2)) # [BS * T, O, d]
        
        box_categories = self.box_categories.unsqueeze(0).expand(bs_t // self.nb_frames, -1,-1,-1).reshape(-1, O, d).type(object_tokens.dtype)
        # normalization
        box_emb = self.c_coord_to_feature(boxes / 224.0)
        object_tokens = object_tokens + box_categories + box_emb # [BS, T, O, d]
        
        object_tokens = object_tokens.permute(1, 0, 2) # [O, BS * T, d]
        
        return object_tokens, attn_mask