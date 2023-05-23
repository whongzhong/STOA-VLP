import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, abspath)
import time
import json
import torch
import random
import argparse
import numpy as np

from utils.util import get_logger
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_qa import CLIP4Clip
from modules.optimization import BertAdam
from dataloaders.dataloader_util import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")
from tqdm import tqdm

global logger


def get_args(description='Video Task Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train",    action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",     action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--resume",      action='store_true', help=" ")
    parser.add_argument('--loose_type',  action='store_true', help="Default using tight type for retrieval.")

    parser.add_argument("--datatype", default="msrvtt_qa", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--batch_size',         default=256,    type=int, help='batch size')
    parser.add_argument('--batch_size_val',     default=1000,   type=int, help='batch size')
    parser.add_argument('--max_words',          default=32,     type=int, help='')
    parser.add_argument('--max_frames',         default=12,     type=int, help='')
    parser.add_argument('--feature_framerate',  default=1,      type=int, help='')
    parser.add_argument('--num_thread_reader',  default=1,      type=int, help='')
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--n_display',          default=100,    type=int, help='Information display frequence')
    
    parser.add_argument('--clip_archive', type=str, default='data/models/clip', help='clip config path')
    parser.add_argument('--labelmap_path', type=str, default='data/object_feature/label_map.json"', help='VinVL labelmap path')

    parser.add_argument('--train_frame_order',  default=0,      type=int, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument('--eval_frame_order',   default=0,      type=int, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--slice_framepos',     default=2,      type=int, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

    parser.add_argument('--seed',       default=2021, type=int, help='random seed')
    parser.add_argument("--world_size", default=0, type=int, help="distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
    parser.add_argument("--rank",       default=0, type=int, help="distributed training")
    parser.add_argument('--n_gpu',      default=1, type=int, help="Changed in the execute process.")

    parser.add_argument('--epochs',     default=20,     type=int,   help='upper epoch limit')
    parser.add_argument('--lr',         default=5e-5,   type=float, help='initial learning rate')
    parser.add_argument('--coef_lr',    default=1e-3,     type=float, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--msrvtt_train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--msrvtt_val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--msrvtt_train_json', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--msrvtt_features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--msrvtt_qa_train_json', type=str, default='train.jsonl', help='')
    parser.add_argument('--msrvtt_qa_val_json', type=str, default='val.jsonl', help='')
    parser.add_argument('--msrvtt_qa_test_json', type=str, default='test.jsonl', help='')
    parser.add_argument('--msrvtt_qa_anslabel_json', type=str, default='', help='train_ans2label.json')
    parser.add_argument('--msvd_features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--msvd_qa_train_json', type=str, default='train_qa.json', help='')
    parser.add_argument('--msvd_qa_val_json', type=str, default='val_qa.json', help='')
    parser.add_argument('--msvd_qa_test_json', type=str, default='test_qa.json', help='')
    parser.add_argument('--msvd_qa_anslabel_json', type=str, default='', help='train_ans2label.json')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")

    parser.add_argument("--cache_dir",  default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--need_object", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument("--use_action", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument("--phrase_fuse", default="None", type=str, help="Whether to incorparate object infomation")
    parser.add_argument("--object_fuse", default="attention", type=str, help="Whether to incorparate object infomation")
    parser.add_argument("--insert_attr", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument("--attr_sim", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument("--attr_vtm", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument("--frame_loc", action='store_true', help="Whether to incorparate object infomation")
    
    parser.add_argument("--replace_cls", action='store_true', help="Whether to incorparate object infomation")
    parser.add_argument('--object_path', type=str, default='data/.train.csv', help='')
    parser.add_argument("--object_count", default=10, type=int, help="distribted training")
    parser.add_argument("--object_layer", default=10, type=int, help="layer that needs to incorparate object information, == 0 means no, > 0 means using specific layer, <0 means using layer lower than this numbers")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.output_dir, "ckpt")):
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu



    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    
    global_step = 0
    restore_dict = {}
    restore_path = os.path.join(args.output_dir, 'restore.bin')
    if args.init_model and not os.path.exists(restore_path):
        model_state_dict = torch.load(args.init_model, map_location='cpu')

        if 'model_state_dict' in model_state_dict:
            print('loading model from its model_state_dict')
            model_state_dict = model_state_dict['model_state_dict']
    elif os.path.exists(restore_path):
        print(f'find previous checkpoint, try to resume training from {restore_path}')
        t = torch.load(restore_path, map_location='cpu')
        
        model_state_dict = t['model_state_dict']
        global_step = t['global_step']
        if 'sampler' in t:
            restore_dict['sampler'] = t['sampler']
        if 'optim_state_dict' in t:
            restore_dict['optim_state_dict'] = t['optim_state_dict']
        if 'epoch' in t:
            restore_dict['epoch'] = t['epoch']
        assert model_state_dict is not None, "the model is not correctly loaded"
    else:
        logger.info("!!!!!!!!!!!!!!found no pretrained ckpts!!!!!!!!!!!!!!!!!!!!")
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, object_count=args.object_count)

    model.to(device)
    

    return model, global_step, restore_dict


def prep_optimizer(args, model, num_train_optimization_steps, device, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." in n and "fusion" not in n)]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]
    decay_clip_param_tp_fuse = [(n, p) for n, p in decay_param_tp if ("clip." in n and "fusion" in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and "fusion" not in n)]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
    no_decay_clip_param_tp_fuse = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and "fusion" in n)]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        # MMmodel
        {'params': [p for n, p in decay_clip_param_tp_fuse], 'weight_decay': weight_decay, 'lr': 1e-5},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},
        {'params': [p for n, p in no_decay_clip_param_tp_fuse], 'weight_decay': 0.0, 'lr': 1e-5}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        print(f'prepare optimizer from {resume}')
        t = torch.load(resume, map_location=device)
        assert t['optim_state_dict'] is not None, "reloading optimizer is not correct"
        optimizer.load_state_dict(t['optim_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, sampler, type_name="", optimizer=None, step=None):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "ckpt", "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", step))
    
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        os.remove(resume)
    assert optimizer is not None, 'optimizer is invalid'
    assert step is not None, 'step must be not None'
    checkpoint = {'global_step': step,
                  'model_state_dict': model_to_save.state_dict(),
                  'optim_state_dict': optimizer.state_dict(), 
                  'epoch': epoch}

    torch.save(checkpoint, output_model_file)
    torch.save(checkpoint, resume)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def eval_epoch(args, model, test_dataloader, device, n_gpu, step, epoch, tokenizer):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    n_correct = 0
    n_sample = 0
    object_layer = []
    count = 0
    vid_mapping = json.load(open('dataloaders/id_mapping.json','r'))
    if args.object_layer > 0:
        object_layer = [args.object_layer - 1]
    elif args.object_layer < 0:
        object_layer = [i for i in range(-args.object_layer)]
    with torch.no_grad():
        predicted_tensor = None
        targets_tensor = None
        vid_tensor = None
        mask_tensor = None
        answer_tensor = None
        count = 0
        for bid, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            vid_ids = batch[-1]
            batch = batch[:-1]
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            if args.need_object:
                question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list = batch
            else:
                question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask = batch

            num_correct, num_total, predict_ids = model(question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask, objects=object_tensor, object_layer=object_layer, attr_triple = (attr_pairs_text, attr_pairs_mask, attr_pairs_segment), attr_sim=args.attr_sim, insert_attr=args.insert_attr, replace_cls=args.replace_cls, phrase_fuse=args.phrase_fuse, object_fuse=args.object_fuse, obj_cls_tuple=(object_cls_tensor, id_list), use_action=args.use_action)
            if predicted_tensor is None:
                predicted_tensor = predict_ids
                targets_tensor = align_text
                mask_tensor = align_mask
                if args.datatype =='msrvtt_qa':
                    vid_tensor = targets_tensor.new_tensor([int(vid.strip("video")) for vid in vid_ids])
                else:
                    vid_tensor = targets_tensor.new_tensor([vid for vid in vid_ids])
                answer_tensor = answer_ids
            else:
                predicted_tensor = torch.cat((predicted_tensor, predict_ids))
                targets_tensor = torch.cat((targets_tensor, align_text))
                answer_tensor = torch.cat((answer_tensor, answer_ids))
                mask_tensor = torch.cat((mask_tensor, align_mask))
                if args.datatype =='msrvtt_qa':
                    vid_tensor = torch.cat((vid_tensor, vid_tensor.new_tensor([int(vid.strip("video")) for vid in vid_ids])))
                else:
                    vid_tensor = torch.cat((vid_tensor, vid_tensor.new_tensor([vid for vid in vid_ids])))

                
            n_correct += num_correct
            n_sample += num_total

            if args.local_rank == 0:
                print("{}/{}\r".format(bid, len(test_dataloader)))
    
        n_correct = torch.tensor([ n_correct ]).to('cuda')
        n_sample = torch.tensor([ n_sample ]).to('cuda')
        
        correct_gather = [torch.tensor([ n_correct]).to('cuda') for _ in range(args.world_size)]
        sample_gather = [torch.tensor([ n_sample ]).to('cuda') for _ in range(args.world_size)]
        
        
        
        torch.distributed.all_gather(correct_gather, n_correct, async_op=False)
        torch.distributed.all_gather(sample_gather, n_sample, async_op=False)
        torch.distributed.barrier()
        
        sample_list = torch.cat(sample_gather, dim=0)
        max_sample_num = int(torch.max(sample_list).cpu())
        
        if predicted_tensor.shape[0] < max_sample_num:
            pad_len = max_sample_num - predicted_tensor.shape[0]
            
            predicted_tensor = torch.cat([predicted_tensor] + [torch.full_like(predicted_tensor[:1], -1) for _ in range(pad_len)])
            targets_tensor = torch.cat([targets_tensor] + [torch.full_like(targets_tensor[:1], -1) for _ in range(pad_len)])
            mask_tensor = torch.cat([mask_tensor] + [torch.full_like(mask_tensor[:1], -1) for _ in range(pad_len)])
            vid_tensor = torch.cat([vid_tensor] + [torch.full_like(vid_tensor[:1], -1) for _ in range(pad_len)])
            answer_tensor = torch.cat([answer_tensor] + [torch.full_like(answer_tensor[:1], -1) for _ in range(pad_len)])
            
        
        predict_gather = [torch.empty_like(predicted_tensor) for _ in range(args.world_size)]
        targets_gather = [torch.empty_like(targets_tensor) for _ in range(args.world_size)]
        vid_gather = [torch.empty_like(vid_tensor) for _ in range(args.world_size)]
        mask_gather = [torch.empty_like(mask_tensor) for _ in range(args.world_size)]
        answer_gather = [torch.empty_like(answer_tensor) for _ in range(args.world_size)]
        
        torch.distributed.all_gather(predict_gather, predicted_tensor, async_op=False)
        torch.distributed.all_gather(targets_gather, targets_tensor, async_op=False)
        torch.distributed.all_gather(vid_gather, vid_tensor, async_op=False)
        torch.distributed.all_gather(mask_gather, mask_tensor, async_op=False)
        torch.distributed.all_gather(answer_gather, answer_tensor, async_op=False)
        torch.distributed.barrier()
        
        
        predict_gather = torch.cat(predict_gather, dim=0)
        targets_gather = torch.cat(targets_gather, dim=0)
        mask_gather = torch.cat(mask_gather, dim=0)
        vid_gather = torch.cat(vid_gather, dim=0)
        answer_gather = torch.cat(answer_gather, dim=0)
        
        correct_gather = torch.tensor(correct_gather)
        sample_gather = torch.tensor(sample_gather)
        answer_list = []
        for i in tqdm(range(vid_gather.shape[0])):
            if vid_gather[i] == -1:
                continue
            targets = tokenizer.decode(targets_gather[i][0][mask_gather[i][0]==1][1:-1].tolist()).strip()
            predicts = tokenizer.decode([int(predict_gather[i])])
            vid = f'video{vid_gather[i]}'
            answer_list.append({'vid': vid, 'golden': targets, "prediction": predicts, 'correct_label': int(predict_gather[i]) == int(answer_gather[i])})
            
        print('correct_gather', correct_gather.sum(), 'sample_gather', sample_gather.sum())
        total_acc = correct_gather.sum() / sample_gather.sum() * 100
    
    if args.local_rank == 0:
        answer_list.insert(0, {'ACC': total_acc.item()})
        with open(os.path.join(args.output_dir, f"prediction_epoch_{epoch}_step_{step}.json"), 'w') as f:
            json.dump(answer_list, f)
        logger.info(f"epoch: {epoch}, step: {step}: VideoQA-Metric")
        logger.info('\t>>>  Acc: {:.2f} - data_len: {:.1f} - loader_len: {:.1f}'.format(total_acc, sample_gather.sum(), len(test_dataloader)))
    model.train()
    return total_acc

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, sampler, local_rank=0, tokenizer=None):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    
    object_layer = []
    if args.object_layer > 0:
        object_layer = [args.object_layer - 1]
    elif args.object_layer < 0:
        object_layer = [i for i in range(-args.object_layer)]
    logger.info(f"epoch {epoch}")
        
    logger.info(f"incorparate object in layer {str(object_layer)}")
    logger.info(f"using task insert_attr: {args.insert_attr}")
    logger.info(f"using task attr_sim: {args.attr_sim}")
    logger.info(f"using task replace_cls: {args.replace_cls}")
    logger.info(f"using task phrase_fuse type: {args.phrase_fuse}")
    logger.info(f"using task object_fuse type: {args.object_fuse}")
    logger.info(f"using action: {args.use_action}")
    for step, batch in enumerate(train_dataloader):
        batch = batch[:-1]
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        if args.need_object:
            question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask, object_tensor, attr_pairs_text, attr_pairs_mask, attr_pairs_segment, object_cls_tensor, id_list = batch
        else:
            question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask = batch

        loss = model(question_text, question_mask, pairs_segment, video, video_mask, answer_ids, align_text, align_mask, objects=object_tensor, object_layer=object_layer, attr_triple = (attr_pairs_text, attr_pairs_mask, attr_pairs_segment), attr_sim=args.attr_sim, insert_attr=args.insert_attr, replace_cls=args.replace_cls, phrase_fuse=args.phrase_fuse, object_fuse=args.object_fuse, obj_cls_tuple=(object_cls_tensor, id_list), use_action=args.use_action)
        
        loss.backward()

        total_loss += float(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step()
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

        global_step += 1
        
        if global_step % log_step == 0 and local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", 
                        epoch + 1, args.epochs, 
                        step + 1, len(train_dataloader), 
                        "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                        float(loss),
                        (time.time() - start_time) / log_step )
            start_time = time.time()

    if len(train_dataloader) > 0:
        total_loss = total_loss / len(train_dataloader)
    else:
        total_loss = 0
    return total_loss, global_step





def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    model, global_step, restore_dict = init_model(args, device,n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    print("##### Parameter Freezed #####")
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train
            
            if name.find('object') >= 0:
                continue
            elif name.find("fusion") == 0:
                continue
            # paramenters which < freeze_layer_num will be freezed
            print(name, "is freezed")
            param.requires_grad = False
    print("##### Parameter Freezed End #####")

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length, test_sampler = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
    val_dataloader, val_length, val_sampler = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = len(train_dataloader) * args.epochs
        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, args.local_rank, coef_lr=coef_lr)
        
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        if 'epoch' not in restore_dict:
            restore_dict['epoch'] = 0
        for epoch in range(restore_dict['epoch'], args.epochs):
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, sampler = train_sampler, local_rank=args.local_rank)

            acc_test = eval_epoch(args, model, test_dataloader, device, n_gpu, global_step, epoch + 1, tokenizer)
            
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = None
                if args.rank == 0:
                    output_model_file = save_model(epoch, args, model, sampler = train_sampler, type_name="", optimizer=optimizer, step=global_step)

                if best_score <= acc_test:
                    best_score = acc_test
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the Acc is: {:.4f}".format(best_output_model_file, best_score))
            train_sampler.set_epoch(epoch + 1)

    elif args.do_eval:
        acc_test = eval_epoch(args, model, test_dataloader, device, n_gpu, global_step + 10, 5, tokenizer)


if __name__ == '__main__':
    main()
