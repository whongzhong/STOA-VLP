from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, abspath)
import torch
import numpy as np
import random
import os
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.dataloader_util import DATALOADER_DICT
from modules.modeling_mc import CLIP4Clip
from modules.optimization import BertAdam
from utils.util import get_logger
from tqdm import tqdm
torch.distributed.init_process_group(backend="nccl")
global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    
    parser.add_argument('--clip_archive', type=str, default='data/models/clip', help='clip config path')
    parser.add_argument('--labelmap_path', type=str, default='data/object_feature/label_map.json"', help='VinVL labelmap path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='mask ratio')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

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
        logger.info("!!!!!!!!!found no pretrained ckpts!!!!!!!!!!!!!!!!")
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)
    

    return model, global_step, restore_dict

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

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
        {'params': [p for n, p in decay_clip_param_tp_fuse], 'weight_decay': weight_decay, 'lr': 1e-5},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},
        {'params': [p for n, p in no_decay_clip_param_tp_fuse], 'weight_decay': 0.0, 'lr': 1e-5}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
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
    # torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(checkpoint, resume)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model




@torch.no_grad()
def mc_eval(args, model, val_loader,device, step, epoch):
    model.module.eval()
    with torch.no_grad():
        n_sample = 0
        n_correct = 0
        n_batch = 0
        loss = 0

        for step, batch in tqdm(enumerate(val_loader)):
                # multi-gpu does scattering it-self
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

                options, options_mask, options_segment, video, video_mask, answer = batch
                tmp_correct , tmp_sample, tmp_loss = model.module.get_mc_metrics(options, options_mask, options_segment, video, video_mask, answer)
                n_correct += tmp_correct
                n_sample += tmp_sample
                loss += tmp_loss
                n_batch += 1
                
        n_correct = torch.tensor([ n_correct ]).to('cuda')
        n_sample = torch.tensor([ n_sample ]).to('cuda')
        correct_gather = [torch.tensor([ n_correct]).to('cuda') for _ in range(args.world_size)]
        sample_gather = [torch.tensor([ n_sample ]).to('cuda') for _ in range(args.world_size)]
        torch.distributed.all_gather(correct_gather, n_correct, async_op=False)
        torch.distributed.all_gather(sample_gather, n_sample, async_op=False)
        torch.distributed.barrier()
        correct_gather = torch.tensor(correct_gather)
        sample_gather = torch.tensor(sample_gather)
        acc = correct_gather.sum() / sample_gather.sum() * 100
        if args.rank == 0:
            logger.info(f"epoch {epoch}, step: {step}: ACC: {acc.item()}")
            print('MC','   ACC:',acc.item(),'%')
            
    model.module.train()


def train_epoch(epoch, args, model, train_dataloader, test_dataloader, device, n_gpu, optimizer, scheduler, global_step, sampler, local_rank=0, val_loader=None):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    mc_eval(args, model, test_dataloader,device, global_step, epoch + 1)
    for step, batch in enumerate(train_dataloader):

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        # if n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.rank == 0:
            print(f'step is {step}, loss is ', loss)

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

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
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    if len(train_dataloader) > 0:
        total_loss = total_loss / len(train_dataloader)
    else:
        total_loss = total_loss
    return total_loss, global_step

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model, global_step, restore_dict = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
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

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                if 'positional_embedding_8x8' in name or 'fusion' in name:
                    continue
                print(name, ' is freezed')
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
    else:
        val_dataloader, val_length = test_dataloader, test_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer, restore_dict)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"

        if 'epoch' not in restore_dict:
            restore_dict['epoch'] = 0
        for epoch in range(restore_dict['epoch'], args.epochs):
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, test_dataloader,device, n_gpu, optimizer,
                                               scheduler, global_step, sampler = train_sampler, local_rank=args.local_rank, val_loader=val_dataloader)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            

            mc_eval(args, model, val_dataloader,device, global_step, epoch + 1)
            output_model_file = None
            if args.rank == 0:
                output_model_file = save_model(epoch, args, model, sampler = train_sampler, type_name="", optimizer=optimizer, step=global_step)
            train_sampler.set_epoch(epoch + 1)




if __name__ == "__main__":
    main()