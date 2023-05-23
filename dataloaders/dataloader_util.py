import torch
import numpy as np
from torch.utils.data import DataLoader

from dataloaders.dataloader_webvid import webvid_TrainDataLoader

from dataloaders.dataloader_msrvtt_caption import MSRVTT_DataLoader as MSRVTT_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_TrainDataLoader as MSRVTT_Caption_TrainDataLoader
from dataloaders.dataloader_msvd_caption import MSVD_DataLoader as MSVD_Caption_DataLoader


from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader as MSRVTT_Retrieval_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader as MSRVTT_Retrieval_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader as MSVD_Retrieval_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader as LSMDC_Retrieval_DataLoader
from dataloaders.dataloader_didemo_retrieval import DiDeMo_DataLoader as DiDeMo_Retrieval_DataLoader

from dataloaders.dataloader_msrvtt_qa import MSRVTT_QA_DataLoader
from dataloaders.dataloader_msvd_qa import MSVD_QA_DataLoader
from dataloaders.dataloader_msrvtt_mc import MSRVTT_DataLoader as MSRVTT_MC_DataLoader

def dataloader_webvid_pretrain(args, tokenizer, restore_dict = None):
    webvid_dataset = webvid_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        labelmap_path=args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        split=args.worker_index,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(webvid_dataset)    
    dataloader = DataLoader(
        webvid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(webvid_dataset), train_sampler

def dataloader_caption_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        labelmap_path=args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_caption_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.data_path,
        features_path=args.features_path,
        labelmap_path=args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_caption_msvd_train(args, tokenizer, restore_dict = None):
    msvd_dataset = MSVD_Caption_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path=args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_caption_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_Caption_DataLoader(
        subset="test",
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path=args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)

def dataloader_retrieval_didemo_train(args, tokenizer, restore_dict = None):
    didemo_dataset = DiDeMo_Retrieval_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size ,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler

def dataloader_retrieval_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMo_Retrieval_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
    )
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)

def dataloader_retrieval_msrvtt_train(args, tokenizer, restore_dict = None):
    msrvtt_dataset = MSRVTT_Retrieval_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_retrieval_msrvtt_val(args, tokenizer):
    msrvtt_valset = MSRVTT_Retrieval_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_valset, shuffle=False)
    dataloader = DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
    )

    return dataloader

def dataloader_retrieval_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_Retrieval_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_retrieval_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_Retrieval_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_retrieval_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_Retrieval_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_retrieval_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_Retrieval_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size ,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler

def dataloader_retrieval_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_Retrieval_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        labelmap_path = args.labelmap_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count    
        )
    dataloader = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(lsmdc_testset)



def dataloader_msrvtt_qa_train(args, tokenizer, restore_dict = None):
    msrvtt_trainset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_train_json,
        features_path=args.msrvtt_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        split="train"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_trainset)
    dataloader = torch.utils.data.DataLoader(
        msrvtt_trainset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_trainset), train_sampler


def dataloader_msrvtt_qa_val(args, tokenizer):
    msrvtt_valset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_val_json,
        features_path=args.msrvtt_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        split="val"
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_valset)
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
    )
    return dataloader_msrvtt, len(msrvtt_valset), val_sampler


def dataloader_msrvtt_qa_test(args, tokenizer):
    msrvtt_testset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_test_json,
        features_path=args.msrvtt_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        split="test"
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
    )
    return dataloader_msrvtt, len(msrvtt_testset), test_sampler

def dataloader_msvd_qa_train(args, tokenizer, restore_dict = None):
    msvd_trainset = MSVD_QA_DataLoader(
        json_path=args.msvd_qa_train_json,
        features_path=args.msvd_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msvd_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        subset="train"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_trainset)
    dataloader = torch.utils.data.DataLoader(
        msvd_trainset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_trainset), train_sampler


def dataloader_msvd_qa_val(args, tokenizer):
    msvd_valset = MSVD_QA_DataLoader(
        json_path=args.msvd_qa_val_json,
        features_path=args.msvd_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msvd_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        subset="val"
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(msvd_valset)
    dataloader_msvd = torch.utils.data.DataLoader(
        msvd_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
    )
    return dataloader_msvd, len(msvd_valset), val_sampler


def dataloader_msvd_qa_test(args, tokenizer):
    msvd_testset = MSVD_QA_DataLoader(
        json_path=args.msvd_qa_test_json,
        features_path=args.msvd_features_path,
        labelmap_path = args.labelmap_path,
        qa_anslabel_json_path=args.msvd_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        need_object=args.need_object,
        object_path=args.object_path,
        object_count=args.object_count,
        subset="test"
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(msvd_testset)
    dataloader_msvd = torch.utils.data.DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
    )
    return dataloader_msvd, len(msvd_testset), test_sampler


def dataloader_mc_msrvtt_train(args, tokenizer, restore_dict = None):
    msrvtt_dataset = MSRVTT_Retrieval_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        labelmap_path = args.labelmap_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_mc_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_MC_DataLoader(
        mc_json_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset,shuffle=False)
    dataloader = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )

    return dataloader,len(msrvtt_testset)


def dataloader_mc_msrvtt_val(args, tokenizer):
    msrvtt_valset = MSRVTT_MC_DataLoader(
        mc_json_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_valset,shuffle=False)
    dataloader = DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
    )

    return dataloader,len(msrvtt_valset)


DATALOADER_DICT = {
    "pretrain": {"train": webvid_TrainDataLoader, "val":None, "test":dataloader_retrieval_msrvtt_test},
    "msrvtt_caption": {"train":dataloader_caption_msrvtt_train, "val":None, "test":dataloader_caption_msrvtt_test},
    "msvd_caption": {"train":dataloader_caption_msvd_train, "val":None, "test":dataloader_caption_msvd_test},
    "msrvtt_retrieval": {"train":dataloader_retrieval_msrvtt_train, "val": None, "test":dataloader_retrieval_msrvtt_test},
    "msvd_retrieval": {"train":dataloader_retrieval_msvd_train, "val":dataloader_retrieval_msvd_test, "test":dataloader_retrieval_msvd_test},
    "lsmdc_retrieval": {"train":dataloader_retrieval_lsmdc_train, "val": None, "test":dataloader_retrieval_lsmdc_test},
    "didemo_retrieval": {"train": dataloader_retrieval_didemo_train, "val": None, "test":dataloader_retrieval_didemo_test},
    "msrvtt_qa": {"train": dataloader_msrvtt_qa_train, "val": dataloader_msrvtt_qa_val, "test": dataloader_msrvtt_qa_test},
    "msvd_qa": {"train": dataloader_msvd_qa_train, "val": dataloader_msvd_qa_val, "test": dataloader_msvd_qa_test},
    "msrvtt_mc": {"train":dataloader_mc_msrvtt_train, "val":dataloader_mc_msrvtt_val, "test":dataloader_mc_msrvtt_test}
    }
