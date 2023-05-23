#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================
# File: data_formatter.py
# Date: 2021-06-28
# Desc: 
# Version: 0.1
#==============================
import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
from datetime import datetime as dt

import struct
from torchkit.data import example_pb2
from PIL import Image
from tqdm import tqdm
import glob
import os
import csv
import multiprocessing as mp

headers = ['videoid','name','page_idx','page_dir','duration','contentUrl']
count = 0
vids=[]
with open('results_2M_train.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if count != 0:
           vids.append(row[0])
        count += 1
        if count % 1000 == 0:
            print(count,'id has loaded...')
'''
with open('results_2M_val.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if count != 0:
           vids.append(row[0])
        count += 1
        if count % 100 == 0:
            print(count,'id has loaded...')
'''
class Formatter:
    def __init__(self,
                 video_dir,
                 output_dir,
                 examples_per_shard=1000):
        #self.input_path = input_path

        self.index_path = os.path.join(output_dir, 'path.index')
        self.examples_per_shard = examples_per_shard
        self.video_dir = video_dir
        self.output_dir = output_dir
        #self.input_dir_list = self._load_image_dir_list(
        #    self.input_path)

    
    


    def write(self, id,transform=None):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        count = 0 
        shard_size = 0 
        shard_idx = -1
        shard_writer = None
        shard_path = None
        shard_offset = None
        idx_writer = open(self.index_path, 'w')

        b_time = time.time()
        #video_list = os.listdir(self.video_dir)
        video_list = vids
        count = 0
        for video in tqdm(video_list):
            count += 1
            if count %20 != id:
                continue
            vid = video.replace('.mp4','')
            cap = cv2.VideoCapture(self.video_dir+'/'+video)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(count,video,' frames:',frameCount,' fps:',fps)
            assert fps!=0,video
            '''
            if fps == 0:
                return
            else:
                continue
                '''
            imgs=[]
            for frame_id in range(0,frameCount,fps):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
                ret, frame = cap.read()
                if not ret: break
                imgs.append(frame)

            try:
                imgs = np.stack(imgs)
            except Exception as e:
                print(f'### Empty: {vid}, info: {e}')
                continue
            print(imgs.shape)
            img_bytes = imgs.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
            if shard_size == 0:
                print("{}: {} processed".format(dt.now(), count))
                shard_idx += 1
                record_filename = '{0}-{1:05}.tfrecord'.format('path', shard_idx)
                if shard_writer is not None:
                    shard_writer.close()
                shard_path = os.path.join(self.output_dir, record_filename) 
                shard_writer = tf.io.TFRecordWriter(shard_path)
                shard_offset = 0
        
            example_bytes = example.SerializeToString()
            shard_writer.write(example_bytes)
            shard_writer.flush()
            idx_writer.write(f'{vid}\t{shard_path}\t{shard_offset}\n')
            shard_offset += (len(example_bytes) + 16)

            if count % 1000 == 0:
                avg_time = (time.time() - b_time) * 1000 / count
                print(''.format(count))
            shard_size = (shard_size + 1) % self.examples_per_shard

        if shard_writer is not None:
            shard_writer.close()
        idx_writer.close()

def main(id):
    video_dir = ''
    output_dir = ''

    formatter = Formatter(video_dir,output_dir)
    formatter.write(id, )
    #formatter.read()


if __name__ == '__main__':
    for i in range(20):
        p = mp.Process(target=main,args=(i,))
        p.start()



