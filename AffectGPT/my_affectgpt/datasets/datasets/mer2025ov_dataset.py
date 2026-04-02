import os
import tqdm
import copy
import json
import random
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Optional, Sequence

import decord
from decord import VideoReader

import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from my_affectgpt.processors import transforms_video, AlproVideoTrainProcessor
from my_affectgpt.conversation.conversation_video import Conversation,SeparatorStyle
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.datasets.datasets.base_grpo_dataset import BaseGRPODataset
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio # 将上面的函数功能拆解为两个
from toolkit.utils.functions import string_to_list
import config

from toolkit.utils.read_files import *


#############################################################
## With 1200 samples
#############################################################
# # 要让模型同时支持audio, video, text三部分输入信息才行
# class MER2025OV_Dataset(BaseDataset):
#     def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
#                        dataset_cfg=None, model_cfg=None):
        
#         self.dataset = 'MER2025OV'
#         if dataset_cfg is not None:
#             self.label_type = dataset_cfg.label_type
#             self.face_or_frame = dataset_cfg.face_or_frame
#             print (f'Read data type: ######{self.label_type}######')
#             print (f'Read data type: ######{self.face_or_frame}######')
#             self.needed_data = self.get_needed_data(self.face_or_frame)
#             print (self.needed_data) # ['audio', 'frame', 'face']
        
#         ################# 直接手动指定所有信息的存储路径 #################
#         name2subtitle = {}
#         subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
#         df = pd.read_csv(subtitle_csv)
#         for _, row in df.iterrows():
#             name = row['name']
#             subtitle = row['english']
#             if pd.isna(subtitle): subtitle=""
#             name2subtitle[name] = subtitle
#         self.name2subtitle = name2subtitle
        
#         name2openset = {} # 532
#         openset_csv = config.PATH_TO_LABEL[self.dataset]
#         df = pd.read_csv(openset_csv)
#         for _, row in df.iterrows():
#             name = row['name']
#             openset = row['openset']
#             openset = string_to_list(openset)
#             name2openset[name] = ", ".join(openset)
#         self.name2openset = name2openset
        
#         vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
#         wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
#         face_root= config.PATH_TO_RAW_FACE[self.dataset]
#         ##################################################################

#         # use base model initialize approach
#         super().__init__(vis_processor=vis_processor, 
#                          txt_processor=txt_processor,
#                          img_processor=img_processor,
#                          vis_root=vis_root,
#                          face_root=face_root,
#                          wav_root=wav_root,
#                          model_cfg=model_cfg,
#                          dataset_cfg=dataset_cfg)
        
#     def _get_video_path(self, sample):
#         full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
#         return full_video_fp
    
#     def _get_audio_path(self, sample):
#         full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
#         return full_audio_fp

#     def _get_face_path(self, sample):
#         full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
#         return full_face_fp
    
#     # for inference
#     def read_test_names(self):
#         label_csv = config.PATH_TO_LABEL[self.dataset]
#         test_names  = func_read_key_from_csv(label_csv, 'name')
#         assert len(test_names) == 1200
#         return test_names

#     def get_test_name2gt(self):
#         return self.name2openset




#############################################################
## For 20000 candidates
#############################################################
# 要让模型同时支持audio, video, text三部分输入信息才行
class MER2025OV_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                       dataset_cfg=None, model_cfg=None):
        
        self.dataset = 'MER2025OV'
        if dataset_cfg is not None:
            self.label_type = dataset_cfg.label_type
            self.face_or_frame = dataset_cfg.face_or_frame
            print (f'Read data type: ######{self.label_type}######')
            print (f'Read data type: ######{self.face_or_frame}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data) # ['audio', 'frame', 'face']
        
        ################# 直接手动指定所有信息的存储路径 #################
        name2subtitle = {}
        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset]
        df = pd.read_csv(subtitle_csv)
        for _, row in df.iterrows():
            name = row['name']
            subtitle = row['english']
            if pd.isna(subtitle): subtitle=""
            name2subtitle[name] = subtitle
        self.name2subtitle = name2subtitle
        
        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root= config.PATH_TO_RAW_FACE[self.dataset]
        ##################################################################

        # use base model initialize approach
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=vis_root,
                         face_root=face_root,
                         wav_root=wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
        
    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
        return full_video_fp
    
    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp

    def _get_face_path(self, sample):
        full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
        return full_face_fp
    
    # for inference
    def read_test_names(self):
        label_csv = os.path.join(config.DATA_DIR[self.dataset], 'track_all_candidates.csv')      
        test_names  = func_read_key_from_csv(label_csv, 'name')
        assert len(test_names) == 20000
        return test_names


class MER2025OV_GRPO_Dataset(BaseGRPODataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                       dataset_cfg=None, model_cfg=None):

        self.dataset = 'MER2025OV'
        self.question_type = "ovlabel"
        if dataset_cfg is not None:
            self.face_or_frame = dataset_cfg.face_or_frame
            self.question_type = dataset_cfg.get("question_type", "ovlabel")
            print (f'Read data type: ######{self.face_or_frame}######')
            print (f'Question type: ######{self.question_type}######')
            self.needed_data = self.get_needed_data(self.face_or_frame)
            print (self.needed_data)

        subtitle_csv = config.PATH_TO_TRANSCRIPTIONS[self.dataset] #读取文本转写数据
        df = pd.read_csv(subtitle_csv)
        # subtitle_chieng.csv expected format (at least these columns):
        # name,english
        # 000001,I'm very happy today.
        # 000002,
        # 000003,Why did you do that?
        self.name2subtitle = {}
        for _, row in df.iterrows():
            name = row["name"]
            subtitle = row["english"]
            if pd.isna(subtitle):
                subtitle = ""
            self.name2subtitle[name] = subtitle

        label_csv = config.PATH_TO_LABEL[self.dataset]#读取标签数据
        df = pd.read_csv(label_csv)
        # track2_test.csv expected format (at least these columns):
        # name,openset
        # 000001,"['happy', 'excited']"
        # 000002,"[]"
        # 000003,"['sad']"
        self.name2openset = {}
        self.annotation = []
        for _, row in df.iterrows():
            name = row["name"]
            openset = row["openset"]
            openset_list = string_to_list(openset)
            if len(openset_list) == 0:
                openset_list = ["neutral"]
            openset_text = ", ".join(openset_list)
            self.name2openset[name] = openset_text
            self.annotation.append(
                {
                    "name": name,
                    "subtitle": self.name2subtitle.get(name, ""),
                    "ovlabel": openset_text,
                    "ovlabel_list": openset_list,
                    "question_type": self.question_type,
                }
            )

        vis_root = config.PATH_TO_RAW_VIDEO[self.dataset]
        wav_root = config.PATH_TO_RAW_AUDIO[self.dataset]
        face_root = config.PATH_TO_RAW_FACE[self.dataset]

        super().__init__(
            vis_processor=vis_processor,
            txt_processor=txt_processor,
            img_processor=img_processor,
            vis_root=vis_root,
            face_root=face_root,
            wav_root=wav_root,
            model_cfg=model_cfg,
            dataset_cfg=dataset_cfg,
        )

    def _get_video_path(self, sample):
        full_video_fp = os.path.join(self.vis_root, sample['name'] + '.mp4')
        return full_video_fp

    def _get_audio_path(self, sample):
        full_audio_fp = os.path.join(self.wav_root, sample['name'] + '.wav')
        return full_audio_fp

    def _get_face_path(self, sample):
        full_face_fp = os.path.join(self.face_root, sample['name'], sample['name'] + '.npy')
        return full_face_fp

    def build_prompt_text(self, sample):
        question = self.func_get_qa_ovlabel(sample, question_only=True)
        prompt = self.get_prompt_for_multimodal(self.face_or_frame, sample["subtitle"], question)
        return self.replace_token_for_multimodal(prompt)

    def build_reward_meta(self, sample):
        return {
            "name": sample["name"],
            "subtitle": sample["subtitle"],
            "gt_openset_text": sample["ovlabel"],
            "gt_openset_list": copy.deepcopy(sample["ovlabel_list"]),
        }
