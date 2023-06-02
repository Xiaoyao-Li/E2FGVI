import sys
import os
sys.path.append(os.getcwd())
import cv2
import json
import pickle
from PIL import Image
from typing import Tuple

import numpy as np
import torch
from core.utils import to_tensors
from datasets.misc import collate_fn_epicsub
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# class EpicKitchen(Dataset):
#     """ Dataset for epic kitchen
#     """
#     LENGTH_FRAME_ID = 10
#     def __init__(self, part: str, clip: str, 
#                  basedir='/mnt/seagate12t/EPIC-KITCHEN/EPIC-KITCHEN') -> None:
#         super(EpicKitchen, self).__init__()
#         info = json.load(open(os.path.join(basedir, 'info.json'), 'r'))

#         self.image_dir = os.path.join(basedir, info[part][clip]['path'])
#         self.mask_dir = os.path.join(basedir, part, 'mask_frames', clip)
#         self.out_basedir = os.path.join(basedir, part, 'agentago_frames', clip)
#         os.makedirs(self.out_basedir, exist_ok=True)
#         self.total_count = info[part][clip]['count']

#     def _index_to_img_fn(self, index) -> str:
#         return f'frame_{index + 1:0{self.LENGTH_FRAME_ID}d}.jpg'

#     def __len__(self) -> int:
#         return self.total_count

#     def __getitem__(self, index) -> Tuple:
#         img_path = os.path.join(self.basedir, self._index_to_img_fn(index))
#         img = cv2.imread(img_path)
#         img = Image.fram
#         aug_input = T.AugInput(img, sem_seg=None)
#         self.aug(aug_input)
#         img = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1))

#         return {"image": img, "height": self.img_height, "width": self.img_width, 
#                 'out_path': os.path.join(self.out_basedir, self._index_to_img_fn(index))}
    
#     def get_dataloader(self, **kwargs):
#         return DataLoader(self, **kwargs)


class EpicKitchenSub(Dataset):
    """ Dataset for sub epic kitchen (start and end frame of each action clip)
    """
    _LENGTH_FRAME_ID = 10
    _LENGTH_NEIGHBOR = 5
    _LENGTH_FRAMES_PER_INS = 1 + 2 * _LENGTH_NEIGHBOR
    _WIDTH = 456
    _HEIGHT = 256
    def __init__(self, part: str, clip: str,
                 basedir='/mnt/seagate12t/EPIC-KITCHEN/EPIC-KITCHEN') -> None:
        super(EpicKitchenSub, self).__init__()
        # read dataset metadata
        info = json.load(open(os.path.join(basedir, 'info.json'), 'r'))
        self.annotations = pd.read_csv(os.path.join(basedir, 'EPIC100_annotations.csv'))

        self.part = part
        self.clip = clip
        # filter annotations
        self.annotations = self.annotations[(self.annotations['participant_id'] == self.part) & (self.annotations['video_id'] == self.clip)]

        self.rgb_dir = os.path.join(basedir, info[part][clip]['path'])
        self.mask_dir = os.path.join(basedir, part, 'mask_frames', clip)
        self.out_basedir = os.path.join(basedir, part, 'agentago_frames', clip)
        os.makedirs(self.out_basedir, exist_ok=True)
        self.clip_length = info[part][clip]['count']

        self._construct_index_mapping_list()

    def _construct_index_mapping_list(self) -> None:
        # construct index mapping list
        self.index_mapping_list = []
        for _, row in self.annotations.iterrows():
            start_frame = row['start_frame']
            end_frame = row['stop_frame']
            
            # insert start frames into index_mapping_list
            frames_list = [[i] for i in range(start_frame, end_frame + 1) if not os.path.exists(os.path.join(self.out_basedir, self._index_to_img_fn(i)))]
            self._init_padding_params()
            self.index_mapping_list += frames_list


    def _index_to_img_fn(self, index) -> str:
        return f'frame_{index:0{self._LENGTH_FRAME_ID}d}.jpg'

    def __len__(self) -> int:
        return len(self.index_mapping_list)

    def __getitem__(self, index) -> Tuple:
        frames_id_list = self.index_mapping_list[index]
        frames_list = []
        masks_list = []
        for frame_id in frames_id_list:
            # read rgb image
            img_path = os.path.join(self.rgb_dir, self._index_to_img_fn(frame_id))
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # read mask
            mask_path = os.path.join(self.mask_dir, self._index_to_img_fn(frame_id))
            mask = Image.open(mask_path)
            mask = np.array(mask.convert('L'))
            mask = np.array(mask > 0).astype(np.uint8)
            mask = cv2.dilate(mask,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
            mask = Image.fromarray(mask * 255)
            frames_list.append(img)
            masks_list.append(mask)
        imgs = to_tensors()(frames_list).unsqueeze(0) * 2 - 1
        masks = to_tensors()(masks_list).unsqueeze(0)
        masked_imgs = imgs * (1 - masks)
        masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :self._HEIGHT + self.h_pad, :]
        masked_imgs = torch.cat(
            [masked_imgs, torch.flip(masked_imgs, [4])],
            4)[:, :, :, :, :self._WIDTH + self.w_pad]
        
        outpaths = [os.path.join(self.out_basedir, self._index_to_img_fn(frame_id)) for frame_id in frames_id_list]
        
        return {"masked_imgs": masked_imgs, "outpaths": outpaths}
    
    def _init_padding_params(self):
        MOD_SIZE_H = 60
        MOD_SIZE_W = 108
        self.h_pad = (MOD_SIZE_H - self._HEIGHT % MOD_SIZE_H) % MOD_SIZE_H
        self.w_pad = (MOD_SIZE_W - self._WIDTH % MOD_SIZE_W) % MOD_SIZE_W

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

if __name__ == '__main__':
    dataloader = EpicKitchenSub(part='P01', clip='P01_01'
                                ).get_dataloader(batch_size=4,
                                                 collate_fn=collate_fn_epicsub,
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 shuffle=False,)
    
    for i_b, batch in enumerate(dataloader):
        print(i_b, batch['masked_imgs'].shape, len(batch['outpaths'][0]))