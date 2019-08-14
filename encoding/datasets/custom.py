import os
import glob
import numpy as np

import torch

from PIL import Image
from tqdm import trange

from .base import BaseDataset

class CustomDataset(BaseDataset):
    NUM_CLASS = 13

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CustomDataset, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.transform = transform
        self.target_transform = target_transform

        img_folder = os.path.join(root, 'custom/' + split)
        self.images = []
        for path in os.listdir(img_folder):
            if path.endswith('.png'):
                self.images.append(os.path.join(img_folder, path))

    def __getitem__(self, index):
        path = self.images[index]

        img = Image.open(path).convert('RGB')
        if self.split == 'test':
            return img, os.path.basename(path)

        basename, ext = os.path.splitext(path)
        mask = self._mask(basename)

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _mask(self, basename):
        masks = []

        # 'cat.png' => ['cat_01.bmp', 'cat_02.bmp', ...]
        for i in range(self.NUM_CLASS):
            maskpath = '{basename}_{i:02}.bmp'.format(basename = basename, i = i + 1)
            m = np.array(Image.open(maskpath).convert('L'))
            # 0,255の2値画像なので、255で割って i+1を掛ければ、マスク値のindexになる
            m = m / 255 * (i+1)
            masks.append(m)
        masks = np.array(masks)

        # 先頭に 0埋めされたmaskを追加する。
        # これがindex = 0になる
        np.insert(masks, 0, 0, axis=0)

        # 何番目の画像が最初にnonzeroになるかでindexを算出する。
        # データ的には 1 pixel : NUM_CLASS label なのだが、
        # NNモデル的には 1 pixel : 1 labelなので、何かしらの手段で圧縮する必要がある
        flatten = (masks!=0).argmax(axis=0)

        return Image.fromarray(np.uint8(flatten))
