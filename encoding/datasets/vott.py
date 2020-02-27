import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor

from .base import BaseDataset
class VOTTSegmentation(BaseDataset):
    NUM_CLASS = 4
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(VOTTSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.dataset_root = os.path.join(root, 'vott-json-export')

        self.json = self._load_manifest_json()

        self.tags = [None] + [tag['name'] for tag in self.json['tags']]

        # FIXME:
        # せっかくBaseDatasetにoverride用のプロパティが存在するのに
        # 直接NUM_CLASS名指しで叩いてくるスクリプトがいるので
        # 一旦こうするしかない
        VOTTSegmentation.NUM_CLASS = len(self.tags)

        self.palette = self._generate_mask_palette(self.json['tags'])

        self.assets = list(self.json['assets'].values())
        # self.assets = [asset for asset in self.json['assets'].values() if len(asset['regions']) > 0]

        # 10分の1はvalidation用
        if self.mode == 'train':
            self.assets = [a for i,a in enumerate(self.assets) if i % 10 != 0]
        else:
            self.assets = [a for i,a in enumerate(self.assets) if i % 10 == 0]

    def __getitem__(self, index):
        asset = self.assets[index]
        name = asset['asset']['name']
        img_path = os.path.join(self.dataset_root, name)
        mask_path = os.path.join(self.dataset_root, name + '.target.png')

        img = Image.open(img_path).convert('RGB')
        mask = self._mask(mask_path, img.size, asset['regions'])

        ## ここから定型句
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
        return len(self.assets)

    def _load_manifest_json(self):
        for path in os.listdir(self.dataset_root):
            if path.endswith('.json'):
                return json.load(open(os.path.join(self.dataset_root, path), 'r'))
        else:
            raise AssertionError

    def _generate_mask_palette(self, tags):
        # index 0 は nothing
        # palette は8bitの[(r,g,b), ...]
        # putpaletteは適当に残りを埋めてくれるので短くても構わない
        palette = [(0,0,0)] + [ImageColor.getcolor(tag['color'], 'RGB') for tag in tags]
        # putpaletteに渡す時は、[r,g,b, r,g,b, ...]
        return np.array(palette, dtype = 'int8').flatten()

    def _mask(self, mask_path, image_size, regions):
        mask = Image.new('P', image_size, 0) # P: palette(index color)
        mask.putpalette(self.palette)
        drawer = ImageDraw.Draw(mask)

        # self.tagsに登場する順番で、マスクを描画したい
        # 描画は順に上書きされるので、tagsで後に登場するものほど優先して使われるということ
        regions = sorted(regions, key = lambda region: self.tags.index(region['tags'][0]))
        for i, region in enumerate(regions):
            # 1領域1タグ前提。
            # if len(region['tags']) != 1:
            #     print(mask_path)
            #     print("too many tags annotated into single region:")
            #     print(region['tags'])
            palette_index = self.tags.index(region['tags'][0])

            drawer.polygon(
                [(p['x'], p['y']) for p in region['points']],
                outline=palette_index,
                fill=palette_index
                )

        mask.save(mask_path)
        return mask
