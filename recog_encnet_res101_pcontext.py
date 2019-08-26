# -*- coding: utf-8 -*-
import os
import sys
import torch

def resize(img, crop_size=480):
    """
    長辺を合わせるように正方形化する
    """
    w, h = img.size
    if w > h:
        ow = crop_size
        oh = int(1.0 * ow * h / w)
    else:
        oh = crop_size
        ow = int(1.0 * oh * w / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    img = ImageOps.expand(img, border=(0,0, crop_size-ow, crop_size-oh), fill=0)

# コマンドライン引数を確認
if len(sys.argv) < 2:
    print('{} imagefile'.format(sys.argv[0]))
    exit(254)
imgpath = sys.argv[1]
if not os.path.isfile(imgpath):
    print('it is not actual file: {}'.format(imgpath))
    exit(254)

# 引数の画像読み込みと加工(と、別名での保存)
from PIL import Image, ImageOps
import torchvision.transforms as transform
img = Image.open(imgpath).convert('RGB')
resize(img)
img.save(os.path.splitext(imgpath)[0] + '.target.png')

# 画像をテンソルに変換する
input_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize([.485, .456, .406], [.229, .224, .225])
])
normalized_img = input_transform(img)
## 1つしかデータがないデータセットとして扱う(1要素配列のイメージ)
input_x = normalized_img.unsqueeze(0)

# モデルを読み込んで適用する
from encoding.models import get_segmentation_model
model = get_segmentation_model(
    'encnet',
    pretrained=False,
    dataset='pcontext',
    backbone='resnet101',
    jpu=True, lateral=False
    )

# モデルに学習結果を読み込む
checkpoint = torch.load(
   os.getcwd() + '/experiments/segmentation/runs/pcontext/encnet/encnet_res101_pcontext/checkpoint.pth.tar'
    )
model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(torch.load('/home/ubuntu/.encoding/models/encnet_resnet101_pcontext-9f27ea13.pth'))

## multi GPU専用のモデルになってしまっているので、こいつが必要
from encoding.parallel import DataParallelModel
model = DataParallelModel(model, device_ids=[0, 1]).cuda()
result = model(input_x)
output_y = result[0]
# auxloss = result[1]

# 結果テンソルをマスク画像の状態まで戻す
import encoding.utils as utils
# 画像1枚をデータセット化したので、[0]で取り出す。
pred = torch.max(output_y[0], 1)[1].cpu().numpy()
mask = utils.get_mask_pallete(pred, 'pcontext')
mask.save(os.path.splitext(imgpath)[0] + '.mask.png')
