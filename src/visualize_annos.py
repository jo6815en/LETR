from argparse import Namespace

import torch
import matplotlib.pyplot as plt
from einops import rearrange

from datasets.coco import build

args = Namespace(
    coco_path="/Users/s0000960/data/forestseg_lth/forestseg_data/datasets/coco_style",
    eval=False,
)
dataset = build("train", args=args)

for img, anno in dataset:

    img = rearrange(img, 'c h w -> h w c')
    mean = torch.tensor([0.538, 0.494, 0.453])[None, None, :]
    std_dev = torch.tensor([0.257, 0.263, 0.273])[None, None, :]
    img = img*std_dev + mean

    img_h, img_w = img.shape[:2]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h, img_w, img_h, img_w, img_h])
    lines = anno['lines'] * scale_fct[None, :]
    lines = lines.flip([-1])  # this is yxyx format

    plt.axis('off')
    fig = plt.figure()
    plt.imshow(img)
    for line in lines:
        y1, x1, y2, x2, y3, x3, y4, x4 = line  # this is yxyx
        p1 = (x1, y1)
        p2 = (x2, y2)
        p3 = (x3, y3)
        p4 = (x4, y4)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1.5, color='darkorange', zorder=1)
        plt.plot([p3[0], p4[0]], [p3[1], p4[1]], linewidth=1.5, color='blue', zorder=1)
    plt.show()
    # plt.savefig('hej.png')
