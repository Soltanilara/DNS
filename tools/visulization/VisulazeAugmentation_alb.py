import os
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from glob import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.path_imgs = glob(osp.join(root, '*.jpg'))

    def __getitem__(self, idx):
        # return Image.open(self.path_imgs[idx])
        return cv2.imread(self.path_imgs[idx])

    def __len__(self):
        return len(self.path_imgs)


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        return tuple(zip(*batch))
    else:
        return tuple(batch)


if __name__ == '__main__':
    # root_imgs = '/Users/shidebo/dataset/AV/Sorted/ASB1F/ccw/210830_181331_mav_320x240ASB1F/000_negative'
    root_imgs = '/Volumes/dataset/av/dual_fisheye/sorted/Kemper3F/ccw/220311_224215_mav_320x240_Kemper3F_ccw/030-negative'

    trfm = A.Compose([
        A.Crop(x_min=0, y_min=0, x_max=520, y_max=192, always_apply=True),
        A.Rotate(limit=10),
        A.ColorJitter(),
        ToTensorV2(),
    ])

    # trfm = transforms.Compose([
    #     # transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
    #     # transforms.CenterCrop(size=224),
    #     # transforms.RandomEqualize(),
    #     transforms.ToTensor(),
    #     transforms.RandomApply([transforms.RandomErasing(scale=(0.05, 0.2), ratio=(0.2, 5))], p=1),
    # ])

    dataset = TestDataset(root=root_imgs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for i in loader:
        img_raw = i[0]

        img_aug = trfm(image=img_raw)
        img_aug = img_aug['image'].permute([1, 2, 0]).numpy()
        # img_aug = trfm(img_raw)
        # img_aug = img_aug.permute([1, 2, 0]).numpy()

        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(img_raw)
        plt.title('Original')
        plt.axis('off')
        fig.add_subplot(122)
        plt.imshow(img_aug)
        plt.title('Augmented')
        plt.axis('off')

        plt.show()
        pass
