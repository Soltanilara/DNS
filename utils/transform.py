import random

import numpy as np
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_trfm(type, args):
    # if type == 'train':
    #     return transforms.Compose([
    #         transforms.Resize((224, 448)),
    #         transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
    #         transforms.RandomApply([transforms.ColorJitter(brightness=0.5, hue=0.2)], p=0.5),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.2, 5), p=0.5),
    #     ])
    # if type == 'val':
    #     return transforms.Compose([
    #         transforms.Resize((224, 448)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    if type == 'train':
        trfms = [A.Resize(height=224, width=448)]
        if args.Rotate:
            trfms.append(A.Rotate(limit=10))
        if args.ColorJitter:
            trfms.append(A.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5))
        if args.CoarseDropout:
            trfms.append(
                A.CoarseDropout(max_holes=4, min_holes=1, max_height=224, max_width=112, min_height=20, min_width=20,
                                p=0.75))
        trfms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()])

        return A.Compose(trfms)
        # return A.Compose([
        #     A.Resize(height=224, width=448),
        #     A.Rotate(limit=10),
        #     A.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5),
        #     A.CoarseDropout(max_holes=4, min_holes=1, max_height=224, max_width=112, min_height=20, min_width=20,
        #                     p=0.75),
        #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ToTensorV2(),
        # ])
    if type == 'val':
        return A.Compose([
            A.Resize(height=224, width=448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class BatchTransform:
    def __init__(self, type):
        self.type = type
        self.param_range = self.get_param_range(type)

    def get_param_range(self, type):
        if type == 'train':
            param_range = {
                'Resize': {'height': 224, 'width': 448},
                'Rotate': {'limit': 10},
                'ColorJitter': {'brightness': 0.5, 'hue': 0.5, 'contrast': 0.5},
                'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                'CoarseDropout': {'max_holes': 4, 'min_holes': 1, 'max_height': 224, 'min_height': 20,
                                  'max_width': 112, 'min_width': 20, 'p': 0.75}
            }
        elif type == 'val':
            param_range = {
                'Resize': {'height': 224, 'width': 448},
                'Normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            }
        return param_range

    def get_param(self):
        self.param = {}
        for trfm, params in self.param_range.items():
            self.param[trfm] = {}
            if trfm in ['Resize', 'Normalize']:
                self.param[trfm] = self.param_range[trfm]
            elif trfm in ['Rotate']:
                for param in params:
                    limit = self.param_range[trfm][param]
                    self.param[trfm][param] = random.uniform(-limit, limit)
            elif trfm in ['CoarseDropout']:
                self.param[trfm]['holes'] = random.randint(
                    self.param_range[trfm]['min_holes'], self.param_range[trfm]['max_holes'])
                self.param[trfm]['height'] = random.randint(
                    self.param_range[trfm]['min_height'], self.param_range[trfm]['max_height'])
                self.param[trfm]['width'] = random.randint(
                    self.param_range[trfm]['min_width'], self.param_range[trfm]['max_width'])
            else:
                for param in params:
                    limit = self.param_range[trfm][param]
                    if param == 'hue':
                        self.param[trfm][param] = random.uniform(-limit, limit)
                    else:
                        self.param[trfm][param] = random.uniform(max(0, 1 - limit), 1 + limit)
        return self.param

    def get_always_apply(self):
        always_apply = {}
        for trfm in self.param_range:
            if trfm in ['Resize', 'Normalize']:
                always_apply[trfm] = True
            else:
                p = self.param_range[trfm]['p'] if 'p' in self.param_range[trfm].keys() else 0.5
                always_apply[trfm] = random.random() < p
        return always_apply

    def apply(self, batch):
        param = self.get_param()
        always_apply = self.get_always_apply()
        if self.type == 'train':
            transform = A.Compose([
                A.Resize(height=param['Resize']['height'], width=param['Resize']['width']),
                # A.Rotate(
                #     limit=(param['Rotate']['limit'], param['Rotate']['limit']), always_apply=always_apply['Rotate']),
                # A.ColorJitter(
                #     brightness=(param['ColorJitter']['brightness'], param['ColorJitter']['brightness']),
                #     contrast=(param['ColorJitter']['contrast'], param['ColorJitter']['contrast']),
                #     hue=(param['ColorJitter']['hue'], param['ColorJitter']['hue']),
                #     always_apply=always_apply['ColorJitter']
                # ),
                A.CoarseDropout(max_holes=param['CoarseDropout']['holes'], min_holes=param['CoarseDropout']['holes'],
                                max_height=param['CoarseDropout']['height'], min_height=param['CoarseDropout']['height'],
                                max_width=param['CoarseDropout']['width'], min_width=param['CoarseDropout']['width'],
                                always_apply=always_apply['CoarseDropout']),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        elif self.type == 'val':
            transform = A.Compose([
                A.Resize(height=param['Resize']['height'], width=param['Resize']['width']),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        img_aug = [transform(image=np.array(s[0]), target=s[1])['image'].unsqueeze(dim=0) for s in batch]
        img_aug = torch.cat(img_aug, dim=0)
        return img_aug
