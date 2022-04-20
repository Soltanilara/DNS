from typing import Any, Tuple

import numpy as np
from torchvision.datasets import CocoDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AvCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None, transform=None, target_transform=None):
        super(AvCocoDetection, self).__init__(root, annFile, transform, target_transform, transforms)
        self.summary = summarizeDataset(self)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            result = self.transform(image=np.array(image), target=target)
            image = result['image']
            target = result['target']
            # image, target = self.transform(image=np.array(image), target=target)

        return image, target


def summarizeDataset(dataset: CocoDetection):
    lap2CatId = {}
    location2Lap = {}
    PN2CatId = {
        'positive': [],
        'negative': [],
    }
    catId2Prob = {}
    cats = dataset.coco.loadCats(dataset.coco.getCatIds())
    for cat in cats:
        catId = cat['id']
        catName = cat['name']
        lap = cat['lap']
        direction = cat['direction']
        location = cat['location']
        if lap not in lap2CatId:
            lap2CatId[lap] = [catId]
        else:
            lap2CatId[lap].append(catId)

        if 'negative' in catName:
            PN2CatId['negative'].append(catId)
        else:
            PN2CatId['positive'].append(catId)

        if location not in location2Lap:
            location2Lap[location] = {
                'cw': [],
                'ccw': [],
            }

        if lap not in location2Lap[location][direction]:
            location2Lap[location][direction].append(lap)

        num_img = len(dataset.coco.getImgIds(catIds=catId))
        prob = [i+1 for i in range(num_img)]
        prob_sum = np.sum(prob)
        prob /= prob_sum
        catId2Prob[catId] = prob

        summary = {
            'lap2CatId': lap2CatId,
            'PN2CatId': PN2CatId,
            'location2Lap': location2Lap,
            'catId2Prob': catId2Prob
        }
    return summary


def get_trfm(type):
    # transform_train = transforms.Compose([
    #     transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
    #     transforms.RandomApply([transforms.ColorJitter(brightness=0.5, hue=0.2)], p=0.5),
    #     transforms.Resize((224, 448)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.2, 5), p=0.5),
    # ])
    # transform_val = transforms.Compose([
    #     transforms.Resize((224, 448)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    if type == 'train':
        return A.Compose([
            A.Resize(height=224, width=448),
            A.CoarseDropout(max_holes=4, min_holes=1, max_height=224, max_width=112, min_height=20, min_width=20, p=0.75),
            A.Rotate(limit=10),
            A.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if type == 'val':
        return A.Compose([
            A.Resize(height=224, width=448),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_dataset(root, annFile, type):
    trfm = get_trfm(type)
    dataset = AvCocoDetection(
        root=root,
        annFile=annFile,
        transform=trfm
    )

    print('____________________')
    print('{}: {} images'.format(type, len(dataset)))
    location2Lap = dataset.summary['location2Lap']
    for location in location2Lap:
        num_lap = len(location2Lap[location]['cw'])
        print('{}: {} laps'.format(location, num_lap))
    print('\n')

    return dataset
