from typing import Any, Tuple

import numpy as np
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset

from utils.transform import BatchSameTransform, get_trfm
from utils.loader import TestLoader


class AvCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None, transform=None, target_transform=None, batch_transform=False):
        super(AvCocoDetection, self).__init__(root, annFile, transform, target_transform, transforms)
        self.summary = summarizeDataset(self)
        self.batch_transform = batch_transform
        self.collate_fn = collate_fn_batch if batch_transform else None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        if self.batch_transform:
            image = self._load_image(id)
            target = self._load_target(id)
        else:
            transformed = self.transform(image=np.array(self._load_image(id)), target=self._load_target(id))
            image = transformed['image']
            target = transformed['target']

        return image, target


def collate_fn_batch(batch):
    # images = [i[0] for i in batch]
    # return images
    return batch


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


def get_dataset(root, annFile, type, args):
    if args.batch_trfm:
        print('Using batch transformation')

    dataset = AvCocoDetection(
        root=root,
        annFile=annFile,
        transform=BatchSameTransform(type, args) if args.batch_trfm else get_trfm(type, args),
        batch_transform=args.batch_trfm if type == 'train' else False
    )

    print('____________________')
    print('{}: {} images'.format(type, len(dataset)))
    location2Lap = dataset.summary['location2Lap']
    for location in location2Lap:
        num_lap = len(location2Lap[location]['cw'])
        print('{}: {} laps'.format(location, num_lap))
    print('\n')

    return dataset


def get_dataloader_val(dataset: AvCocoDetection):
    loaders = {}
    for location, laps in dataset.summary['location2Lap'].items():
        laps_landmark, laps_test = {}, {}
        directions = ['cw', 'ccw']
        for direction in directions:
            laps_landmark[direction] = laps[direction][0]
            laps_test[direction] = laps[direction][1]

        test_catIds = dataset.summary['lap2CatId'][laps_test['cw']] + \
                      dataset.summary['lap2CatId'][laps_test['ccw']]
        inds_test = []
        for catId in test_catIds:
            inds_test += dataset.coco.catToImgs[catId]
        dataset_test = Subset(dataset, inds_test)

        loaders[location] = {}
        loaders[location]['test'] = DataLoader(
            dataset=dataset_test, shuffle=False, batch_size=1, pin_memory=False, drop_last=False)
        loaders[location]['test'].catIds = test_catIds
        loaders[location]['landmark'] = TestLoader(dataset, laps_landmark)
    return loaders
