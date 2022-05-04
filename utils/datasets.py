from typing import Any, Tuple

import numpy as np
from torchvision.datasets import CocoDetection

from utils.transform import BatchTransform


class AvCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None, transform=None, target_transform=None):
        super(AvCocoDetection, self).__init__(root, annFile, transform, target_transform, transforms)
        self.summary = summarizeDataset(self)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

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


def get_dataset(root, annFile, type):
    dataset = AvCocoDetection(
        root=root,
        annFile=annFile,
        transform=BatchTransform(type)
    )

    print('____________________')
    print('{}: {} images'.format(type, len(dataset)))
    location2Lap = dataset.summary['location2Lap']
    for location in location2Lap:
        num_lap = len(location2Lap[location]['cw'])
        print('{}: {} laps'.format(location, num_lap))
    print('\n')

    return dataset
