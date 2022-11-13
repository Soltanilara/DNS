import numpy as np
import torch
from torchvision.datasets import CocoDetection


def sortImgs(dataset):
    imgs = dataset.img
    targets = dataset.targets
    start_inds = [0]
    imgs_new = []
    targets_new = []
    cls = 0

    for i in range(len(targets) - 1):
        if targets[i] != targets[i + 1]:
            start_inds.append(i + 1)

    start_inds.append(len(targets))

    for i in range(0, len(start_inds) - 1, 2):
        if imgs[start_inds[i]][0].split('/')[6][3:] != 'negative':
            targets_new.extend([cls] * (start_inds[i + 1] - start_inds[i]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i]: start_inds[i + 1]]])
            cls += 1
            targets_new.extend([cls] * (start_inds[i + 2] - start_inds[i + 1]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i + 1]: start_inds[i + 2]]])
            cls += 1
        else:
            targets_new.extend([cls] * (start_inds[i + 2] - start_inds[i + 1]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i + 1]: start_inds[i + 2]]])
            cls += 1
            targets_new.extend([cls] * (start_inds[i + 1] - start_inds[i]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i]: start_inds[i + 1]]])
            cls += 1

    dataset.samples = imgs_new
    dataset.img = imgs_new
    dataset.targets = targets_new


def get_imgId2landmarkId(dataset: CocoDetection):
    imgId2landmarkId = {}
    landmark_borders = {
        'start': [],
        'end': [],
    }
    landmark_id = -1
    catIds = dataset.coco.getCatIds()

    if len(catIds) % 4 != 0:
        ignore_1st = len(catIds) // 4 * 2
        ignore_ids = [ignore_1st, ignore_1st + int(len(catIds) / 2)]

    for catId in catIds:
        img_ids = sorted(dataset.coco.getImgIds(catIds=catId))

        if catId not in ignore_ids:
            if 'negative' in dataset.coco.loadCats(catId)[0]['name']:
                landmark_id += 1
            else:
                landmark_borders['start'].append(img_ids[0])
                landmark_borders['end'].append(img_ids[-1])

        for img_id in img_ids:
            imgId2landmarkId[img_id] = landmark_id

    return imgId2landmarkId, landmark_borders


def gen_6_proto(landmarks):
    landmarks_all = None
    for i_landmark in range(landmarks.shape[0]):
        landmarks_6 = None
        for l in range(6):
            window = landmarks[i_landmark, l: l+10, :, :, :]
            window = window.unsqueeze(dim=0)
            landmarks_6 = window if landmarks_6 is None else torch.cat([landmarks_6, window])
        landmarks_6 = landmarks_6.unsqueeze(dim=0)
        landmarks_all = landmarks_6 if landmarks_all is None else torch.cat([landmarks_all, landmarks_6])
    return landmarks_all


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

