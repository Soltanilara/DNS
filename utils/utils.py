import numpy as np
import torch
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import os.path as osp


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


def get_imgId2landmarkId(dataset: CocoDetection, specify_catIds=None):
    imgId2landmarkId = {}
    landmark_borders = {
        'start': [],
        'end': [],
    }
    landmark_id = -1
    if not specify_catIds:
        catIds = dataset.coco.getCatIds()
    else:
        catIds = specify_catIds
        gt = []

    if len(catIds) % 4 != 0:
        ignore_1st = len(catIds) // 4 * 2
        ignore_ids = [ignore_1st, ignore_1st + int(len(catIds) / 2)]

    for i, catId in enumerate(catIds):
        img_ids = sorted(dataset.coco.getImgIds(catIds=catId))

        if i == 0:
            start = 0
            end = len(img_ids)
        else:
            start = end
            end += len(img_ids)

        if i not in ignore_ids:
            if 'negative' in dataset.coco.loadCats(catId)[0]['name']:
                landmark_id += 1
            else:
                landmark_borders['start'].append(start)
                landmark_borders['end'].append(end)

        for img_id in img_ids:
            imgId2landmarkId[img_id] = landmark_id

        if specify_catIds:
            gt += [int('negative' not in dataset.coco.loadCats(catId)[0]['name'])] * len(img_ids)

    if specify_catIds:
        return imgId2landmarkId, landmark_borders, gt
    else:
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
    return v.lower() in ("yes", "y", "true", "t", "1")


def per_landmark(probs, borders, threshold, tp, fn, tn, fp):
    last_landmark_end = 0
    prob_pos_list = []
    prob_neg_list = []
    for i, start in enumerate(borders['start']):
        prob_pos_start = max(0, start - 24, last_landmark_end)
        prob_pos_end = start + 15 + 9
        prob_neg_start = last_landmark_end
        prob_neg_end = max(prob_pos_start, prob_neg_start)

        prob_pos = probs[prob_pos_start: prob_pos_end]

        last_landmark_end = prob_pos_end
        prob_neg = probs[prob_neg_start: prob_neg_end]

        prob_pos_list.append(prob_pos)
        prob_neg_list.append(prob_neg)

        if i == len(borders['start']) - 1:
            prob_neg_list.append(probs[last_landmark_end:])

        # Per landmark
        if len(prob_pos) == 0 or max(prob_pos) >= threshold:
            tp += 1
        else:
            fn += 1

        if len(prob_neg) == 0 or max(prob_neg) >= threshold:
            fp += 1
        else:
            tn += 1

    return tp, fn, tn, fp


def plot_figure(location, prob, borders, xlabel, ylabel, path_save=None, show=None):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    # ax = fig_frame.add_subplot(len(locations), 2, fig_ind)
    # fig_ind += 1
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.plot(prob)
    ax.grid()
    plt.title(location)
    plt.vlines(borders['start'], 0, 1, 'g', 'dashed')
    plt.vlines(borders['end'], 0, 1, 'r', 'dashed')
    plt.ylim(0, 1)
    plt.xlim(left=0)
    if show:
        plt.show()
        pass
    if path_save:
        plt.savefig(osp.join(path_save), dpi=300)
    plt.close()
