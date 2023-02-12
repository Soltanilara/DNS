import os
import os.path as osp
import random
from time import time, sleep
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from utils.loader import TestLoader
from utils.utils import get_imgId2landmarkId, gen_6_proto, plot_figure, per_landmark, str2bool
from utils.datasets import AvCocoDetection


def encode_one_side_landmark(model, landmarks, num_landmarks):

    indiv_protos = None
    indiv_eigs = None
    for i in range(16):
        landmarks_i = landmarks[i*60: (i+1)*60, :, :, :]
        indiv_protos_i = model.encoder(landmarks_i)
        indiv_protos = torch.cat([indiv_protos, indiv_protos_i], dim=0) if indiv_protos is not None else indiv_protos_i
        if model.cov:
            indiv_eigs_i = model.cov(indiv_protos_i) + 1e-8
            indiv_eigs = torch.cat([indiv_eigs, indiv_eigs_i], dim=0) if indiv_eigs is not None else indiv_eigs_i

    indiv_protos = indiv_protos.view([num_landmarks, 6, 10] + [indiv_protos.shape[-1]])
    proto_sup = torch.mean(indiv_protos, dim=2).squeeze()

    if model.cov:
        indiv_eigs = indiv_eigs.view([num_landmarks, 6, 10] + [indiv_eigs.shape[-1]])
        eigs_sup = torch.mean(indiv_eigs, dim=2).squeeze()
    else:
        eigs_sup = None

    return proto_sup, eigs_sup


def LoadModel_LandmarkStats(device, loader, model):
    """
    Returns the loaded model, as well as necessary statistics from the landmark classes.
    Path landmark image folders are assumed to be under root_dir/landmarks and named as 0, 1, 2, etc.

    :param device: device used to store variables and the model
    :param loader: dataloader for landmark
    :param model_path: name of the model to be loaded
    """

    landmarks = loader.get_all_landmarks()
    landmarks = gen_6_proto(landmarks)
    num_landmarks = landmarks.shape[0]

    landmarks = landmarks.view([-1] + list(landmarks.shape)[-3:]).cuda(device)

    proto_sup_l, eigs_sup_l = encode_one_side_landmark(model, landmarks[:, :, :, :224], num_landmarks)
    proto_sup_r, eigs_sup_r = encode_one_side_landmark(model, landmarks[:, :, :, 224:], num_landmarks)
    if model.cov:
        return model, torch.cat([proto_sup_l, proto_sup_r], dim=2), torch.cat([eigs_sup_l, eigs_sup_r], dim=2)
    else:
        return model, torch.cat([proto_sup_l, proto_sup_r], dim=2), None


def encode_one_side_match(model, image, device=0):
    qry_proto = model.encoder(image.cuda(device)).squeeze()
    qry_eigs = model.cov(qry_proto).squeeze() + 1e-8 if model.cov else None

    if len(qry_proto.shape) == 2:
        qry_proto = torch.mean(qry_proto, dim=0)
        qry_eigs = torch.mean(qry_eigs, dim=0) if model.cov else None

    return qry_proto, qry_eigs


def MatchDetector(model, image, lm_proto, lm_eigs, probabilities, threshold, device):
    """
    - Returns a landmark match/no match decision as "match" (boolean)
    - Updates the stored probability vector (similarities interpreted as probabilities) corresponding to the few recent
      individual images
    :param model: loaded model from LoadModel_LandmarkStats(...)
    :param image: incoming single image frame
    :param lm_proto: mean for the upcoming landmark, indexed from the second output of LoadModel_LandmarkStats(...)
    :param lm_eigs: covariance for the upcoming landmark, indexed from the third output of LoadModel_LandmarkStats(...)
    :param probabilities: current probability vector
    :param spread: spread parameter for similarity kernel
    """
    # qry_proto = model.encoder(image.cuda(device)).squeeze()
    # qry_eigs = model.cov(qry_proto).squeeze() + 1e-8
    # qry_proto = torch.mean(qry_proto, dim=0)
    # qry_eigs = torch.mean(qry_eigs, dim=0)

    qry_proto_l, qry_eigs_l = encode_one_side_match(model, image[:, :, :, :224], device)
    qry_proto_r, qry_eigs_r = encode_one_side_match(model, image[:, :, :, 224:], device)

    qry_proto = torch.cat([qry_proto_l, qry_proto_r], dim=0)

    diff = lm_proto - qry_proto

    if model.cov:
        qry_eigs = torch.cat([qry_eigs_l, qry_eigs_r], dim=0)
        dist = diff / (qry_eigs + lm_eigs) * diff
        if hasattr(model, 'norm_list') and model.norm_dist:
            norm_l = torch.sum(dist[:, :1000], dim=1, keepdim=True)
            norm_r = torch.sum(dist[:, 1000:], dim=1, keepdim=True)
            dist = torch.cat([norm_l, norm_r], dim=1)
    else:
        dist = diff ** 2

    prob = model.classifier(dist)
    prob = torch.max(prob).unsqueeze(dim=0)
    probabilities = torch.cat([probabilities[1:], prob], 0)
    if torch.mean(probabilities) > threshold:
        match = True
        # probabilities = torch.zeros(probabilities.size(), requires_grad=False).cuda(device)
    else:
        match = False
    return match, probabilities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Model name')
    parser.add_argument('-d', '--device', type=int, required=False, default=0,
                        help='device ID')
    parser.add_argument('-s', '--size', type=int, required=False, default=10,
                        help='size of the support and query set')
    parser.add_argument('-p', '--path_model', type=str, required=True,
                        help='path to the model')
    parser.add_argument('-fast', '--fast', type=str2bool, required=False, default=False,
                        help='only one lap will be the memory lap if True')
    args = parser.parse_args()
    model_name = args.name
    model_path = args.path_model

    ts = time()

    np.random.seed(0)
    torch.manual_seed(0)

    root_dataset = '/home/nick/dataset/dual_fisheye_indoor/PNG'
    dir_coco = 'coco/dual_fisheye/cross_test/3'
    # dir_coco = 'coco/dual_fisheye/cross_test/6/PNG'


    if not osp.exists(model_path):
        print('Waiting for model {}'.format(model_path))
    while not osp.exists(model_path):
        sleep(600)

    dir_log = osp.join('output', 'log', model_name)
    if not osp.exists(dir_log):
        os.makedirs(dir_log)

    # locations = ['ASB1F', 'ASB1F_New', 'ASB2F', 'Bainer2F', 'Ghausi2F', 'Ghausi2F_Lounge', 'Kemper3F', 'Math_outside', 'ASB_Outside', 'Facility_outside']
    # locations = ['ASB1F', 'ASB2F', 'Bainer2F', 'Ghausi2F', 'Ghausi2F_Lounge', 'Kemper3F']
    # locations = ['Bainer2F']
    # locations = ['Ghausi2F_Lounge', 'Bainer2F']
    locations = ['EnvironmentalScience1F', 'Kemper3F', 'WestVillageStudyHall']
    # locations = ['ASB1F']

    device = args.device
    channels, im_height, im_width = 3, 224, 448
    threshold = 0.5
    qry_size = args.size

    transform = A.Compose([
        A.Resize(height=im_height, width=im_width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    print('loading model: {}'.format(model_path))
    model = torch.load(model_path, map_location='cpu').cuda(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # fig_frame = plt.figure(figsize=(32, 8*len(locations)/2))
    # fig_average = plt.figure(figsize=(32, 8*len(locations)/2))
    # fig_ind = 1

    logs = {}
    for i, location in enumerate(locations):
        landmark_total = 0
        landmark_matched = 0
        runs_cnt = 0
        frame_probs = {}
        moving_avg_probs = {}
        test_lap_borders = {}
        dir_location = osp.join(dir_coco, location)
        laps = os.listdir(dir_location)
        print('[{}/{}] Testing: {}'.format(i+1, len(locations), location))
        if args.fast:
            laps_landmark = [laps[int(len(laps)/2)]]
            print('Choosing {} as the memory lap'.format(laps[0]))
        else:
            laps_landmark = laps

        for lap_landmark in laps_landmark:
            for lap_test in laps:
                if lap_test == lap_landmark:
                    continue
                runs_cnt += 1
                coco_path_test = osp.join(dir_location, lap_test)
                coco_path_landmark = osp.join(dir_location, lap_landmark)

                dataset_test = AvCocoDetection(
                    root=root_dataset,
                    annFile=coco_path_test,
                    transform=transform
                )
                dataset_landmark = AvCocoDetection(
                    root=root_dataset,
                    annFile=coco_path_landmark,
                    transform=transform
                )

                dataloader_landmark = TestLoader(dataset_landmark)

                model, proto_sup, eigs_sup = LoadModel_LandmarkStats(device, dataloader_landmark, model)

                num_landmark = len(dataloader_landmark.pos_catIds)
                matched = np.zeros(num_landmark)
                landmark = 0
                i = 0
                frame_prob = []
                moving_avg_prob = []
                tp, fn, tn, fp = 0, 0, 0, 0
                qry_imgs = None
                # TODO use window size between 3 and 6
                probabilities = torch.zeros(15, requires_grad=False).cuda(device)
                imgId2landmarkId, landmark_borders = get_imgId2landmarkId(dataset_test)
                dataloader = DataLoader(
                    dataset=dataset_test, shuffle=False, batch_size=1, pin_memory=False, drop_last=False)
                pbar = tqdm(dataloader)
                for i, img in enumerate(pbar):
                    if qry_imgs is None:
                        qry_imgs = img[0]
                        frame_prob += [0]
                        moving_avg_prob += [0]
                    elif qry_imgs.shape[0] < qry_size:
                        qry_imgs = torch.cat([qry_imgs, img[0]], dim=0)
                        frame_prob += [0]
                        moving_avg_prob += [0]
                    else:
                        qry_imgs = torch.cat([qry_imgs[1:], img[0]], dim=0)
                        landmark = imgId2landmarkId[i]
                        lm_proto = proto_sup[landmark, :]
                        lm_eigs = eigs_sup[landmark, :] if model.cov else None
                        match, probabilities = MatchDetector(
                            model, qry_imgs, lm_proto, lm_eigs, probabilities, threshold, device)
                        frame_prob += [probabilities[-1].cpu().item()]
                        moving_avg_prob.append(probabilities.mean().cpu().item())
                        tp, fn, tn, fp = per_landmark(moving_avg_prob, landmark_borders, threshold, tp, fn, tn, fp)

                        if match:
                            # print('\nUpdate to landmark ', str(landmark), ' at frame ', i)
                            matched[landmark] = 1
                            pbar.set_description('runs: [{}/{}], matched: [{}/{}]'.format(
                                runs_cnt, len(laps_landmark)*(len(laps)-1), int(np.sum(matched)), len(matched)))
                landmark_total += num_landmark
                landmark_matched += np.sum(matched)

                dir_output_location = osp.join(dir_log, 'figures', location)
                if not osp.exists(dir_output_location):
                    os.makedirs(dir_output_location)
                name_landmark = '_'.join(lap_landmark[:-5].split('_')[-2:])
                name_test = '_'.join(lap_test[:-5].split('_')[-2:])
                postfix = '{}_memory_{}_test_{}'.format(location, name_landmark, name_test)
                plot_figure(
                    location=location,
                    prob=frame_prob,
                    borders=landmark_borders,
                    xlabel='Frame number',
                    ylabel='Landmark frame probability',
                    path_save=osp.join(dir_output_location, 'landmark frame probability_'+postfix+'.png')
                )
                plot_figure(
                    location=location,
                    prob=moving_avg_prob,
                    borders=landmark_borders,
                    xlabel='Frame number',
                    ylabel='Moving average probability',
                    path_save=osp.join(dir_output_location, 'Moving average probability_'+postfix+'.png')
                )
                frame_probs[postfix] = frame_prob
                moving_avg_probs[postfix] = moving_avg_prob
                test_lap_borders[postfix] = landmark_borders

        landmark_matched = int(landmark_matched)
        landmark_total = int(landmark_total)
        log = '{}: matched [{}/{}], recognition rate {}'.format(location, landmark_matched, landmark_total,
                                                                landmark_matched/landmark_total)
        print(log)
        logs[location] = log
        with open(osp.join(dir_log, osp.basename(model_path)+'.json'), 'w') as f:
            json.dump(logs, f, indent=4)
        with open(osp.join(dir_log, 'plot_log_{}.json'.format(location)), 'w') as f:
            json.dump({
                'frame_probs': frame_probs,
                'moving_avg_probs': moving_avg_probs,
                'test_lap_borders': test_lap_borders,
            }, f, indent=4)

    print(time() - ts)

    # fig_frame.savefig(osp.join(dir_figure, 'landmark frame probability' + '.png'), dpi=600)
    # fig_average.savefig(osp.join(dir_figure, 'Moving average probability' + '.png'), dpi=600)
