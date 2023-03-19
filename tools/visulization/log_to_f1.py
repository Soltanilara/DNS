import json
import os
import os.path as osp
import numpy as np

from utils.utils import per_landmark


def exponential_moving_average(probs_frame, window_size, smoothing=2):
    probs_emv = []
    mul = smoothing / (1 + window_size)
    for i, p in enumerate(probs_frame):
        if i < window_size - 1:
            if i == 0:
                emv_last = 0
            m = smoothing / (1 + i)
        else:
            m = mul
        p_emv = p * m + emv_last * (1 - mul)
        probs_emv.append(p_emv)
        emv_last = p_emv
    return probs_emv


if __name__ == '__main__':
    dir_log = '/home/nick/projects/FSL/output/log/'
    names = [
        # 'f1_base',
        # 'f1_skip_cov',
        # 'f1_swav',
        # 'f1_swav_skip_cov',
        # 'f1_scratch',

        # 'swav',
        # 'swav_skip',
        # 'resnet50',
        # 'resnet50_skip',
        # # 'scratch',
        # 'f1_scratch',

        # 'swav_new',
        # 'swav_epoch_2',
        # 'swav_epoch_10',
        # 'swav_epoch_18',
        # 'swav_epoch_26',
        # 'swav_epoch_34',
        # 'swav_epoch_40',

        # 'swav_no_aug_best',
        # 'swav_no_aug_epoch_2',
        # 'swav_no_aug_epoch_10',
        # 'swav_no_aug_epoch_18',
        # 'swav_no_aug_epoch_26',
        # 'swav_no_aug_epoch_34',
        # 'swav_no_aug_epoch_40',

        # 'swav_single_trfm_best',
        # 'swav_single_trfm_epoch_2',
        # 'swav_single_trfm_epoch_10',
        # 'swav_single_trfm_epoch_18',
        # 'swav_single_trfm_epoch_26',
        # 'swav_single_trfm_epoch_34',
        # 'swav_single_trfm_epoch_40',
        #
        # 'swav_skip_cov_single_trfm_best',
        # 'swav_skip_cov_single_trfm_epoch_2',
        # 'swav_skip_cov_single_trfm_epoch_10',
        # 'swav_skip_cov_single_trfm_epoch_18',
        # 'swav_skip_cov_single_trfm_epoch_26',
        # 'swav_skip_cov_single_trfm_epoch_34',
        # 'swav_skip_cov_single_trfm_epoch_40',
        #
        # 'swav_single_trfm_aug_light_best',
        # 'swav_single_trfm_aug_light_epoch_2',
        # 'swav_single_trfm_aug_light_epoch_10',
        # 'swav_single_trfm_aug_light_epoch_18',
        # 'swav_single_trfm_aug_light_epoch_26',
        # 'swav_single_trfm_aug_light_epoch_34',
        # 'swav_single_trfm_aug_light_epoch_40',
        #
        # '15_swav_batch_trfm_best',
        # '15_swav_batch_trfm_epoch_2',
        # '15_swav_batch_trfm_epoch_10',
        # '15_swav_batch_trfm_epoch_18',
        # '15_swav_batch_trfm_epoch_26',
        # '15_swav_batch_trfm_epoch_34',
        # '15_swav_batch_trfm_epoch_40',
        #
        # '15_swav_skip_cov_batch_trfm_best',
        # '15_swav_skip_cov_batch_trfm_epoch_2',
        # '15_swav_skip_cov_batch_trfm_epoch_10',
        # '15_swav_skip_cov_batch_trfm_epoch_18',
        # '15_swav_skip_cov_batch_trfm_epoch_26',
        # '15_swav_skip_cov_batch_trfm_epoch_34',
        # '15_swav_skip_cov_batch_trfm_epoch_40',
        #
        # '15_resnet50_batch_trfm_best',
        # '15_resnet50_batch_trfm_epoch_2',
        # '15_resnet50_batch_trfm_epoch_10',
        # '15_resnet50_batch_trfm_epoch_18',
        # '15_resnet50_batch_trfm_epoch_26',
        # '15_resnet50_batch_trfm_epoch_34',
        # '15_resnet50_batch_trfm_epoch_40',
        #
        # '15_resnet50_skip_cov_batch_trfm_best',
        # '15_resnet50_skip_cov_batch_trfm_epoch_2',
        # '15_resnet50_skip_cov_batch_trfm_epoch_10',
        # '15_resnet50_skip_cov_batch_trfm_epoch_18',
        # '15_resnet50_skip_cov_batch_trfm_epoch_26',
        # '15_resnet50_skip_cov_batch_trfm_epoch_34',
        # '15_resnet50_skip_cov_batch_trfm_epoch_40',

        # '11_swav_batch_trfm_best',
        # '11_swav_batch_trfm_epoch_2',
        # '11_swav_batch_trfm_epoch_10',
        # '11_swav_batch_trfm_epoch_18',
        # '11_swav_batch_trfm_epoch_26',
        # '11_swav_batch_trfm_epoch_34',
        # '11_swav_batch_trfm_epoch_40',

        '12_3_3_swav_best',
        '12_3_3_swav_skip_cov_best',
        '12_3_3_resnet50_best',
        '12_3_3_resnet50_skip_cov_best',
        '12_3_3_scratch_best',

        # '12_3_3_swav_skip_cov_epoch_2',
        # '12_3_3_swav_skip_cov_epoch_10',
        # '12_3_3_swav_skip_cov_epoch_18',
        # '12_3_3_swav_skip_cov_epoch_26',
        # '12_3_3_swav_skip_cov_epoch_34',
        # '12_3_3_swav_skip_cov_epoch_40',
    ]
    # locations = ['WestVillageStudyHall', 'Kemper3F', 'EnvironmentalScience1F']
    # locations = ['WestVillageStudyHall', 'Kemper3F']
    # locations = ['WestVillageStudyHall', 'EnvironmentalScience1F', 'ASB1F', 'PhysicsBuilding', 'WestVillageOffice']
    locations = ['ASB1F', 'WestVillageStudyHall', 'EnvironmentalScience1F']
    threshold = 0.3

    for name in names:
        dir_name = osp.join(dir_log, name)
        tp, fn, tn, fp = 0, 0, 0, 0
        for location in locations:
            fname = 'plot_log_{}.json'.format(location)
            path_location = osp.join(dir_name, fname)

            with open(path_location, 'r') as f:
                log = json.load(f)
                # for lap, probs in log['moving_avg_probs'].items():
                for lap, probs in log['frame_probs'].items():
                    probs = exponential_moving_average(probs, 10)
                    landmark_borders = log['test_lap_borders'][lap]
                    tp, fn, tn, fp = per_landmark(probs, landmark_borders, threshold, tp, fn, tn, fp)

        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        f1 = 2 * recall * precision / (recall + precision)
        print('Model: {}, F1 = {:.3f}'.format(name, f1))

        # fpr = fp / (fp + tn)
        # print('Model: {}, FPR = {:.3f}'.format(name, fpr))

