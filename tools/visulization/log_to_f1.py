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
        'f1_base',
        'f1_skip_cov',
        'f1_swav',
        'f1_swav_skip_cov',
        'f1_scratch',
    ]
    locations = ['WestVillageStudyHall', 'Kemper3F', 'EnvironmentalScience1F']
    # locations = ['WestVillageStudyHall', 'Kemper3F']
    threshold = 0.4

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
        # f1 = 2 * recall * precision / (recall + precision)
        # print('Model: {}, F1 = {:.3f}'.format(name, f1))

        # todo: false positive rate
        fpr = fp / (fp + tn)
        print('Model: {}, FPR = {:.3f}'.format(name, fpr))

