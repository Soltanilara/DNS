import os.path as osp

import numpy as np
import json
from matplotlib import pyplot as plt

from utils.utils import per_landmark, simple_moving_average


def calculate_auroc(tpr, fpr):
    tpr_dict = {}
    for i in range(len(fpr)):
        key = fpr[i]
        value = tpr[i]
        if key in tpr_dict:
            tpr_dict[key].append(value)
        else:
            tpr_dict[key] = [value]
    unique_fpr = sorted(tpr_dict.keys())
    avg_tpr = [np.mean(tpr_dict[key]) for key in unique_fpr]
    auroc = np.trapz(avg_tpr, unique_fpr)

    return auroc

if __name__ == '__main__':

    dir_log = 'output/log/'
    names = [
        '12_3_3_swav_best',
    ]
    locations = ['ASB1F', 'WestVillageStudyHall', 'EnvironmentalScience1F']

    for name in names:
        dir_name = osp.join(dir_log, name)

        for location in locations:
            fname = 'plot_log_{}.json'.format(location)
            path_location = osp.join(dir_name, fname)
            TPRs = []
            FPRs = []
            thresholds = [i / 20 for i in range(0, 21)]
            for threshold in thresholds:
                tp, fn, tn, fp = 0, 0, 0, 0
                with open(path_location, 'r') as f:
                    log = json.load(f)
                    for lap, probs in log['frame_probs'].items():
                        probs = simple_moving_average(probs, 10)
                        landmark_borders = log['test_lap_borders'][lap]
                        tp, fn, tn, fp = per_landmark(probs, landmark_borders, threshold, tp, fn, tn, fp)

                TPRs.append(tp / (tp+fn+1e-3))
                FPRs.append(fp / (fp+tn+1e-3))

            x = [i for i in FPRs[::2]]
            y = [i for i in TPRs[::2]]
            thr = [i for i in thresholds[::2]]
            fig, ax = plt.subplots()
            plt.plot(FPRs, TPRs)
            ax.scatter(x, y)
            for i, thr in enumerate(thr):
                ax.annotate(str(thr), (x[i], y[i]))

            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.title('ROC Curve_{} ({})'.format(name, location))
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.show()

            print('AUROC ({}): '.format(location),calculate_auroc(TPRs,FPRs))
