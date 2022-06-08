import os
import os.path as osp

import json
from matplotlib import pyplot as plt


if __name__ == '__main__':
    locations = ['Ghausi2F_Lounge', 'Kemper3F']
    # locations = ['Ghausi2F_Lounge']
    for location in locations:
        path_log = '/Users/shidebo/Downloads/workstation/dual_fisheye/exclude_Ghausi2F_Lounge_Kemper3F/strong_aug/plot_log_{}.json'.format(location)

        with open(path_log, 'r') as f:
            log = json.load(f)

        frame_probs = log['frame_probs']
        moving_avg_probs = log['moving_avg_probs']
        test_lap_borders = log['test_lap_borders']

        TPRs = []
        FPRs = []
        thresholds = [i/100 for i in range(50, 100)]
        for threshold in thresholds:
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for test_name, probs in moving_avg_probs.items():
                pos = []
                for i, start in enumerate(test_lap_borders[test_name]['start']):
                    pos.extend([i for i in range(max(0, start-24), start+15+9)])

                recognized = False
                misrecognized = False
                for i, p in enumerate(probs):
                    if i in pos:
                        misrecognized = False
                        if p >= threshold and not recognized:
                            TP += 1
                            recognized = True
                    else:
                        if p >= threshold and not misrecognized:
                            FP += 1
                            misrecognized = True
                        recognized = False

            TPRs.append(TP / (17 * len(moving_avg_probs)))
            FPRs.append(FP / (17 * len(moving_avg_probs)))


        x = [i for i in FPRs[::10]]
        y = [i for i in TPRs[::10]]
        thr = [i for i in thresholds[::10]]
        fig, ax = plt.subplots()
        plt.plot(FPRs, TPRs)
        ax.scatter(x, y)
        for i, thr in enumerate(thr):
            ax.annotate(str(thr), (x[i], y[i]))

        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.title('ROC Curve ({})'.format(location))
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.show()

