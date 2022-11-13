import os
import os.path as osp

import json
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # locations = ['Bainer2F', 'Ghausi2F_Lounge']
    locations = ['EnvironmentalScience1F', 'Kemper3F', 'WestVillageStudyHall']
    model_names = [
        # 'base',
        # 'r', 'j', 'd',
        # 'rj', 'rd', 'jd',
        # 'rjd'

        # 'j', 'size_1', 'size_3'

        # 'efficientnet-b0',
        # 'efficientnet-b1',
        # 'efficientnet-b2',
        # 'efficientnet-b3',
        # 'efficientnet-b4',

        # 'efficientnet-b0_size_10',
        # 'efficientnet-b1_size_10',
        # 'efficientnet-b2_size_10',

        # 'efficientnet-b4_size_10',
        # 'resnet50',
        # 'resnet50_swav',

        # 'resnet50_swav_new',
        # 'resnet50_qry_1',
        # 'efficientnet-b0_new',
        # 'efficientnet-b1_new',
        # 'efficientnet-b2_new',

        # 'resnet50_png',

        # '15locations',
        '4locations',

    ]
    for model_name in model_names:
        dict_threshold_result = {}
        path_save = osp.join('/home/nick/projects/FSL/temp/ROC', model_name+'.json')
        for location in locations:
            path_log = '/home/nick/projects/FSL/output/log/{}/plot_log_{}.json'.format(model_name, location)

            with open(path_log, 'r') as f:
                log = json.load(f)

            frame_probs = log['frame_probs']
            moving_avg_probs = log['moving_avg_probs']
            test_lap_borders = log['test_lap_borders']

            TPRs = []
            FPRs = []
            thresholds = [i/20 for i in range(0, 20)]
            dict_threshold_result[location] = {}
            for threshold in thresholds:
                TP_frame = 0
                TN_frame = 0
                FP_frame = 0
                FN_frame = 0
                TP_landmark = 0
                TN_landmark = 0
                FP_landmark = 0
                FN_landmark = 0
                for test_name, probs in moving_avg_probs.items():
                    prob_pos_list = []
                    prob_neg_list = []
                    last_landmark_end = 0
                    for i, landmark_start in enumerate(test_lap_borders[test_name]['start']):
                        prob_pos_start = max(0, landmark_start - 24, last_landmark_end)
                        prob_pos_end = landmark_start + 15 + 9
                        prob_neg_start = last_landmark_end
                        prob_neg_end = max(prob_pos_start, prob_neg_start)

                        prob_pos = probs[prob_pos_start: prob_pos_end]

                        last_landmark_end = prob_pos_end
                        prob_neg = probs[prob_neg_start: prob_neg_end]

                        prob_pos_list.append(prob_pos)
                        prob_neg_list.append(prob_neg)

                        if i == len(test_lap_borders[test_name]['start']) - 1:
                            prob_neg_list.append(probs[last_landmark_end:])

                        # Per frame
                        TP = sum([1 for i in prob_pos if i >= threshold])
                        TP_frame += TP
                        FN_frame += (len(prob_pos) - TP)

                        FP = sum([1 for i in prob_neg if i >= threshold])
                        FP_frame += FP
                        TN_frame += (len(prob_neg) - FP)


                        # Per landmark
                        if len(prob_pos) == 0 or max(prob_pos) >= threshold:
                            TP_landmark += 1
                        else:
                            FN_landmark += 1

                        if len(prob_neg) == 0 or max(prob_neg) >= threshold:
                            FP_landmark += 1
                        else:
                            TN_landmark += 1

                TPRs.append(TP_landmark / (TP_landmark + FN_landmark))
                FPRs.append(FP_landmark / (FP_landmark + TN_landmark))
                dict_threshold_result[location][threshold] = [
                    [TP_frame, FN_frame, TN_frame, FP_frame],
                    [TP_landmark, FN_landmark, TN_landmark, FP_landmark],
                ]

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
            plt.title('ROC Curve_{} ({})'.format(model_name, location))
            plt.xlabel('FP Rate')
            plt.ylabel('TP Rate')
            plt.show()

        with open(path_save, 'w') as f:
            json.dump(dict_threshold_result, f)
