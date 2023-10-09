import json
import os.path as osp

from utils.utils import per_landmark, simple_moving_average


if __name__ == '__main__':
    dir_log = 'output/log/'
    names = [
        '12_3_3_swav_best',
    ]
    locations = ['ASB1F', 'WestVillageStudyHall', 'EnvironmentalScience1F']
    threshold = 0.5

    for name in names:
        dir_name = osp.join(dir_log, name)
        tp, fn, tn, fp = 0, 0, 0, 0
        for location in locations:
            fname = 'plot_log_{}.json'.format(location)
            path_location = osp.join(dir_name, fname)

            with open(path_location, 'r') as f:
                log = json.load(f)
                for lap, probs in log['frame_probs'].items():
                    # probs = exponential_moving_average(probs, 10)
                    probs = simple_moving_average(probs, 10)
                    landmark_borders = log['test_lap_borders'][lap]
                    tp, fn, tn, fp = per_landmark(probs, landmark_borders, threshold, tp, fn, tn, fp)

        precision = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('Model: {}, F1 = {:.3f}'.format(name, f1))
