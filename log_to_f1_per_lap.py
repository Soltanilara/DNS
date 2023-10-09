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
    results_per_lap = {}
    for name in names:
        results_per_lap[name] = {}
        dir_name = osp.join(dir_log, name)
        for location in locations:
            results_per_lap[name][location] = {}
            fname = 'plot_log_{}.json'.format(location)
            path_location = osp.join(dir_name, fname)

            with open(path_location, 'r') as f:
                log = json.load(f)
                for lap, probs in log['frame_probs'].items():
                    probs = simple_moving_average(probs, 10)
                    landmark_borders = log['test_lap_borders'][lap]
                    tp, fn, tn, fp = per_landmark(probs, landmark_borders, threshold, 0, 0, 0, 0)

                    precision = tp / (tp + fp + 1e-3)
                    recall = tp / (tp + fn + 1e-3)
                    f1 = 2 * recall * precision / (recall + precision)

                    lap_memory = lap[lap.index('memory') + 7: lap.index('memory') + 20]
                    lap_test = lap[lap.index('test') + 5:]
                    if lap_memory not in results_per_lap[name][location]:
                        results_per_lap[name][location][lap_memory] = {lap_test: f1}
                    else:
                        results_per_lap[name][location][lap_memory][lap_test] = f1

    for model, locations in results_per_lap.items():
        print('\n\n============================================')
        print('\n\nmodel: {}'.format(model))

        for location in locations:
            print('\nlocation: {}'.format(location))
            for i, (memory, tests) in enumerate(results_per_lap[model][location].items()):
                if i == 0:
                    print('memory\\test\t\t{}'.format('\t'.join(list(results_per_lap[model][location].keys()))))
                line = ''

                line += ('{}\t'.format(memory))

                for j, test in enumerate(results_per_lap[model][location][memory].items()):
                    if i == j:
                        line += ('\t{}'.format('  -    '))
                    line += ('\t{:.3f}'.format(results_per_lap[model][location][memory][test[0]]))

                if i == len(results_per_lap[model][location][memory]):
                    line += ('\t{}'.format('  -    '))
                print(line)
