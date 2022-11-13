from matplotlib import pyplot as plt
import json


def hist_probs_all(log):
    probs_list = list(log['moving_avg_probs'].values())

    probs_truncated = []
    cnt, cnt_all = 0, 0
    for prob in probs_list:
        for p in prob:
            if p <= 0.35:
                cnt += 1

            if p > 0.1:
                probs_truncated.append(p)

    plt.figure()
    plt.hist(probs_truncated, bins=50)
    plt.show()


if __name__ == '__main__':
    path_logs = [
        '/Users/shidebo/Downloads/workstation/dual_fisheye/exclude_Ghausi2F_Lounge_Kemper3F/resnet50/plot_log_Ghausi2F_Lounge.json',
        '/Users/shidebo/Downloads/workstation/dual_fisheye/exclude_Ghausi2F_Lounge_Kemper3F/resnet50/plot_log_Bainer2F.json',
    ]

    for path_log in path_logs:
        with open(path_log, 'r') as f:
            log = json.load(f)

        # Plot the distribution of all the probs
        hist_probs_all(log)

        # Plot the distribution of probs in negative and positive sample separately
        prob_pos_list, prob_neg_list = [], []

        for test_name in log['test_lap_borders'].keys():
            probs = log['moving_avg_probs'][test_name]
            last_landmark_end = 0

            for i, landmark_start in enumerate(log['test_lap_borders'][test_name]['start']):
                prob_pos_start = max(0, landmark_start - 24, last_landmark_end)
                prob_pos_end = landmark_start + 15 + 9
                prob_neg_start = last_landmark_end
                prob_neg_end = max(prob_pos_start, prob_neg_start)

                prob_pos = probs[prob_pos_start: prob_pos_end]

                last_landmark_end = prob_pos_end
                prob_neg = probs[prob_neg_start: prob_neg_end]

                prob_pos_list.append(prob_pos)
                prob_neg_list.append(prob_neg)

                if i == len(log['test_lap_borders'][test_name]['start']) - 1:
                    prob_neg_list.append(probs[last_landmark_end:])

        plt.figure(figsize=(5, 8), tight_layout=True)
        for i, (prob_list, title) in enumerate(zip([prob_neg_list, prob_pos_list], ['Negative', 'Positive'])):
            probs = []
            for l in prob_list:
                probs.extend(l)

            plt.subplot(2, 1, i+1)
            plt.hist(probs, bins=20)
            plt.title(title)
            plt.xlabel('Probability')
            plt.ylabel('Counts')
        plt.show()

