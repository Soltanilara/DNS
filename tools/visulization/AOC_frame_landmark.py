import os
import os.path as osp

import matplotlib.pyplot as plt
import json


# dir_log = '/home/nick/projects/FSL/temp/ROC'
dir_log = '/Users/shidebo/SynologyDrive/Projects/AV/code/temp/ROC'

dict_model_name_diaplay_name = {}
model_names = [
    # 'base',
    # 'r', 'j', 'd',
    # 'rj', 'rd', 'jd',
    # 'rjd'

    # 'j', 'size_1', 'size_3',

    # 'efficientnet-b0',
    # 'efficientnet-b1',
    # 'efficientnet-b2',
    # 'efficientnet-b3',
    # 'efficientnet-b4',

    # 'efficientnet-b0_size_10',
    # 'efficientnet-b1_size_10',
    # 'efficientnet-b2_size_10',

    # 'resnet50',
    # 'resnet50_swav',
    # 'resnet50_swav_new',
    # 'resnet50_qry_1',
    # 'efficientnet-b0_new',
    # 'efficientnet-b1_new',
    # 'efficientnet-b4_size_10',
    # 'j',
    'resnet50_png'
    # '15locations',
    # '4locations'
]
type = 'Per Landmark'
# type = 'Per Frame'

AOC = {}
F1 = {}
tp_fn_tn_fp = {}
for model_name in model_names:
    name_display = ''
    # if 'base' in model_name:
    #     name_display += 'base'
    # if 'r' in model_name:
    #     name_display += 'Rotate'
    if 'j' in model_name:
        if name_display != '':
            name_display += '+'
        # name_display += 'ColorJitter'
        name_display += 'resnet50_swav'
    # if 'd' in model_name:
    #     if name_display != '':
    #         name_display += '+'
    #     name_display += 'CoarseDropout'
    # if model_name == 'size_1':
    #     name_display = 'resnet-50_size_1'
    # if model_name == 'size_3':
    #     name_display = 'resnet-50_size_3'
    if name_display == '':
        name_display = model_name
    dict_model_name_diaplay_name[model_name] = name_display

    path_log = osp.join(dir_log, '{}.json'.format(model_name))
    with open(path_log, 'r') as f:
        log = json.load(f)

    locations = list(log.keys())
    for location, counts in log.items():
        thresholds = list(counts.keys())

    ROC = {}
    F1_model = {}
    AOC_model = 0
    tp_fn_tn_fp[model_name] = {}
    for threshold in thresholds:
        TP_frame, FN_frame, TN_frame, FP_frame = 0, 0, 0, 0
        for location in locations:
            if type == 'Per Frame':
                [[_, _, _, _], [tp, fn, tn, fp]] = log[location][threshold]
            elif type == 'Per Landmark':
                [[tp, fn, tn, fp], [_, _, _, _]] = log[location][threshold]
            TP_frame += tp
            FN_frame += fn
            TN_frame += tn
            FP_frame += fp
        TPR = TP_frame / (TP_frame + FN_frame)
        FPR = FP_frame / (FP_frame + TN_frame)
        Precision = TP_frame / max((TP_frame + FP_frame), 1)
        Recall = TP_frame / (TP_frame + FN_frame)
        ROC[threshold] = {
            'TP' : TP_frame,
            'FN' : FN_frame,
            'TN' : TN_frame,
            'FP' : FP_frame,
            'TPR': TPR,
            'FPR': FPR,
            }
        F1_model[threshold] = 2 * Precision * Recall / max((Precision + Recall), 1)
        AOC_model += TPR
        tp_fn_tn_fp[model_name][threshold] = [TP_frame, FN_frame, TN_frame, FP_frame]
    AOC[model_name] = AOC_model
    F1[model_name] = F1_model

aoc_plot = {}
f1_plot = {}
for model_name, aoc in AOC.items():
    aoc_plot[dict_model_name_diaplay_name[model_name]] = aoc
    f1_plot[dict_model_name_diaplay_name[model_name]] = F1[model_name]

plt.figure()
values = list(AOC.values())
plt.bar(list(dict_model_name_diaplay_name.values()), values)
for x, y in zip(list(dict_model_name_diaplay_name.values()), values):
    plt.text(x, y+0.05, '%.2f' %y, ha='center', va='bottom')
plt.xticks(rotation=30)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
plt.title('AOC_{}'.format(type))
plt.show()

# plt.figure()
# for i, (model_name, f1) in enumerate(f1_plot.items()):
#     plt.subplot(2, 4, i+1)
#     plt.title(model_name)
#     plt.plot(list(f1.keys()), list(f1.values()))
# plt.show()

plt.figure(figsize=(10, 6))
for model_name, f1 in f1_plot.items():
    plt.plot(list(f1.keys()), list(f1.values()))
plt.title('F1 Scores_{}'.format(type))
plt.legend(list(f1_plot.keys()))
plt.show()

plt.figure(figsize=(8, 10))
items = ['TP', 'FN', 'TN', 'FP']
ys = {}
for model_name, thresholds in tp_fn_tn_fp.items():
    ys[model_name] = {}
    for i, item in enumerate(items):
        ys[model_name][item] = [val[i] for val in list(thresholds.values())]
x = list(thresholds.keys())
legends = [dict_model_name_diaplay_name[model_name] for model_name in list(ys.keys())]
plot_items = ['TP', 'FP']
for i, plot_item in enumerate(plot_items):
    plt.subplot(2, 1, i+1)
    for _, data in ys.items():
        plt.plot(x, data[plot_item])
    plt.title('{}_{}'.format(plot_item, type))
    plt.legend(legends)
plt.show()


