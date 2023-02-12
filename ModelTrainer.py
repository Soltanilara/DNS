import os
import os.path as osp
import sys
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import csv
import json
import time
from collections import OrderedDict
from tqdm import tqdm

from models.StampNet import load_model
from testing import LoadModel_LandmarkStats, MatchDetector
from utils.loader import ConsecLoader, TestLoader
from utils.datasets import get_dataset, get_dataloader_val
from utils.utils import str2bool, get_imgId2landmarkId, per_landmark, plot_figure

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', '--name', type=str, required=True,
                    help='Model name')
parser.add_argument('-device', '--device', type=int, required=False,
                    help='Device ID')
parser.add_argument('-b', '--backbone', default='resnet50', type=str, required=False,
                    help='Model backbone')
parser.add_argument('-aug', '--aug', default=True, type=str2bool, required=False,
                    help='Whether or not apply data augmentation in training')
parser.add_argument('-val_freq', '--val_freq', default=2, type=int, required=False,
                    help='Validation frequency (default: 2)')
parser.add_argument('-epoch_size', '--epoch_size', default=16, type=int, required=False,
                    help='Number of episodes in a epoch (default: 16)')
parser.add_argument('-epoch_pre', '--epoch_pre', default=3, type=int, required=False,
                    help='Number of epochs in pre-training (default: 3)')
parser.add_argument('-epoch_fine', '--epoch_fine', default=30, type=int, required=False,
                    help='Number of epochs in fine-tuning (default: 30)')
parser.add_argument('-batch_pre', '--batch_pre', default=16, type=int, required=False,
                    help='Batch size in pre-training (default: 16)')
parser.add_argument('-batch_fine', '--batch_fine', default=3, type=int, required=False,
                    help='Batch size in fine-tuning (default: 3)')
parser.add_argument('-size_sup', '--sup_size', default=10, type=int, required=False,
                    help='Size of the support set (default: 10)')
parser.add_argument('-size_qry', '--qry_size', default=10, type=int, required=False,
                    help='Size of the query set (default: 10)')
parser.add_argument('-num_qry', '--qry_num', default=6, type=int, required=False,
                    help='Size of the query set (default: 6)')
parser.add_argument('-skip_cov', '--skip_cov', default=False, type=str2bool, required=False,
                    help='Whether or not to skip the cov module (default: False)')
parser.add_argument('-norm_dist', '--norm_dist', default=False, type=str2bool, required=False,
                    help='Whether or not to normalize the distance (default: False)')
parser.add_argument('-debug', '--debug', default=False, type=str2bool, required=False,
                    help='Debug mode (default: False)')

args = parser.parse_args()

print(torch.cuda.is_available())
# torch.autograd.set_detect_anomaly(True)

ts = time.time()

np.random.seed(0)
torch.manual_seed(0)

if sys.platform == 'linux':
    dir_dataset = '/home/nick/dataset/dual_fisheye_indoor/PNG'
    dir_coco = '/home/nick/projects/FSL/coco/dual_fisheye/15_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F/PNG'
    dir_output = '/home/nick/projects/FSL/output'
    dir_ckpt = '/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt'
else:
    dir_dataset = '/Users/shidebo/dataset/AV/Sorted/'

device = args.device if torch.cuda.is_available() else "cpu"  # use GPU if available
backbone = args.backbone
n_epochs_pre = args.epoch_pre
n_epochs_fine =args.epoch_fine
n_episodes = args.epoch_size
batch_pre = args.batch_pre
batch_fine = args.batch_fine
sup_size = args.sup_size
qry_size = args.qry_size
qry_num = args.qry_num
skip_cov = args.skip_cov
norm_dist = args.norm_dist
debug = args.debug
stats_freq = args.val_freq
channels, im_height, im_width = 3, 224, 224
lr = 1e-3
sch_param_1 = 10
sch_param_2 = 0.5
FC_len = 1000
course_name = 'dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_{}'.format(args.name)
savename = course_name+'_batch'+str(batch_pre)+'_' + str(sup_size)+'-shot_lr_'+str(lr)+'_lrsch_'+str(sch_param_2)+'_'+str(sch_param_1)+'_'+str(n_episodes)+'episodes'
print(savename)
if norm_dist:
    print('Normalizing distance')

dir_ckpt = osp.join(dir_ckpt, savename)
dir_ckpt_all = osp.join(dir_ckpt, 'ckpt_all')
for d in [dir_ckpt_all, dir_ckpt_all]:
    if not osp.exists(d):
        os.makedirs(d)

type_train = 'train' if args.aug else 'val'
dataset_train = get_dataset(dir_dataset, osp.join(dir_coco, 'train.json'), type_train, args)
dataset_val = get_dataset(dir_dataset, osp.join(dir_coco, 'val.json'), 'val', args)

dataloaders_val = get_dataloader_val(dataset_val)


# Train def-------------------------------------------------------------------------
from tqdm import trange


def ftrain(model, optimizer, dataset_train, dataset_val, sup_size, qry_size, qry_num, max_epoch,
           epoch_size, accuracy_stats, loss_stats, stats_freq, sch_param_1, sch_param_2, batch,
           acc_first_epoch, loss_first_epoch, mode):
    """
    Trains the StampNet
    Args:
        model
        optimizer
        train_x (np.array): images of training set
        train_y(np.array): labels of training set
        n_way (int): number of classes
    ftrain     n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
        max_epoch (int): max epochs to train on
        epoch_size (int): episodes per epoch
        ...
    """

    # multiply by sch_param_2 to shrink the learning rate every sch_param_1 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sch_param_1, gamma=sch_param_2, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop
    best_f1 = 0
    model_best_path = ''
    data_val = []

    model.train()
    while epoch < max_epoch and not stop:
        bar = tqdm(range(epoch_size), desc="Epoch {:d} train".format(epoch + 1))
        for episode_batch in bar:
            if debug and episode_batch:
                break
            loader = ConsecLoader(batch, sup_size, qry_size, qry_num, dataset_train)
            sample = loader.get_batch()
            optimizer.zero_grad()

            loss, output = model.set_forward_loss(sample)
            running_loss = float(output['loss'])
            running_f1 = float(output['acc'])
            loss.backward()
            optimizer.step()
            loss_stats['train'].append(running_loss)
            accuracy_stats['train'].append(running_f1)
            bar.set_postfix({
                'Loss': running_loss,
                'Acc': running_f1
            })
            if epoch == 0:
                acc_first_epoch['train'].append(running_f1)
                loss_first_epoch['train'].append(running_loss)
                # if episode_batch < 20:
                #     print('Epoch 1 batch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(episode_batch, running_loss,
                #                                                                   running_acc)) #print first few batch results
        epoch += 1
        if epoch % stats_freq == 0:
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch, running_loss, running_f1))
            # Validation--------------------------------------
            model.eval()
            with torch.no_grad():
                running_loss = []
                running_f1 = []
                data_val_epoch = {}
                pbar = tqdm(dataset_val.summary['location2Lap'].keys())
                for location in pbar:
                    dataloader_landmark = dataloaders_val[location]['landmark']
                    dataloader_test = dataloaders_val[location]['test']

                    model, proto_sup, eigs_sup = LoadModel_LandmarkStats(device, dataloader_landmark, model)

                    imgId2landmarkId, landmark_borders, gt = get_imgId2landmarkId(dataset_val, dataloader_test.catIds)

                    qry_imgs = None
                    frame_prob = []
                    tp, fn, tn, fp = 0, 0, 0, 0
                    threshold = 0.5
                    probabilities = torch.zeros(15, requires_grad=False).cuda(device)
                    for i, img in enumerate(dataloader_test):
                        pbar.set_description_str('-- Validation: {} [{}/{}]'.format(location, i, len(dataloader_test)))
                        if qry_imgs is None:
                            qry_imgs = img[0]
                            frame_prob += [0]
                            # tn += 1
                        elif qry_imgs.shape[0] < qry_size:
                            qry_imgs = torch.cat([qry_imgs, img[0]], dim=0)
                            frame_prob += [0]
                            # tn += 1
                        else:
                            qry_imgs = torch.cat([qry_imgs[1:], img[0]], dim=0)
                            landmark = imgId2landmarkId[list(imgId2landmarkId.keys())[i]]
                            lm_proto = proto_sup[landmark, :]
                            lm_eigs = eigs_sup[landmark, :] if eigs_sup is not None else None
                            match, probabilities = MatchDetector(
                                model, qry_imgs, lm_proto, lm_eigs, probabilities, threshold=0.5, device=device)
                            p = probabilities[-1].cpu().item()
                            # if gt[i] >= threshold and p >= threshold: tp += 1
                            # if gt[i] >= threshold and p < threshold: fn += 1
                            # if gt[i] < threshold and p >= threshold: fp += 1
                            # if gt[i] < threshold and p < threshold: tn += 1
                            frame_prob.append(p)
                    tp, fn, tn, fp = per_landmark(frame_prob, landmark_borders, threshold, tp, fn, tn, fp)

                    loss_val = F.binary_cross_entropy(torch.tensor(frame_prob, dtype=float), torch.tensor(gt, dtype=float))
                    precision = tp/(tp+fp+1e-3)
                    recall = tp/(tp+fn+1e-3)
                    f1 = 2*recall*precision/(recall+precision+1e-3)
                    running_loss.append(loss_val)
                    running_f1.append(f1)

                    if debug:
                        plot_figure(location, frame_prob, landmark_borders, 'frame', 'prob', show=True)

                    pbar.set_postfix({
                        'Loss': np.mean(running_loss),
                        'F1': np.mean(running_f1)
                    })
                    data_val_epoch[location] = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}
                    if debug:
                        break
                running_loss_mean = np.mean(running_loss)
                running_f1_mean = np.mean(running_f1)
                loss_stats['val'].append(running_loss_mean)
                accuracy_stats['val'].append(running_f1_mean)
                data_val.append(data_val_epoch)
                print('Epoch_val {:d} -- Loss_val: {:.4f} Acc_val: {:.4f}'.format(epoch, running_loss_mean, running_f1_mean))
                if running_f1_mean >= best_f1:
                    best_f1 = running_f1_mean
                    if model_best_path:
                        os.remove(model_best_path)
                    model_best_path = osp.join(dir_ckpt, 'model_best_' + savename + '.pth')
                    torch.save(model, model_best_path)
                    print('Model saved to: ' + model_best_path)
                path_model = osp.join(dir_ckpt_all, '{}_epoch_{}.pth'.format(savename, epoch))
                torch.save(model, path_model)
        scheduler.step()

        dir_csv = osp.join(dir_output, 'csv')
        if not osp.exists(dir_csv):
            os.makedirs(dir_csv)
        w = csv.writer(open(osp.join(dir_csv, "accuracy_stats_"+savename+".csv") , "w")) # updates a record of accuracies
        for key, val in accuracy_stats.items():
            w.writerow([key, val])

        dir_json = osp.join(dir_output, 'json')
        if not osp.exists(dir_json):
            os.makedirs(dir_json)
        if mode == 'pre':
            path_json = 'data_val_{}_pre.json'.format(savename)
        elif mode == 'fine':
            path_json = 'data_val_{}_fine.json'.format(savename)
        with open(osp.join(dir_json, path_json), 'w') as f:
            json.dump(data_val, f)

        print("\nUsed time: ", time.time() - ts, " s\n")
    return loss_stats, accuracy_stats, data_val, loss_first_epoch, acc_first_epoch


# Begin pre-training------------------------------------------------------------------------

if debug:
    model_path = '/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_all_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/ckpt_all/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_all_FineTune_NewMix_SymMah_batch3_10-shot_lr_1e-05_lrsch_0.5_10_100episodes_epoch_6.pth'
    model = torch.load(model_path, map_location='cpu').cuda(device)
else:
    model = load_model(FCdim=FC_len, backbone=backbone, device=device, skip_cov=skip_cov, norm_dist=norm_dist)

# shows the number of trainable parameters----------------------------------------------
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

# # shows the number of trainable parameters (alternative method)-------------------------
# def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# count = count_parameters(model)
# print(count)

optimizer = optim.Adam(model.parameters(), lr=lr)

accuracy_stats = {
    'train': [],
    'val': []
}
loss_stats = {
    'train': [],
    'val': []
}

acc_first_epoch = {
    'train': []
}
loss_first_epoch = {
    'train': []
}

print(time.time() - ts)
ftrain(model, optimizer, dataset_train, dataset_val,
       sup_size, qry_size, qry_num, n_epochs_pre, n_episodes,
       accuracy_stats, loss_stats, stats_freq,
       sch_param_1, sch_param_2, batch_pre, acc_first_epoch, loss_first_epoch, 'pre')
time.time() - ts

# torch.save(model, 'model_' + savename + '.pth')

# Plotting train and validation loss/accuracy vs epoch (also separately during first epoch)-----------------------------
train_acc_df = pd.DataFrame.from_dict(accuracy_stats['train']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
val_acc_df = pd.DataFrame.from_dict(accuracy_stats['val']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Epochs"})
val_acc_df.Epochs = np.arange(1, len(val_acc_df) + 1)
train_loss_df = pd.DataFrame.from_dict(loss_stats['train']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
val_loss_df = pd.DataFrame.from_dict(loss_stats['val']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Epochs"})
val_loss_df.Epochs = np.arange(1, len(val_loss_df) + 1)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
sns.lineplot(data=train_acc_df, x="Batches", y="value", hue="variable", ax=axes[0, 0], legend=False).set_title(
    'Training accuracy vs batch')
sns.lineplot(data=val_acc_df, x="Epochs", y="value", hue="variable", ax=axes[0, 1], legend=False,
             palette=['red']).set_title(
    'Validation accuracy vs epoch')
sns.lineplot(data=train_loss_df, x="Batches", y="value", hue="variable", ax=axes[1, 0], legend=False).set_title(
    'Training loss vs batch')
sns.lineplot(data=val_loss_df, x="Epochs", y="value", hue="variable", ax=axes[1, 1], legend=False,
             palette=['red']).set_title(
    'Validation loss vs epoch')
fig.savefig(osp.join(dir_output, 'figures', 'Loss Acc '+str(n_epochs_pre)+' epochs '+savename+'.png'))

train_first_epoch_acc_df = pd.DataFrame.from_dict(acc_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
train_first_epoch_loss_df = pd.DataFrame.from_dict(loss_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
sns.lineplot(data=train_first_epoch_acc_df, x="Batches", y="value", hue="variable", ax=axes[0]).set_title(
    'Training accuracy vs batch for the first epoch')
sns.lineplot(data=train_first_epoch_loss_df, x="Batches", y="value", hue="variable", ax=axes[1]).set_title(
    'Training loss vs batch for the first epoch')
fig.savefig(osp.join(dir_output, 'figures', 'Epoch 1 ' + str(n_epochs_pre) + ' epochs ' + savename + '.png'))
#-----------------------------------------------------------------------------------------------------------------------

if n_epochs_pre > 0:
    acc_list = accuracy_stats['val']
    max_value = max(acc_list)
    max_index = acc_list.index(max_value)
    print(max_index+1) #prints the epoch number of pre-trained model with max accuracy

if n_epochs_pre > 0:
    path_model = osp.join(dir_ckpt, 'model_best_' + savename + '.pth')
    model = torch.load(path_model).cuda(device) #loads the model with highest validation accuracy

## Begin fine-tuning----------------------------------------------------------------------------------------------------------

n_episodes = 100
lr = 1e-5
savename = course_name +'_FineTune_NewMix_SymMah_batch' + str(batch_fine) +'_' + str(sup_size) + '-shot' + '_lr_' + str(lr) + '_lrsch_' + str(sch_param_2) + '_' + str(sch_param_1) + '_' + str(n_episodes) + 'episodes'
print(savename)

for param in model.parameters():
   param.requires_grad = True #unfreezes the frozen parameters

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

optimizer = optim.Adam(model.parameters(), lr=lr)

accuracy_stats = {
    'train': [],
    'val': []
}
loss_stats = {
    'train': [],
    'val': []
}

acc_first_epoch = {
    'train': []
}
loss_first_epoch = {
    'train': []
}

print(time.time() - ts)

ftrain(model, optimizer, dataset_train, dataset_val,
       sup_size, qry_size, qry_num, n_epochs_fine, n_episodes,
       accuracy_stats, loss_stats, stats_freq,
       sch_param_1, sch_param_2, batch_fine, acc_first_epoch, loss_first_epoch, 'fine')
time.time() - ts

# torch.save(model, 'model_' + savename + '.pth')

train_acc_df = pd.DataFrame.from_dict(accuracy_stats['train']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
val_acc_df = pd.DataFrame.from_dict(accuracy_stats['val']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Epochs"})
val_acc_df.Epochs = np.arange(1, len(val_acc_df) + 1)
train_loss_df = pd.DataFrame.from_dict(loss_stats['train']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
val_loss_df = pd.DataFrame.from_dict(loss_stats['val']).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Epochs"})
val_loss_df.Epochs = np.arange(1, len(val_loss_df) + 1)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
sns.lineplot(data=train_acc_df, x="Batches", y="value", hue="variable", ax=axes[0, 0], legend=False).set_title(
    'Training accuracy vs batch')
sns.lineplot(data=val_acc_df, x="Epochs", y="value", hue="variable", ax=axes[0, 1], legend=False,
             palette=['red']).set_title(
    'Validation accuracy vs epoch')
sns.lineplot(data=train_loss_df, x="Batches", y="value", hue="variable", ax=axes[1, 0], legend=False).set_title(
    'Training loss vs batch')
sns.lineplot(data=val_loss_df, x="Epochs", y="value", hue="variable", ax=axes[1, 1], legend=False,
             palette=['red']).set_title(
    'Validation loss vs epoch')
fig.savefig(osp.join(dir_output, 'figures', 'Loss Acc '+str(n_epochs_fine)+' epochs '+savename+'.png'))

train_first_epoch_acc_df = pd.DataFrame.from_dict(acc_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
train_first_epoch_loss_df = pd.DataFrame.from_dict(loss_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
sns.lineplot(data=train_first_epoch_acc_df, x="Batches", y="value", hue="variable", ax=axes[0]).set_title(
    'Training accuracy vs batch for the first epoch')
sns.lineplot(data=train_first_epoch_loss_df, x="Batches", y="value", hue="variable", ax=axes[1]).set_title(
    'Training loss vs batch for the first epoch')
fig.savefig(osp.join(dir_output, 'figures', 'Epoch 1 '+str(n_epochs_fine)+' epochs '+savename+'.png'))

acc_list = accuracy_stats['val']
max_value = max(acc_list)
max_index = acc_list.index(max_value)
print('Best epoch: {}'.format((max_index+1)*2)) #prints the epoch number of fine-tuned model with max accuracy

# Load the best-performing model and save with suitable name and format for evaluation----------------------------------
model_best_path = osp.join(dir_ckpt, 'model_best_' + savename + '.pth')
model = torch.load(model_best_path).cuda(device) #loads the fine-tuned model with highest validation accuracy
encoder = model.encoder
cov = model.cov
classifier = model.classifier
model = nn.Sequential(OrderedDict([('encoder', encoder), ('cov', cov), ('classifier', classifier)]))
path_model = osp.join(dir_ckpt_all, 'ModelMCN_{}.pth'.format(course_name))
# torch.save(model, 'ckpt/ModelMCN_'+course_name+'.pth')
torch.save(model, path_model)
os.remove(model_best_path)
