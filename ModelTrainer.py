import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import csv
import time
from collections import OrderedDict

from models.StampNet import load_model
from utils.loader import RandNegLoader, ConsecLoader
from utils.utils import sortImgs, summarizeDataset

print(torch.cuda.is_available())
# torch.autograd.set_detect_anomaly(True)

ts = time.time()

np.random.seed(0)
torch.manual_seed(0)

if sys.platform == 'linux':
    # root_dir = "/home/nick/dataset/outside/"
    # root_dir = "/home/nick/dataset/val_from_train/"
    root_dir = "/home/nick/dataset/all8/"
    # root_dir = "/home/nick/dataset/ASB1F_V1/"
    # root_dir = "/mnt/data/dataset/av/outside/"
else:
    # root_dir = '/Users/shidebo/dataset/AV/Sorted/ASB1F'
    root_dir = '/Users/shidebo/dataset/AV/Sorted/'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = 0 if torch.cuda.is_available() else "cpu"  # use GPU if available
n_epochs = 40
n_episodes = 16
batch = 32
sup_size = 10
qry_size = 10
qry_num = 6
channels, im_height, im_width = 3, 224, 224
lr = 1e-3
stats_freq = 1
sch_param_1 = 20
sch_param_2 = 0.5
FC_len = 1000
course_name = '6_exclude_ASB1F_Bainer2F'
savename = course_name +'_batch' + str(batch) +'_' + str(sup_size) + '-shot_lr_' + str(lr) + '_lrsch_' + str(sch_param_2) + '_' + str(sch_param_1) + '_' + str(n_episodes) + 'episodes'
print(savename)


# Old dataset
# dataset = datasets.ImageFolder(root=root_dir + "train", transform=transform) #"train" folder contains image subfolders for positive and negative landmark images
# sortImgs(dataset)
# targets = [s[1] for s in dataset.samples]
#
# dataset_val = datasets.ImageFolder(root=root_dir + "val", transform=transform)
# targets_val = [s[1] for s in dataset_val.samples]

dataset_train = datasets.coco.CocoDetection(
    root=root_dir,
    annFile='coco/train.json',
    transform=transform
)
dataset_val = datasets.coco.CocoDetection(
    root=root_dir,
    annFile='coco/val.json',
    transform=transform
)


# Train def-------------------------------------------------------------------------
from tqdm import trange


def ftrain(model, optimizer, dataset_train, dataset_val, sup_size, qry_size, qry_num, max_epoch,
           epoch_size, accuracy_stats, loss_stats, stats_freq, sch_param_1, sch_param_2, batch,
           acc_first_epoch, loss_first_epoch):
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
    best_acc = 0
    model_best_path = ''

    while epoch < max_epoch and not stop:
        model.train()
        for episode_batch in trange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            loader = ConsecLoader(batch, sup_size, qry_size, qry_num, dataset_train, summarizeDataset(dataset_train))
            sample = loader.get_batch()
            optimizer.zero_grad()

            loss, output = model.set_forward_loss(sample)
            running_loss = float(output['loss'])
            running_acc = float(output['acc'])
            loss.backward()
            optimizer.step()
            loss_stats['train'].append(running_loss)
            accuracy_stats['train'].append(running_acc)
            if epoch == 0:
                acc_first_epoch['train'].append(running_acc)
                loss_first_epoch['train'].append(running_loss)
                if episode_batch < 20:
                    print('Epoch 1 batch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(episode_batch, running_loss,
                                                                                  running_acc)) #print first few batch results
        epoch += 1
        if epoch % stats_freq == 0:
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch, running_loss, running_acc))
        # Validation--------------------------------------
        with torch.no_grad():
            running_loss = 0.0
            running_acc = 0.0
            model.eval()
            for _ in trange(epoch_size, desc="Epoch {:d} val".format(epoch)):
                # loader = RandNegLoader(batch, sup_size, qry_size, qry_num, dataset_val, summarizeSuperCat(dataset_val))
                loader = ConsecLoader(batch, sup_size, qry_size, qry_num, dataset_val, summarizeDataset(dataset_val))
                sample = loader.get_batch()
                loss, output = model.set_forward_loss(sample)
                running_loss += float(output['loss'])
                running_acc += float(output['acc'])
            epoch_loss = running_loss / epoch_size
            epoch_acc = running_acc / epoch_size
            loss_stats['val'].append(epoch_loss)
            accuracy_stats['val'].append(epoch_acc)
            print('Epoch_val {:d} -- Loss_val: {:.4f} Acc_val: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                if model_best_path:
                    os.remove(model_best_path)
                model_best_path = 'ckpt/model_best_' + savename + '.pth'
                torch.save(model, model_best_path)
                print('Model saved to: ' + model_best_path)
        w = csv.writer(open("accuracy_stats_"+savename+".csv", "w")) # updates a record of accuracies
        for key, val in accuracy_stats.items():
            w.writerow([key, val])
        scheduler.step()
        print("\n", time.time() - ts, "\n")
    return loss_stats, accuracy_stats, loss_first_epoch, acc_first_epoch


# Begin pre-training------------------------------------------------------------------------

model = load_model(FCdim=FC_len, input_size=qry_num, device=device)

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
       sup_size, qry_size, qry_num, n_epochs, n_episodes,
       accuracy_stats, loss_stats, stats_freq,
       sch_param_1, sch_param_2, batch, acc_first_epoch, loss_first_epoch)
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
fig.savefig('Loss Acc ' + str(n_epochs) + ' epochs ' + savename + '.png')

train_first_epoch_acc_df = pd.DataFrame.from_dict(acc_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
train_first_epoch_loss_df = pd.DataFrame.from_dict(loss_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
sns.lineplot(data=train_first_epoch_acc_df, x="Batches", y="value", hue="variable", ax=axes[0]).set_title(
    'Training accuracy vs batch for the first epoch')
sns.lineplot(data=train_first_epoch_loss_df, x="Batches", y="value", hue="variable", ax=axes[1]).set_title(
    'Training loss vs batch for the first epoch')
fig.savefig('Epoch 1 ' + str(n_epochs) + ' epochs ' + savename + '.png')
#-----------------------------------------------------------------------------------------------------------------------

acc_list = accuracy_stats['val']
max_value = max(acc_list)
max_index = acc_list.index(max_value)
print(max_index+1) #prints the epoch number of pre-trained model with max accuracy

# model = torch.load('ckpt/model_best_' + str(max_index+1) + '_epochs_' + savename + '.pth').cuda(device) #loads the model with highest validation accuracy
model = torch.load('ckpt/model_best_' + savename + '.pth').cuda(device) #loads the model with highest validation accuracy

## Begin fine-tuning----------------------------------------------------------------------------------------------------------

n_episodes = 100
batch = 4
lr = 1e-5
savename = course_name +'_FineTune_NewMix_SymMah_batch' + str(batch) +'_' + str(sup_size) + '-shot' + '_lr_' + str(lr) + '_lrsch_' + str(sch_param_2) + '_' + str(sch_param_1) + '_' + str(n_episodes) + 'episodes'
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
       sup_size, qry_size, qry_num, n_epochs, n_episodes,
       accuracy_stats, loss_stats, stats_freq,
       sch_param_1, sch_param_2, batch, acc_first_epoch, loss_first_epoch)
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
fig.savefig('Loss Acc ' + str(n_epochs) + ' epochs ' + savename + '.png')

train_first_epoch_acc_df = pd.DataFrame.from_dict(acc_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
train_first_epoch_loss_df = pd.DataFrame.from_dict(loss_first_epoch).reset_index().melt(id_vars=['index']).rename(
    columns={"index": "Batches"})
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))
sns.lineplot(data=train_first_epoch_acc_df, x="Batches", y="value", hue="variable", ax=axes[0]).set_title(
    'Training accuracy vs batch for the first epoch')
sns.lineplot(data=train_first_epoch_loss_df, x="Batches", y="value", hue="variable", ax=axes[1]).set_title(
    'Training loss vs batch for the first epoch')
fig.savefig('Epoch 1 ' + str(n_epochs) + ' epochs ' + savename + '.png')

acc_list = accuracy_stats['val']
max_value = max(acc_list)
max_index = acc_list.index(max_value)
print(max_index+1) #prints the epoch number of fine-tuned model with max accuracy

# Load the best-performing model and save with suitable name and format for evaluation----------------------------------
model = torch.load('ckpt/model_best_' + savename + '.pth').cuda(device) #loads the fine-tuned model with highest validation accuracy
encoder = model.encoder
cov_module = model.cov_module
classifier = model.classifier
model = nn.Sequential(OrderedDict([('encoder', encoder), ('cov', cov_module), ('classifier', classifier)]))
torch.save(model, 'ckpt/ModelMCN_'+course_name+'.pth')