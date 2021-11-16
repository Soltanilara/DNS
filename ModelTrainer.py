import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import csv
import time
from collections import OrderedDict

print(torch.cuda.is_available())
# torch.autograd.set_detect_anomaly(True)

ts = time.time()

np.random.seed(0)
torch.manual_seed(0)

# root_dir = "/home/nick/dataset/outside/"
# root_dir = "/home/nick/dataset/val_from_train/"
root_dir = "/mnt/data/dataset/av/outside/"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


device = 0 if torch.cuda.is_available() else "cpu"  # use GPU if available
n_epochs = 80
n_episodes = 16
batch = 16  #25
n_way = 2  #positive/negative
n_way_val = 2
n_support = 15
n_query = 10
channels, im_height, im_width = 3, 224, 224
lr = 1e-3
save_freq = 1
stats_freq = 1
sch_param_1 = 20
sch_param_2 = 0.5
FC_len = 1000
course_name = 'outside'
savename = course_name+'_xCar_batch'+str(batch)+'_'+str(n_support)+'-shot_way_train_'+str(n_way)+'_lr_'+str(lr)+'_lrsch_'+str(sch_param_2)+'_'+str(sch_param_1)+'_'+str(n_episodes)+'episodes'
print(savename)


dataset = datasets.ImageFolder(root=root_dir + "train", transform=transform) #"train" folder contains image subfolders for positive and negative landmark images
targets = [s[1] for s in dataset.samples]

dataset_val = datasets.ImageFolder(root=root_dir + "val", transform=transform)
targets_val = [s[1] for s in dataset_val.samples]


def extract_sample(batch, n_way, n_support, n_query, dataset, targets, train):
    n_class_tot = len(dataset.classes)
    n_data_tot = len(dataset)
    sample_batch = torch.empty([batch, n_way, n_support + n_query] + list(dataset[0][0].shape))
    for b in range(batch):
        k = np.random.choice(np.unique(range(np.floor(n_class_tot/2).astype(int))), 1, replace=False).item()
        sample = torch.empty([0, n_support + n_query] + list(dataset[0][0].shape))
        for i, cls in enumerate([2*k, 2*k+1]):
            if not train or i == 0:
                weights = list(map(int, [x == y for (x, y) in zip(targets, [cls] * n_data_tot)]))
            else:
                weights = list(map(int, [x != y for (x, y) in zip(targets, [cls - 1] * n_data_tot)]))
            sampler = WeightedRandomSampler(weights, n_support + n_query, replacement=False)
            loader = DataLoader(dataset=dataset, shuffle=False, batch_size=n_support + n_query, sampler=sampler,
                                drop_last=False)  # , batch_size = n_support+n_query
            sample_cls = next(iter(loader))[0].unsqueeze(dim=0)
            sample = torch.cat([sample, sample_cls], dim=0)
        sample_batch[b] = sample
    return ({
        'images': sample_batch,
        'batch': batch,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


# Build model-------------------------------------------------------------------

def load_model(**kwargs):
    """
    Loads the network model
    Arg:
        FCdim: latent space dimensionality
    """
    FCdim = kwargs['FCdim']

    encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50') #pretrained ResNet-50
    for param in encoder.parameters():
        param.requires_grad = False #freezes the pretrained model parameters
    encoder.fc = nn.Linear(2048,FCdim)

    cov_module = nn.Sequential(
        nn.Linear(FCdim, FCdim),
        nn.Tanh(),
        nn.Linear(FCdim, FCdim),
        nn.Softplus()
    )
    return StampNet(encoder, cov_module)

class StampNet(nn.Module):
    def __init__(self, encoder, cov_module):
        super(StampNet, self).__init__()
        if device == "cpu":
            self.encoder = encoder
            self.cov_module = cov_module
        else:
            self.encoder = encoder.cuda(device)
            self.cov_module = cov_module.cuda(device)


    def set_forward_loss(self, sample):
        """
        Takes the sample batch and computes loss, accuracy and output for classification task
        """
        sample_images = sample['images'].cuda(device)
        batch = sample['batch']
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        x_support = sample_images[:, :, :n_support]
        x_query = sample_images[:, :, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(1, n_way, 1, 1).expand(batch, n_way, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).cuda(device)

        # concatenate and prepare images of the support and query sets for inputting to the network
        x = torch.cat([x_support.contiguous().view(batch * n_way * n_support, *x_support.size()[3:]),
                       x_query.contiguous().view(batch * n_way * n_query, *x_query.size()[3:])], 0)
        z = self.encoder.forward(x)
        indiv_protos = z
        indiv_eigs = self.cov_module.forward(z) + 1e-8

        proto_indiv_sup = indiv_protos[:batch * n_way * n_support].view(batch, n_way, n_support, -1)
        proto_qry = indiv_protos[batch * n_way * n_support:].view(batch, 1, n_way, n_query, -1).expand(-1, n_way, -1,
                                                                                                       -1, -1)
        eigs_indiv_sup = indiv_eigs[:batch * n_way * n_support].view(batch, n_way, n_support, -1)
        eigs_qry = indiv_eigs[batch * n_way * n_support:].view(batch, 1, n_way, n_query, -1).expand(-1, n_way, -1, -1,
                                                                                                    -1)
        proto_sup = torch.mean(proto_indiv_sup, 2).view(batch, n_way, 1, -1)
        deltasq_sup = torch.mean(torch.pow(proto_indiv_sup-proto_sup,2),2).view(batch, n_way, 1, -1)
        eigs_sup = torch.mean(eigs_indiv_sup, 2).view(batch, n_way, 1, -1) +deltasq_sup

        proto_sup = proto_sup.view(batch, n_way, 1, 1, -1).expand(-1, -1, n_way, n_query, -1)
        eigs_sup = eigs_sup.view(batch, n_way, 1, 1, -1).expand(-1, -1, n_way, n_query, -1)

        diff = proto_sup - proto_qry
        dists = torch.sum((diff / (eigs_sup + eigs_qry)).view(batch * n_way * n_way * n_query, -1) * diff.view(
            batch * n_way * n_way * n_query, -1), dim=1).view(batch, n_way, n_way, n_query).permute(0, 2, 3, 1)

        log_p_y = F.log_softmax(-dists, dim=3)

        loss_val = -log_p_y.gather(3, target_inds).squeeze().view(
            -1).mean()  # sum negative log softmax for the correct class, normalize by number of entries
        _, y_hat = log_p_y.max(3)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        acc_means = torch.eq(y_hat, target_inds.squeeze()).view(batch, -1).float().mean(1)

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'acc_means': acc_means
        }


# Train def-------------------------------------------------------------------------
# from tqdm.notebook import trange
from tqdm import trange


def ftrain(model, optimizer, train_x, train_y, val_x, val_y, n_way, n_way_val, n_support, n_query, max_epoch,
           epoch_size, accuracy_stats, loss_stats, save_freq, stats_freq, sch_param_1, sch_param_2, batch,
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

    while epoch < max_epoch and not stop:
        model.train()
        for episode_batch in trange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(batch, n_way, n_support, n_query, train_x, train_y, True)
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
        if epoch % save_freq == 0:
            torch.save(model, 'ckpt/model_' + str(epoch) + '_epochs_' + savename + '.pth')
        if epoch % stats_freq == 0:
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch, running_loss, running_acc))
        # Validation--------------------------------------
        with torch.no_grad():
            running_loss = 0.0
            running_acc = 0.0
            model.eval()
            for episode_batch in trange(epoch_size, desc="Epoch {:d} val".format(epoch)):
                sample = extract_sample(batch, n_way_val, n_support, n_query, val_x, val_y, False)
                loss, output = model.set_forward_loss(sample)
                running_loss += float(output['loss'])
                running_acc += float(output['acc'])
            epoch_loss = running_loss / epoch_size
            epoch_acc = running_acc / epoch_size
            loss_stats['val'].append(epoch_loss)
            accuracy_stats['val'].append(epoch_acc)
            print('Epoch_val {:d} -- Loss_val: {:.4f} Acc_val: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        w = csv.writer(open("accuracy_stats_"+savename+".csv", "w")) # updates a record of accuracies
        for key, val in accuracy_stats.items():
            w.writerow([key, val])
        scheduler.step()
        print("\n", time.time() - ts, "\n")
    return loss_stats, accuracy_stats, loss_first_epoch, acc_first_epoch


# Begin pre-training------------------------------------------------------------------------

model = load_model(FCdim=FC_len)

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
ftrain(model, optimizer, dataset, targets, dataset_val, targets_val,
       n_way, n_way_val, n_support, n_query, n_epochs, n_episodes,
       accuracy_stats, loss_stats, save_freq, stats_freq,
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

model = torch.load('model_' + str(max_index+1) + '_epochs_' + savename + '.pth').cuda(device) #loads the model with highest validation accuracy

## Begin fine-tuning----------------------------------------------------------------------------------------------------------

n_episodes = 100
batch = 4
lr = 1e-5
savename = course_name+'_FineTune_NewMix_SymMah_batch'+str(batch)+'_'+str(n_support)+'-shot_way_train_'+str(n_way)+'_lr_'+str(lr)+'_lrsch_'+str(sch_param_2)+'_'+str(sch_param_1)+'_'+str(n_episodes)+'episodes'
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

(model, optimizer, dataset, targets, dataset_val, targets_val,
       n_way, n_way_val, n_support, n_query, n_epochs, n_episodes,
       accuracy_stats, loss_stats, save_freq, stats_freq,
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
model = torch.load('model_' + str(max_index+1) + '_epochs_' + savename + '.pth').cuda(device) #loads the fine-tuned model with highest validation accuracy
encoder = model.encoder
cov_module = model.cov_module
model=nn.Sequential(OrderedDict([('encoder',encoder),('cov',cov_module)]))
torch.save(model,'ModelMCN_'+course_name+'.pth')