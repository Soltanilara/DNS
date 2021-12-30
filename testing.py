import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

from utils.loader import TestLoader
from utils.utils import summarizeSuperCat, get_imgId2landmarkId


def LoadModel_LandmarkStats(device, loader, model_path):
    """
    Returns the loaded model, as well as necessary statistics from the landmark classes.
    Path landmark image folders are assumed to be under root_dir/landmarks and named as 0, 1, 2, etc.

    :param device: device used to store variables and the model
    :param root_dir: root directory containing the model, as well as the "landmarks" folder
    :param no_landmarks: number of landmarks; folders named as 0, 1, 2, etc.
    :param model_path: name of the model to be loaded
    :param channels: default: 3
    :param im_height: target height after transformation; default: 224
    :param im_width: after transformation; default: 224
    :param transform: composed image transforms
    """

    model = torch.load(model_path, map_location='cpu').cuda(device)
    for param in model.parameters():
        param.requires_grad = False
    landmarks = loader.get_all_landmarks()
    num_landmarks = landmarks.shape[0]
    landmarks = landmarks.view([-1] + list(landmarks.shape)[-3:]).cuda(device)
    indiv_protos = model.encoder(landmarks)
    indiv_eigs = model.cov(indiv_protos) + 1e-8

    indiv_protos = indiv_protos.view([num_landmarks, -1] + [indiv_protos.shape[-1]])
    indiv_eigs = indiv_eigs.view([num_landmarks, -1] + [indiv_eigs.shape[-1]])

    proto_sup = torch.mean(indiv_protos, dim=1).squeeze()
    eigs_sup = torch.mean(indiv_eigs, dim=1).squeeze()

    return model, proto_sup, eigs_sup


def MatchDetector(model, image, lm_proto, lm_eigs, transform, probabilities, spread, threshold, device):
    """
    - Returns a landmark match/no match decision as "match" (boolean)
    - Updates the stored probability vector (similarities interpreted as probabilities) corresponding to the few recent
      individual images
    :param model: loaded model from LoadModel_LandmarkStats(...)
    :param image: incoming single image frame
    :param lm_proto: mean for the upcoming landmark, indexed from the second output of LoadModel_LandmarkStats(...)
    :param lm_eigs: covariance for the upcoming landmark, indexed from the third output of LoadModel_LandmarkStats(...)
    :param transform: composed image transforms
    :param probabilities: current probability vector
    :param spread: spread parameter for similarity kernel
    """
    lm_proto = lm_proto
    lm_eigs = lm_eigs
    qry_proto = model.encoder(image.cuda(device)).squeeze()
    qry_eigs = model.cov(qry_proto).squeeze() + 1e-8

    diff = lm_proto - qry_proto
    dist = diff / (qry_eigs + lm_eigs) * diff

    prob = model.classifier(dist)
    probabilities = torch.cat([probabilities[1:], prob], 0)
    if torch.mean(probabilities) > threshold:
        match = True
        probabilities = torch.zeros(probabilities.size(), requires_grad=False).cuda(device)
    else:
        match = False
    return match, probabilities

# Testing---------------------------------------------------------------------------------------------------------------


ts = time()

np.random.seed(0)
torch.manual_seed(0)

device = 1
channels, im_height, im_width = 3, 224, 224
root_dir = "/home/nick/dataset/ASB1F/"
# root_dir = "/home/nick/dataset/ASB1F/val_landmark/"
# root_dir = "/home/nick/dataset/ASB1F/train_landmark/"
no_landmarks = 16
model_path = '/home/nick/projects/FSL/ckpt/ModelMCN_ASB1F.pth'
# model_path = '/home/nick/projects/FSL/ckpt/model_best_3_epochs_ASB1F_batch30_12-shot_lr_0.001_lrsch_0.5_20_16episodes.pth'
spread = 200  # 1e0
threshold = 0.5

transform = transforms.Compose([
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_test = datasets.coco.CocoDetection(
        root=root_dir,
        annFile='coco/test.json',
        transform=transform
    )
dataloader = TestLoader(dataset_test, summarizeSuperCat(dataset_test))

model, proto_sup, eigs_sup = LoadModel_LandmarkStats(device, dataloader, model_path)

landmark = 0
i = 0
frame_prob = []
moving_avg_prob = []
probabilities = torch.zeros((15), requires_grad=False).cuda(device)
imgId2landmarkId, landmark_borders = get_imgId2landmarkId(dataset_test)
dataloader = dataloader.get_loader()
for i, imgs in enumerate(tqdm(dataloader)):
    landmark = imgId2landmarkId[i]
    lm_proto = proto_sup[landmark, :]
    lm_eigs = eigs_sup[landmark, :]
    match, probabilities = MatchDetector(model, imgs[0], lm_proto, lm_eigs, transform, probabilities, spread, threshold, device)
    frame_prob += [probabilities[-1].cpu().item()]
    moving_avg_prob += [probabilities.mean().cpu().item()]

    if match:
        print('\nUpdate to landmark ', str(landmark), ' at frame ', i)


# plotting--------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('Frame number', fontsize=20)
ax.set_ylabel('Landmark frame probability', fontsize=20)
ax.plot(frame_prob)
ax.grid()
plt.vlines(landmark_borders['start'], 0, 1, 'g', 'dashed')
plt.vlines(landmark_borders['end'], 0, 1, 'r', 'dashed')
plt.ylim(0, 1)
plt.xlim(0, i)
# plt.show
plt.savefig(root_dir+'landmark frame probability.png', dpi=300)
# plt.close()

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('Frame number', fontsize=20)
ax.set_ylabel('Moving average probability', fontsize=20)
ax.plot(moving_avg_prob)
ax.grid()
plt.vlines(landmark_borders['start'], 0, 1, 'g', 'dashed')
plt.vlines(landmark_borders['end'], 0, 1, 'r', 'dashed')
plt.ylim(0, 1)
plt.xlim(0, i)
# plt.show
plt.savefig(root_dir+'Moving average probability.png', dpi=300)
# plt.close()

print(time() - ts)
