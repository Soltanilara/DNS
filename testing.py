import numpy as np
import torch
from torchvision import transforms
import glob
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


def LoadModel_LandmarkStats(device,root_dir,no_landmarks,model_name,channels,im_height,im_width,transform):
    """
    Returns the loaded model, as well as necessary statistics from the landmark classes.
    Path landmark image folders are assumed to be under root_dir/landmarks and named as 0, 1, 2, etc.

    :param device: device used to store variables and the model
    :param root_dir: root directory containing the model, as well as the "landmarks" folder
    :param no_landmarks: number of landmarks; folders named as 0, 1, 2, etc.
    :param model_name: name of the model to be loaded
    :param channels: default: 3
    :param im_height: target height after transformation; default: 224
    :param im_width: after transformation; default: 224
    :param transform: composed image transforms
    """
    no_images = []
    landmarks = torch.empty((0, channels, im_height, im_width), requires_grad=False).cuda(device)
    for i in range (no_landmarks):
        f = glob.glob(root_dir+"landmarks/"+str(i)+"/*")
        no_images.append(len(f))
        for image in f:
            img = Image.open(image)
            img = transform(img).cuda(device)
            landmarks = torch.cat([landmarks,img.unsqueeze(0)],0)
    model = torch.load(root_dir+model_name, map_location='cpu').cuda(device)
    for param in model.parameters():
        param.requires_grad = False
    indiv_protos = model.encoder(landmarks)
    indiv_eigs = model.cov(indiv_protos) + 1e-8
    proto_sup = torch.empty((no_landmarks,1000), requires_grad=False).cuda(device)
    eigs_sup = torch.empty((no_landmarks,1000), requires_grad=False).cuda(device)
    for i in range(no_landmarks):
        start = sum(no_images[:i])
        proto_sup[i,:] = torch.mean(indiv_protos[start:start+no_images[i],:],0)
        deltasq = torch.mean(torch.pow(indiv_protos[start:start+no_images[i],:] - proto_sup[i,:], 2), 0)
        eigs_sup[i,:] = torch.mean(indiv_eigs[start:start+no_images[i],:],0)+deltasq

    return model, proto_sup, eigs_sup

def MatchDetector(model,image,lm_proto,lm_eigs,transform,probabilities,spread,threshold,device):
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
    image = transform(image)
    image = image.unsqueeze(0).cuda(device)
    qry_proto = model.encoder(image).squeeze()
    qry_eigs = model.cov(qry_proto).squeeze() + 1e-8
    dist = torch.abs(torch.dot((lm_proto.squeeze() - qry_proto) / (qry_eigs+lm_eigs.squeeze()), lm_proto.squeeze() - qry_proto))
    prob = torch.Tensor([torch.exp(-dist/spread)]).cuda(device)
    probabilities = torch.cat([probabilities[1:], prob], 0)
    if torch.mean(probabilities)>threshold:
        match = True
        probabilities = torch.zeros(probabilities.size(), requires_grad=False).cuda(device)
    else:
        match = False
    return match, probabilities

# Testing---------------------------------------------------------------------------------------------------------------

device = 3
channels, im_height, im_width = 3, 224, 224
root_dir = "./ASB1F/testing/CW/"
no_landmarks = 8
landmark_frames = [273,322,1149,1193,1550,1578,2379,2409] #frame number of last positive sample for a landmark
model_name = 'ModelMCN.pth'
spread = 1e0
threshold = 0.5
transform = transforms.Compose([
    transforms.Resize((im_height, im_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model, proto_sup, eigs_sup = LoadModel_LandmarkStats(device,root_dir,no_landmarks,model_name,channels,im_height,im_width,transform)

probabilities = torch.zeros((15), requires_grad=False).cuda(device)
landmark = 0
f = sorted(glob.glob(root_dir+"testlap/*"))
i=0
frame_prob = []
moving_avg_prob = []
while i<len(f) and landmark<no_landmarks:
    lm_proto = proto_sup[landmark, :]
    lm_eigs = eigs_sup[landmark, :]
    img = Image.open(f[i])
    match, probabilities = MatchDetector(model, img, lm_proto, lm_eigs, transform, probabilities, spread, threshold, device)
    frame_prob += [probabilities[-1].cpu().item()]
    moving_avg_prob += [probabilities.mean().cpu().item()]
    if match==True:
        landmark+=1
        print('Update to landmark ',str(landmark), ' at frame ', f[i][-9:])
    i+=1
# uncomment below if want to manually update upcoming landmarks if missed
    if i>landmark_frames[landmark]:
        landmark +=1
        probabilities = torch.zeros((15), requires_grad=False).cuda(device)


# plotting--------------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Frame number', fontsize = 20)
ax.set_ylabel('Landmark frame probability', fontsize = 20)
ax.plot(frame_prob)
ax.grid()
plt.ylim(0,1)
plt.xlim(0,i)
# plt.show
plt.savefig(root_dir+'landmark frame probability.png', dpi=300)
# plt.close()

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Frame number', fontsize = 20)
ax.set_ylabel('Moving average probability', fontsize = 20)
ax.plot(moving_avg_prob)
ax.grid()
plt.ylim(0,1)
plt.xlim(0,i)
# plt.show
plt.savefig(root_dir+'Moving average probability.png', dpi=300)
# plt.close()
