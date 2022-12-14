import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models


def load_model(**kwargs):
    """
    Loads the network model
    Arg:
        FCdim: latent space dimensionality
    """
    FCdim = kwargs['FCdim']
    backbone = kwargs['backbone']
    device = kwargs['device']
    skip_cov = kwargs['skip_cov']

    print('Using backbone: {}'.format(backbone))
    if backbone == 'resnet50_swav':
        encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50')  # unsupervised trained
    elif backbone == 'resnet50':
        encoder = models.resnet50(pretrained=True)  # supervised trained
    elif 'efficientnet' in backbone:
        from efficientnet_pytorch import EfficientNet
        encoder = EfficientNet.from_pretrained(backbone)
    elif backbone == 'scratch':
        print('Training ResNet50 from scratch')
        encoder = models.resnet50(pretrained=False)

    for param in encoder.parameters():
        param.requires_grad = False  #freezes the pretrained model parameters
    encoder.fc = nn.Linear(2048, FCdim)  #todo: try different FCdim

    if skip_cov:
        print('Skipping cov module')
        cov = None
    else:
        cov = nn.Sequential(
            nn.Linear(FCdim, FCdim),
            nn.Tanh(),
            nn.Linear(FCdim, FCdim),
            nn.Softplus()
        )

    classifier = nn.Sequential(
        nn.Linear(FCdim*2, FCdim*2),
        nn.ELU(),
        nn.Linear(FCdim*2, 100),
        nn.ELU(),
        nn.Linear(100, 10),
        nn.ELU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    return StampNet(encoder, cov, classifier, device)


class StampNet(nn.Module):
    def __init__(self, encoder, cov_module, classifier, device):
        super(StampNet, self).__init__()

        self.device = device
        if self.device == 'cpu':
            self.encoder = encoder
            self.cov = cov_module
            self.classifier = classifier
        else:
            self.encoder = encoder.cuda(self.device)
            self.cov = cov_module.cuda(self.device) if cov_module else None
            self.classifier = classifier.cuda(self.device)

    def forward_one_side(self, x_support, x_query, batch, sup_num, sup_size, qry_num, qry_size):
        # concatenate and prepare images of the support and query sets for inputting to the network
        x = torch.cat([x_support.contiguous().view(batch * sup_size, *x_support.size()[-3:]),
                       x_query.contiguous().view(batch * qry_num * qry_size, *x_query.size()[-3:])], 0)
        z = self.encoder.forward(x)
        indiv_protos = z

        proto_indiv_sup = indiv_protos[:batch * sup_size].view(batch, sup_num, sup_size, -1)
        proto_sup = torch.mean(proto_indiv_sup, 2)
        proto_sup = proto_sup.view(batch, 1, -1).expand(-1, qry_num, -1)
        proto_qry = indiv_protos[batch * sup_size:].view(batch, qry_num, qry_size, -1)
        proto_qry = torch.mean(proto_qry, 2)
        diff = proto_sup - proto_qry

        if self.cov:
            indiv_eigs = self.cov.forward(z) + 1e-8
            eigs_indiv_sup = indiv_eigs[:batch * sup_size].view(batch, sup_num, sup_size, -1)
            eigs_qry = indiv_eigs[batch * sup_size:].view(batch, qry_num, qry_size, -1)
            eigs_sup = torch.mean(eigs_indiv_sup, 2)
            eigs_qry = torch.mean(eigs_qry, 2)
            eigs_sup = eigs_sup.view(batch, 1, -1).expand(-1, qry_num, -1)
            dists = (diff/(eigs_sup+eigs_qry)).view(batch*qry_num, -1) * diff.view(batch*qry_num, -1)
        else:
            dists = (diff**2).view(batch*qry_num, -1)

        return dists

    def set_forward_loss(self, sample):
        """
        Takes the sample batch and computes loss, accuracy and output for classification task
        """
        sample_images = sample['images'].cuda(self.device)
        batch = sample['batch']
        sup_size = sample['sup_size']
        qry_size = sample['qry_size']
        sup_num = 1
        qry_num = int((sample['images'].shape[1] - sup_size) / qry_size)
        target_inds = Variable(sample['labels'], requires_grad=False).cuda(self.device)

        x_support_l = sample_images[:, :sup_size, :, :, :224]
        x_support_r = sample_images[:, :sup_size, :, :, 224:]
        x_query_l = sample_images[:, sup_size:, :, :, :224]
        x_query_r = sample_images[:, sup_size:, :, :, 224:]

        dists_l = self.forward_one_side(x_support_l, x_query_l, batch, sup_num, sup_size, qry_num, qry_size)
        dists_r = self.forward_one_side(x_support_r, x_query_r, batch, sup_num, sup_size, qry_num, qry_size)

        dists = torch.cat([dists_l, dists_r], dim=1)
        pred = self.classifier(dists).view(batch, qry_num)
        pred_label = torch.round(pred)

        criterion = nn.BCELoss()
        loss = criterion(pred, target_inds)

        acc = torch.eq(pred_label, target_inds).float().mean()
        acc_means = torch.eq(pred_label, target_inds).view(batch, -1).float().mean(1)

        return loss, {
            'loss': loss.item(),
            'acc': acc.item(),
            'acc_means': acc_means
        }
