import torch
from torch import nn
from torch.autograd import Variable


def load_model(**kwargs):
    """
    Loads the network model
    Arg:
        FCdim: latent space dimensionality
    """
    FCdim = kwargs['FCdim']
    device = kwargs['device']

    encoder = torch.hub.load('facebookresearch/swav:main', 'resnet50') #pretrained ResNet-50
    for param in encoder.parameters():
        param.requires_grad = False  #freezes the pretrained model parameters
    encoder.fc = nn.Linear(2048, FCdim)

    cov_module = nn.Sequential(
        nn.Linear(FCdim, FCdim),
        nn.Tanh(),
        nn.Linear(FCdim, FCdim),
        nn.Softplus()
    )

    classifier = nn.Sequential(
        nn.Linear(FCdim, 100),
        nn.Linear(100, 1),
        nn.Sigmoid()
    )

    return StampNet(encoder, cov_module, classifier, device)


class StampNet(nn.Module):
    def __init__(self, encoder, cov_module, classifier, device):
        super(StampNet, self).__init__()

        self.device = device
        if self.device == 'cpu':
            self.encoder = encoder
            self.cov_module = cov_module
            self.classifier = classifier
        else:
            self.encoder = encoder.cuda(self.device)
            self.cov_module = cov_module.cuda(self.device)
            self.classifier = classifier.cuda(self.device)

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

        x_support = sample_images[:, :sup_size]
        x_query = sample_images[:, sup_size:]

        # target indices are 0 ... n_way-1
        # target_inds = torch.arange(0, 2).view(1, n_way, 1, 1).expand(batch, n_way, qry_size, 1).long()

        # concatenate and prepare images of the support and query sets for inputting to the network
        x = torch.cat([x_support.contiguous().view(batch * sup_size, *x_support.size()[-3:]),
                       x_query.contiguous().view(batch * qry_num * qry_size, *x_query.size()[-3:])], 0)
        z = self.encoder.forward(x)
        indiv_protos = z
        indiv_eigs = self.cov_module.forward(z) + 1e-8

        proto_indiv_sup = indiv_protos[:batch * sup_size].view(batch, sup_num, sup_size, -1)
        proto_qry = indiv_protos[batch * sup_size:].view(batch, qry_num, qry_size, -1)
        eigs_indiv_sup = indiv_eigs[:batch * sup_size].view(batch, sup_num, sup_size, -1)
        eigs_qry = indiv_eigs[batch * sup_size:].view(batch, qry_num, qry_size, -1)
        proto_sup = torch.mean(proto_indiv_sup, 2)
        eigs_sup = torch.mean(eigs_indiv_sup, 2)
        proto_qry = torch.mean(proto_qry, 2)
        eigs_qry = torch.mean(eigs_qry, 2)

        # proto_qry = proto_qry.view(batch, qry_num * qry_size, -1)
        # eigs_qry = eigs_qry.view(batch, qry_num * qry_size, -1)

        proto_sup = proto_sup.view(batch, 1, -1).expand(-1, qry_num, -1)
        eigs_sup = eigs_sup.view(batch, 1, -1).expand(-1, qry_num, -1)

        diff = proto_sup - proto_qry
        dists = (diff / (eigs_sup + eigs_qry)).view(batch * qry_num, -1) * diff.view(batch * qry_num, -1)

        pred = self.classifier(dists).view(batch, qry_num)
        pred_label = torch.round(pred)

        loss = nn.BCELoss()
        loss_val = loss(pred, target_inds)

        # loss_val = -log_p_y.gather(3, target_inds).squeeze().view(
        #     -1).mean()  # sum negative log softmax for the correct class, normalize by number of entries
        # _, y_hat = log_p_y.max(3)
        acc_val = torch.eq(pred_label, target_inds).float().mean()
        acc_means = torch.eq(pred_label, target_inds).view(batch, -1).float().mean(1)
        #
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'acc_means': acc_means
        }
