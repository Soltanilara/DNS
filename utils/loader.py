import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class ConsecLoader:
    def __init__(self, batch, sup_size, qry_size, qry_num, dataset, summary):
        self.batch = batch
        self.sup_size = sup_size
        self.qry_size = qry_size
        self.qry_num = qry_num
        self.dataset = dataset
        self.lap2CatId = summary['lap2CatId']
        self.PN2CatId = summary['PN2CatId']
        self.location2Lap = summary['location2Lap']

    def get_sample(self, cls):
        label = []

        # sample a support set
        cat = self.dataset.coco.loadCats(cls)[0]
        sup_img_ids = sorted(self.dataset.coco.getImgIds(catIds=cls))
        sup_l = 0 if len(sup_img_ids) <= self.sup_size else np.random.randint(max(len(sup_img_ids) - self.sup_size, 1))
        inds_sup = sup_img_ids[sup_l: sup_l + self.sup_size]
        loader_sup = DataLoader(dataset=Subset(self.dataset, inds_sup), shuffle=False, batch_size=self.sup_size,
                                pin_memory=False, drop_last=False)
        sample = next(iter(loader_sup))[0]

        # select laps with/without replacement and sample a positive query set from each
        laps_direction = self.location2Lap[cat['location']][cat['direction']]
        laps = list(np.random.choice(list(laps_direction), int(self.qry_num/2), replace=True))
        for lap in laps:
            catId = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            qry_img_ids = self.dataset.coco.getImgIds(catIds=catId)
            qry_l = sup_l
            while catId == cat['id'] and qry_l == sup_l:
                qry_l = 0 if len(qry_img_ids) <= self.qry_size else np.random.randint(max(len(qry_img_ids) - self.qry_size, 1))
            inds_qry = sup_img_ids[qry_l: qry_l + self.qry_size]
            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=False, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.append(1.)

        # select laps with/without replacement and sample a negative query set from each
        laps = list(np.random.choice(list(laps_direction), self.qry_num-int(self.qry_num/2), replace=True))
        for lap in laps:
            catId_pos = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            imgIds = self.dataset.coco.getImgIds(catIds=catId_pos-1)

            qry_ls = range(min(imgIds[0], imgIds[-1]-self.qry_size), imgIds[-1]+1)
            qry_l = np.random.choice(qry_ls)
            inds_qry = [i for i in range(qry_l, qry_l+self.qry_size)]

            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=False, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.append(0.)

        return sample, label

    def get_batch(self):
        sample_batch = torch.empty([self.batch, self.sup_size+self.qry_size*self.qry_num]+list(self.dataset[0][0].shape))
        labels = []
        cls = np.random.choice(self.PN2CatId['positive'], self.batch, replace=True)
        for b, c in zip(range(self.batch), cls):
            sample_batch[b], label = self.get_sample(c.item())
            labels.append(label)
        return ({
            'images': sample_batch,
            'labels': torch.tensor(labels),
            'batch': self.batch,
            'sup_size': self.sup_size,
            'qry_size': self.qry_size,
            'qry_num': self.qry_num
        })


class RandNegLoader(ConsecLoader):
    def get_sample(self, cls):
        label = []

        # sample support set
        cat = self.dataset.coco.loadCats(cls)[0]
        sup_img_ids = sorted(self.dataset.coco.getImgIds(catIds=cls))
        sup_l = 0 if len(sup_img_ids) <= self.sup_size else np.random.randint(max(len(sup_img_ids) - self.sup_size, 1))
        inds_sup = sup_img_ids[sup_l: sup_l + self.sup_size]
        loader_sup = DataLoader(dataset=Subset(self.dataset, inds_sup), shuffle=False, batch_size=self.sup_size,
                                pin_memory=False, drop_last=False)
        sample = next(iter(loader_sup))[0]

        # sample laps without replacement and sample a positive query set from each
        laps_direction = self.direction2Lap[cat['direction']]
        laps = list(np.random.choice(list(laps_direction), int(self.qry_num / 2), replace=True))
        for lap in laps:
            catId = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            qry_img_ids = self.dataset.coco.getImgIds(catIds=catId)
            qry_l = sup_l
            while catId == cat['id'] and qry_l == sup_l:
                qry_l = 0 if len(qry_img_ids) <= self.qry_size else np.random.randint(
                    max(len(qry_img_ids) - self.qry_size, 1))
            inds_qry = sup_img_ids[qry_l: qry_l + self.qry_size]
            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=False, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.extend([1.] * self.qry_size)

        # sample laps without replacement and sample a negative query set from each
        laps = list(np.random.choice(list(laps_direction), int(self.qry_num / 2), replace=True))
        for lap in laps:
            inds_qry = []
            for catId in self.lap2CatId[lap]:
                img_ids = self.dataset.coco.getImgIds(catIds=catId)
                if self.dataset.coco.loadCats(catId)[0]['name'] != cat['name']:
                    inds_qry += img_ids
            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=True, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.extend([0.] * self.qry_size)

        return sample, label


class TestLoader:
    def __init__(self, dataset, summary):
        self.dataset = dataset
        self.pos_catIds = summary['PN2CatId']['positive']

    def get_all_landmarks(self):
        landmarks = None
        for catId in self.pos_catIds:
            ids_img_sup = self.dataset.coco.getImgIds(catIds=catId)
            loader = DataLoader(dataset=Subset(self.dataset, ids_img_sup), shuffle=False, batch_size=len(ids_img_sup),
                                pin_memory=False, drop_last=False)
            landmark = next(iter(loader))[0]
            landmark = landmark.view([1] + list(landmark.shape))
            landmarks = torch.cat([landmarks, landmark], dim=0) if landmarks is not None else landmark
        return landmarks