from random import randint
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset


# class RandNegLoader:
#     def __init__(self, batch, n_way, n_support, n_query, dataset, targets, train):
#         self.batch = batch
#         self.n_way = n_way
#         self.dataset = dataset
#         self.n_class_tot = len(dataset.classes)
#         self.n_data_tot = len(dataset)
#         self.n_support = n_support
#         self.n_query = n_query
#         self.targets = targets
#
#     def get_sample(self):
#         sample = torch.empty([0, self.n_support + self.n_query] + list(self.dataset[0][0].shape))
#         k = np.random.choice(np.unique(range(np.floor(self.n_class_tot/2).astype(int))), 1, replace=False).item()
#         for i, cls in enumerate([2 * k, 2 * k + 1]):
#             if i == 0:
#                 weights = list(map(int, [x == y for (x, y) in zip(self.targets, [cls] * self.n_data_tot)]))
#             else:
#                 weights = list(map(int, [x != y for (x, y) in zip(self.targets, [cls - 1] * self.n_data_tot)]))
#             sampler = WeightedRandomSampler(weights, self.n_support + self.n_query, replacement=False)
#             loader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=self.n_support + self.n_query,
#                                 sampler=sampler, drop_last=False)  # , batch_size = n_support+n_query
#             sample_cls = next(iter(loader))[0].unsqueeze(dim=0)
#             sample = torch.cat([sample, sample_cls], dim=0)
#         return sample
#
#     def get_batch(self):
#         sample_batch = torch.empty([self.batch, self.n_way, self.n_support + self.n_query] + list(self.dataset[0][0].shape))
#         for b in range(self.batch):
#             sample_batch[b] = self.get_sample()
#         return ({
#             'images': sample_batch,
#             'batch': self.batch,
#             'n_way': self.n_way,
#             'n_support': self.n_support,
#             'n_query': self.n_query
#         })


class ConsecLoader:
    def __init__(self, batch, sup_size, qry_size, qry_num, dataset, summary):
        self.batch = batch
        self.sup_size = sup_size
        self.qry_size = qry_size
        self.qry_num = qry_num
        self.dataset = dataset
        self.lap2CatId = summary['superCat2CatId']
        self.PN2CatId = summary['PN2CatId']
        self.direction2Lap = summary['direction2SuperCat']

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
                qry_l = 0 if len(qry_img_ids) <= self.qry_size else np.random.randint(max(len(qry_img_ids) - self.qry_size, 1))
            inds_qry = sup_img_ids[qry_l: qry_l + self.qry_size]
            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=False, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.append(1.)

        # sample laps without replacement and sample a negative query set from each
        laps = list(np.random.choice(list(laps_direction), self.qry_num - int(self.qry_num / 2), replace=True))
        for lap in laps:
            catId_pos = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            pos_imgIds = self.dataset.coco.getImgIds(catIds=catId_pos)

            lap_catIds = self.lap2CatId[lap]
            lap_imgIds = []
            for catId in lap_catIds:
                lap_imgIds.extend(self.dataset.coco.getImgIds(catIds=catId))
            qry_ls = [i for i in range(0, pos_imgIds[0] - self.qry_size)] + \
                     [i for i in range(pos_imgIds[-1], lap_imgIds[-1] - self.qry_size)]
            qry_l = np.random.choice(qry_ls)
            inds_qry = [i for i in range(qry_l, qry_l + self.qry_size)]

            loader_qry = DataLoader(dataset=Subset(self.dataset, inds_qry), shuffle=False, batch_size=self.qry_size,
                                    pin_memory=False, drop_last=False)
            sample_qry = next(iter(loader_qry))[0]
            sample = torch.cat([sample, sample_qry], dim=0)
            label.append(0.)

        return sample, label

    def get_batch(self):
        sample_batch = torch.empty([self.batch, self.sup_size + self.qry_size * self.qry_num] + list(self.dataset[0][0].shape))
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
        self.PN2CatId = summary['PN2CatId']
        self.lap2CatId = summary['superCat2CatId']
        self.direction2Lap = summary['direction2SuperCat']

    def get_loader(self):
        return DataLoader(dataset=self.dataset, shuffle=False, batch_size=1, pin_memory=False, drop_last=False)

    def get_all_landmarks(self):
        pos_catIds = self.PN2CatId['positive']
        landmarks = None
        for catId in pos_catIds:
            ids_img_sup = self.dataset.coco.getImgIds(catIds=catId)
            loader = DataLoader(dataset=Subset(self.dataset, ids_img_sup), shuffle=False, batch_size=len(ids_img_sup),
                                pin_memory=False, drop_last=False)
            landmark = next(iter(loader))[0]
            landmark = landmark.view([1] + list(landmark.shape))
            landmarks = torch.cat([landmarks, landmark], dim=0) if landmarks is not None else landmark
        return landmarks