import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


class ConsecLoader:
    def __init__(self, batch, sup_size, qry_size, qry_num, dataset):
        self.batch = batch
        self.sup_size = sup_size
        self.qry_size = qry_size
        self.qry_num = qry_num
        self.dataset = dataset
        self.lap2CatId = dataset.summary['lap2CatId']
        self.PN2CatId = dataset.summary['PN2CatId']
        self.location2Lap = dataset.summary['location2Lap']
        self.catId2Prob = dataset.summary['catId2Prob']

    def get_sample(self, cls, debug=False):
        label = []
        debug_info = {
            'sup_cat_id': [],
            'sup_inds': [],
            'qry_pos_cat_id': [],
            'qry_pos_inds': [],
            'qry_neg_cat_id': [],
            'qry_neg_inds': [],
        }

        # sample a support set
        cat = self.dataset.coco.loadCats(cls)[0]
        sup_img_ids = sorted(self.dataset.coco.getImgIds(catIds=cls))
        sup_l = 0 if len(sup_img_ids) <= self.sup_size else np.random.randint(max(len(sup_img_ids) - self.sup_size, 1))
        sup_inds = sup_img_ids[sup_l: sup_l + self.sup_size]
        if debug:
            debug_info['sup_cat_id'].append(cat)
            debug_info['sup_inds'].append(sup_inds)
        else:
            loader_sup = DataLoader(dataset=Subset(self.dataset, sup_inds), shuffle=False, batch_size=self.sup_size,
                                    pin_memory=True, drop_last=False, collate_fn=self.dataset.collate_fn)
            if self.dataset.batch_transform:
                sample = self.dataset.transform.apply(next(iter(loader_sup)))
            else:
                sample = next(iter(loader_sup))[0]

        # select laps with/without replacement and sample a positive query set from each
        laps_direction = self.location2Lap[cat['location']][cat['direction']]
        laps = list(np.random.choice(list(laps_direction), int(self.qry_num/2), replace=True))
        for lap in laps:
            try:
                qry_pos_cat_id = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            except:
                print(lap)
            qry_pos_img_ids = self.dataset.coco.getImgIds(catIds=qry_pos_cat_id)
            qry_pos_l = sup_l
            while qry_pos_cat_id == cat['id'] and qry_pos_l == sup_l:
                qry_pos_l = 0 if len(qry_pos_img_ids) <= self.qry_size else np.random.randint(max(len(qry_pos_img_ids) - self.qry_size, 1))
            qry_pos_inds = qry_pos_img_ids[qry_pos_l: qry_pos_l + self.qry_size]

            if debug:
                debug_info['qry_pos_cat_id'].append(qry_pos_cat_id)
                debug_info['qry_pos_inds'].append(qry_pos_inds)
            else:
                loader_qry = DataLoader(dataset=Subset(self.dataset, qry_pos_inds), shuffle=False, batch_size=self.qry_size,
                                        pin_memory=True, drop_last=False, collate_fn=self.dataset.collate_fn)
                if self.dataset.batch_transform:
                    sample_qry = self.dataset.transform.apply(next(iter(loader_qry)))
                else:
                    sample_qry = next(iter(loader_qry))[0]
                sample = torch.cat([sample, sample_qry], dim=0)
                label.append(1.)

        # select laps with/without replacement and sample a negative query set from each
        laps = list(np.random.choice(list(laps_direction), self.qry_num-int(self.qry_num/2), replace=True))
        for lap in laps:
            try:
                pos_cat_id = [i for i in self.lap2CatId[lap] if self.dataset.coco.loadCats(i)[0]['name'] == cat['name']][0]
            except:
                pass
            qry_neg_cat_id = pos_cat_id - 1
            qry_neg_img_ids = self.dataset.coco.getImgIds(catIds=qry_neg_cat_id)

            qry_neg_ls = range(qry_neg_img_ids[-1] - 50, qry_neg_img_ids[-1]+1)  # use 50 negative images
            qry_neg_l = np.random.choice(qry_neg_ls)
            qry_neg_inds = [i for i in range(qry_neg_l, qry_neg_l+self.qry_size)]

            if debug:
                debug_info['qry_neg_cat_id'].append(qry_neg_cat_id)
                debug_info['qry_neg_inds'].append(qry_neg_inds)
            else:
                loader_qry = DataLoader(dataset=Subset(self.dataset, qry_neg_inds), shuffle=False, batch_size=self.qry_size,
                                        pin_memory=True, drop_last=False, collate_fn=self.dataset.collate_fn)
                if self.dataset.batch_transform:
                    sample_qry = self.dataset.transform.apply(next(iter(loader_qry)))
                else:
                    sample_qry = next(iter(loader_qry))[0]
                sample = torch.cat([sample, sample_qry], dim=0)
                label.append(0.)

        if debug:
            return debug
        else:
            return sample, label

    def get_batch(self, debug=False):
        labels = []
        if hasattr(self.dataset[0][0], 'shape'):
            img_size = self.dataset[0][0].shape
        else:
            img_size = [3, self.dataset.transform.param_range['Resize']['height'],
                        self.dataset.transform.param_range['Resize']['width']]
        sample_batch = torch.empty([self.batch, self.sup_size+self.qry_size*self.qry_num]+list(img_size))
        cls = np.random.choice(self.PN2CatId['positive'], self.batch, replace=True)
        if debug:
            return cls
        else:
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.pos_catIds = dataset.summary['PN2CatId']['positive']

    def get_all_landmarks(self):
        landmarks = None
        for catId in self.pos_catIds:
            ids_img_sup = self.dataset.coco.getImgIds(catIds=catId)
            loader = DataLoader(dataset=Subset(self.dataset, ids_img_sup), shuffle=False, batch_size=len(ids_img_sup),
                                pin_memory=True, drop_last=False, collate_fn=self.dataset.collate_fn)
            landmark = next(iter(loader))[0]
            landmark = landmark.view([1] + list(landmark.shape))
            landmarks = torch.cat([landmarks, landmark], dim=0) if landmarks is not None else landmark
        return landmarks