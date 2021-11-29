from random import randint
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader


class RandNegLoader:
    def __init__(self, batch, n_way, n_support, n_query, dataset, targets, train):
        self.batch = batch
        self.n_way = n_way
        self.dataset = dataset
        self.n_class_tot = len(dataset.classes)
        self.n_data_tot = len(dataset)
        self.n_support = n_support
        self.n_query = n_query
        self.targets = targets

    def get_batch(self):
        sample = torch.empty([0, self.n_support + self.n_query] + list(self.dataset[0][0].shape))
        k = np.random.choice(np.unique(range(np.floor(self.n_class_tot/2).astype(int))), 1, replace=False).item()
        for i, cls in enumerate([2 * k, 2 * k + 1]):
            if i == 0:
                weights = list(map(int, [x == y for (x, y) in zip(self.targets, [cls] * self.n_data_tot)]))
            else:
                weights = list(map(int, [x != y for (x, y) in zip(self.targets, [cls - 1] * self.n_data_tot)]))
            sampler = WeightedRandomSampler(weights, self.n_support + self.n_query, replacement=False)
            loader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=self.n_support + self.n_query, sampler=sampler,
                                drop_last=False)  # , batch_size = n_support+n_query
            sample_cls = next(iter(loader))[0].unsqueeze(dim=0)
            sample = torch.cat([sample, sample_cls], dim=0)
        return sample

    def extract_sample(self):
        sample_batch = torch.empty([self.batch, self.n_way, self.n_support + self.n_query] + list(self.dataset[0][0].shape))
        for b in range(self.batch):
            sample_batch[b] = self.get_batch()
        return ({
            'images': sample_batch,
            'batch': self.batch,
            'n_way': self.n_way,
            'n_support': self.n_support,
            'n_query': self.n_query
        })


class ConsecNegLoader(RandNegLoader):
    def get_batch(self):
        sample = torch.empty([0, self.n_support + self.n_query] + list(self.dataset[0][0].shape))
        k = np.random.choice(np.unique(range(np.floor(self.n_class_tot / 2).astype(int))), 1, replace=False).item()
        for i, cls in enumerate([2 * k, 2 * k + 1]):
            if i == 0:
                weights = list(map(int, [x == y for (x, y) in zip(self.targets, [cls] * self.n_data_tot)]))
            else:
                weights = list(map(int, [x != y for (x, y) in zip(self.targets, [cls - 1] * self.n_data_tot)]))
            sampler = WeightedRandomSampler(weights, self.n_support + self.n_query, replacement=False)
            loader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=self.n_support + self.n_query,
                                sampler=sampler,
                                drop_last=False)  # , batch_size = n_support+n_query
            sample_cls = next(iter(loader))[0].unsqueeze(dim=0)
            sample = torch.cat([sample, sample_cls], dim=0)
        return sample


if __name__ == '__main__':
    # TODO: test 11.py
    sampler = RandNegLoader()
