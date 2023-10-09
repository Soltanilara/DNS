import os.path as osp

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


class TestDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.path_imgs = glob(osp.join(root, '*.jpg'))

    def __getitem__(self, idx):
        return Image.open(self.path_imgs[idx])

    def __len__(self):
        return len(self.path_imgs)


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        return tuple(zip(*batch))
    else:
        return tuple(batch)


if __name__ == '__main__':
    root_imgs = '/path/to/image/folder'

    trfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.05, 0.2), ratio=(0.2, 5), p=1),
    ])

    dataset = TestDataset(root=root_imgs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for i in loader:
        img_raw = i[0]
        img_aug = trfm(img_raw)
        img_aug = img_aug.permute([1, 2, 0]).numpy()

        fig = plt.figure()
        fig.add_subplot(211)
        plt.imshow(img_raw)
        plt.title('Original')
        plt.axis('off')
        fig.add_subplot(212)
        plt.imshow(img_aug)
        plt.title('Augmented')
        plt.axis('off')

        plt.show()
        pass
