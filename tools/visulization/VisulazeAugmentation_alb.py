import os.path as osp

from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.path_imgs = glob(osp.join(root, '*.png'))

    def __getitem__(self, idx):
        return cv2.imread(self.path_imgs[idx])

    def __len__(self):
        return len(self.path_imgs)


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        return tuple(zip(*batch))
    else:
        return tuple(batch)


if __name__ == '__main__':
    root_imgs = '/path/to/image/folder'

    trfm = A.Compose([
        A.Resize(height=224, width=448),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5),
        A.CoarseDropout(max_holes=2, min_holes=1, max_height=224, max_width=112, min_height=20,
                        min_width=20, p=0.2),
        ToTensorV2()
    ])

    trfm_light = A.Compose([
        A.Resize(height=224, width=448),
        A.Rotate(limit=5),
        A.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5),
        A.CoarseDropout(max_holes=2, min_holes=1, max_height=224, max_width=112, min_height=20,
                        min_width=20, p=0.2),
        ToTensorV2()
    ])

    dataset = TestDataset(root=root_imgs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for i in loader:
        img_raw = i[0]

        img_aug = trfm_light(image=img_raw)
        img_aug = img_aug['image'].permute([1, 2, 0]).numpy()

        fig = plt.figure(figsize=(6, 2))
        fig.add_subplot(121)
        plt.imshow(img_raw)
        plt.title('Original')
        plt.axis('off')
        fig.add_subplot(122)
        plt.imshow(img_aug)
        plt.title('Augmented')
        plt.axis('off')

        plt.show()
        pass
