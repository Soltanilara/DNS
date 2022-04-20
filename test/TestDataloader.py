import unittest

from torchvision import transforms, datasets

from utils.loader import ConsecLoader
from utils.utils import summarizeDataset


class FSLDataset:
    def __init__(self):
        self.batch_size = 10
        self.sup_size = 10
        self.qry_size = 10
        self.qry_num = 6

        root_dir = "/home/nick/dataset/all/"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset_train = datasets.coco.CocoDetection(
            root=root_dir,
            annFile='../coco/exclude_Bainer2F_Kemper3F/train.json',
            transform=transform
        )
        self.dataset_summary = summarizeDataset(self.dataset_train)

    def get_data(self):
        loader = ConsecLoader(self.batch_size, self.sup_size, self.qry_size, self.qry_num, self.dataset_train,
                              self.dataset_summary)
        batch = loader.get_batch()
        pass


class TestDataloader(unittest.TestCase):

    def test_t(self):
        dataset = FSLDataset()
        data = dataset.get_data()


if __name__ == '__main__':
    unittest.main()
