import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import hflip

# Define a custom dataset class for steering data
class SteerDataset(Dataset):
    def __init__(self, df, transform=None, flip=True):
        # One hot encode the 'direction' column
        cats = ['direction_left', 'direction_right', 'direction_straight']
        dummies = pd.get_dummies(df['direction'], prefix='direction')
        dummies = dummies.T.reindex(cats).T.fillna(0)
        df = pd.concat([df, dummies], axis=1)
        
        # Reorder columns for the dataset
        df = df[[
            'path',
            'direction_left',
            'direction_right',
            'direction_straight',
            'steering',
            'throttle',
        ]]
        
        # Store the image labels and transformation options
        self.img_labels = df
        self.transform = transform
        self.flip = flip

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the image file path
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        
        # Apply the specified image transformation
        image = self.transform(image)

        # Extract direction, steering angle, and throttle as tensors
        direction = torch.tensor(
            self.img_labels.iloc[idx, 1:4], 
            dtype=torch.float32
        )

        steering_angle = torch.tensor(
            self.img_labels.iloc[idx, 4], 
            dtype=torch.float32
        )

        throttle = torch.tensor(
            self.img_labels.iloc[idx, 5], 
            dtype=torch.float32
        )

        # Apply horizontal flip with a 50% probability
        if self.flip and torch.rand(1) < 0.5:
            image = hflip(image)
            direction[[0,1]] = direction[[1,0]]
            steering_angle = 1 - steering_angle
            
        # Return a dictionary containing image and label data
        return {
            'image': image,
            'direction': direction,
            'steering_angle': steering_angle,
            'throttle': throttle,
        }
