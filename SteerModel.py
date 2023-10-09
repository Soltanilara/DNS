import torchvision.models
import torchvision.transforms as T
import torch

# Define transformations for training images
TRAIN_TRANSFORMATIONS = T.Compose([
    T.ToTensor(),
    T.Resize((104, 224)),
    T.RandomApply(transforms=[
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15)
    ], p=0.75),
    T.RandomApply(transforms=[
        T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 3.0))
    ], p=0.5),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define transformations for testing images
TEST_TRANSFORMATIONS = T.Compose([
    T.ToTensor(),
    T.Resize((104, 224)),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define transformations for validation images
VALIDATION_TRANSFORMATIONS = T.Compose([
    T.ToTensor(),
    T.Resize((104, 224)),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define a custom neural network model called EfficientNet
class EfficientNet(torch.nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        # Load a pre-trained EfficientNet model with weights
        self.efficientnet = torchvision.models.efficientnet_b0(weights='DEFAULT')
        # Remove the last layer to modify it for our task
        self.efficientnet = torch.nn.Sequential(*(list(self.efficientnet.children())[:-1]))
        self.flatten = torch.nn.Flatten()
        # Define fully connected layers for additional processing
        self.fc1 = torch.nn.Linear(1280, 500)
        self.relu1 = torch.nn.ReLU()
        # Combine the output of fc1 with the 'direction' input
        self.fc2 = torch.nn.Linear(500+3, 100)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(100, 1)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x, direction):
        # Pass the input through the EfficientNet model
        x = self.efficientnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        # Concatenate 'direction' input with the output of fc1
        x = torch.cat((x, direction), 1)  
        x = self.fc2(x)
        x = self.relu2(x) 
        x = self.fc3(x)
        x = self.sigmoid1(x) 
        return x
