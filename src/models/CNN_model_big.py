import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np

class Resize_CNN(nn.Module):
    def __init__(self, in_channels:int):
        super(Resize_CNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # WOULD BE BETTER TO ONLY HAVE ONE FULLY CONNECTED LAYER
        # In for Linear is out_channels * (img_size / num_maxpools)**2
        self.fc1 = nn.Linear(128*8**2, 1)
        self.fc2 = nn.Linear(128*8**2, 1)

    def forward(self, x):
        #print(f'Input Shape: {x.shape}')
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        #print(f'Output Shape: {x.shape}')
        label_width = self.fc1(x)
        #print(f'label_width Shape: {label_width.shape}')
        label_height = self.fc2(x)
        #print(f'label_height Shape: {label_height.shape}')

        return {'label_width': label_width, 'label_height' : label_height}

if __name__ == '__main__':
    model = Resize_CNN(in_channels=1)
    # Give 64 images of 1x256x256
    x = torch.randn(64,1,256,256)
    output = model(x)
    print(output['label_width'].shape)
    print(output['label_height'].shape)