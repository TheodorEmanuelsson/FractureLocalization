import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np

class Resize_CNN(nn.Module):
    def __init__(self, in_channels:int):
        super(Resize_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, 
                                kernel_size=(3,3), stride = (1,1), padding = (1,1))
        # In for Linear is out_channels * (img_size / num_maxpools)**2
        self.fc1 = nn.Linear(16*64*64, 1)

    def forward(self, x):
        #print(f'Input Shape: {x.shape}')
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        #print(f'Output Shape: {x.shape}')
        label_width = self.fc1(x)
        #print(f'label_width Shape: {label_width.shape}')
        label_height = self.fc1(x)
        #print(f'label_height Shape: {label_height.shape}')

        return {'label_width': label_width, 'label_height' : label_height}

if __name__ == '__main__':
    model = Resize_CNN(in_channels=1)
    # Give 64 images of 1x256x256
    x = torch.randn(64,1,256,256)
    output = model(x)
    print(output['label_width'].shape)
    print(output['label_height'].shape)