import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
    
class ResNet34(nn.Module):
    def __init__(self, in_channels, resblock, out_act=None):
        super().__init__()

        self.out_act = out_act

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1= torch.nn.Linear(512, 1)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = input.reshape(input.shape[0], -1)


        label_width = self.fc1(input)
        label_height = self.fc2(input)

        if self.out_act is not None:
            label_width = nn.Sigmoid(label_width)
            label_height = nn.Sigmoid(label_width)

        return {'label_width': label_width, 'label_height' : label_height}