import torch
import torch.nn as nn
import torchvision.models as models

from math import sqrt


class CNNQNetwork(nn.Module):
    def __init__(self,
                 h,
                 w,
                 size_action,
                 channels,
                 kernel_size=5,
                 stride=2,
                 ):
        super(CNNQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn3 = nn.BatchNorm2d(16)
        self.output = nn.Linear(96, size_action)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.output(x.view(x.size(0), -1))


class EnsembleCNNQNetwork(nn.Module):
    def __init__(self,
                 h,
                 w,
                 size_action,
                 channels,
                 kernel_size=5,
                 stride=2,
                 num_heads=10,
                 ):
        super(EnsembleCNNQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=stride)
        # self.bn3 = nn.BatchNorm2d(16)
        self.out_w = nn.Parameter(torch.randn(num_heads, 96, size_action) * sqrt(2 / (96 + size_action)),
                                  requires_grad=True)
        self.out_b = nn.Parameter(torch.zeros(1, num_heads, size_action), requires_grad=True)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        output = torch.einsum('bi,nio->bno', x, self.out_w) + self.out_b
        return output


class VGGQNetwork(nn.Module):
    def __init__(self, size_action):
        super(VGGQNetwork, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.fc.in_features
        self.vgg16.fc = nn.Linear(num_ftrs, size_action)


    def forward(self, x):
        return None
