import torch
import torch.nn as nn
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from numpy.random import uniform
from numpy.random import choice
from skimage.transform import resize
from skimage.color import rgb2gray

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)


class BDQN:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.copy_steps = config['copy_steps']  # copy steps
        self.exploration_steps = config['exploration_steps']  # length of epsilon greedy exploration
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size
        self.frame_skip = config['frame_skip']
        self.double_q = config['double_q']
        self.num_heads = config['num_heads']

        self.size_action = config['size_action']
        self.image_h = config['image_h']
        self.image_w = config['image_w']
        self.kernel_size = config['kernel_size']
        self.stride = config['stride']

        self.Q = QNetwork(h=self.image_h,
                          w=self.image_w,
                          channels=self.frame_skip,
                          size_action=self.size_action,
                          kernel_size=self.kernel_size,
                          stride=self.stride)
        self.Q_tar = QNetwork(h=self.image_h,
                              w=self.image_w,
                              channels=self.frame_skip,
                              size_action=self.size_action,
                              kernel_size=self.kernel_size,
                              stride=self.stride)

        self.optimizer_Q = torch.optim.RMSprop(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

        self.h = 0  # exploration head depending on the episode

    def sample_index(self):
        self.h = np.random.choice(self.num_heads)

    def phi(self, img):
        # input: img: uint8 numpy array with 3 color channels with shape (h, w, 3)
        # output: tensor with 1 channel with shape (1, h, w)
        grayscale = np.uint8(255 * rgb2gray(img))
        return torch.from_numpy(grayscale)[None, :, :]

    def update(self, buffer):
        t = buffer.sample(self.batch_size)

        obs = t.obs.to('cuda:0').double() / 255.
        action = t.action
        reward = t.reward
        next_obs = t.next_obs.to('cuda:0').double() / 255.
        done = t.done
        mask = t.mask

        with torch.no_grad():
            if self.double_q is True:
                next_action = torch.max(self.Q(next_obs), axis=2, keepdim=True)[1]
                next_Q_all = self.Q_tar(next_obs)
                next_Q = next_Q_all.gather(2, next_action)[:, :, 0]
                Q_target = reward + self.discount * (~done) * next_Q
            else:
                Q_target = reward + self.discount * (~done) * \
                           torch.max(self.Q_tar(next_obs), axis=2, keepdim=False)[0]

        Q_all = self.Q(obs)
        Q = Q_all.gather(2, action[:, None, :].repeat(1, self.num_heads, 1))[:, :, 0]

        loss_Q = torch.mean(mask * (Q - Q_target) ** 2)

        self.optimizer_Q.zero_grad()
        loss_Q.backward()
        self.optimizer_Q.step()

        self.training_step += 1
        if self.training_step % self.copy_steps == 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())

    def act_probabilistic(self, observation: torch.Tensor):
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=2)
        self.Q.train()
        if uniform() < 0.001:
            return choice(self.size_action)
        else:
            return a[:, self.h].item()

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(torch.mean(Q, axis=1), axis=1)
        self.Q.train()
        return a.item()


class QNetwork(nn.Module):
    def __init__(self,
                 h,
                 w,
                 size_action,
                 channels,
                 kernel_size=5,
                 stride=2,
                 num_heads=10,
                 ):
        super(QNetwork, self).__init__()

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
