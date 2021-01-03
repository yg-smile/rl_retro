import torch
import torch.nn as nn
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import binomial
from numpy.random import choice
from skimage.transform import resize
from skimage.color import rgb2gray

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)


class DQN:
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

        with torch.no_grad():
            if self.double_q is True:
                next_action = torch.max(self.Q(next_obs), axis=1, keepdim=True)[1]
                next_Q_all = self.Q_tar(next_obs)
                next_Q = next_Q_all.gather(1, next_action)
                Q_target = reward + self.discount * (~done) * next_Q
            else:
                Q_target = reward + self.discount * (~done) * torch.max(self.Q_tar(next_obs), axis=1, keepdim=True)[0]

        Q_all = self.Q(obs)
        Q = Q_all.gather(1, action)

        loss_Q = torch.mean((Q - Q_target) ** 2)

        self.optimizer_Q.zero_grad()
        loss_Q.backward()
        self.optimizer_Q.step()

        self.training_step += 1
        if self.training_step % self.copy_steps == 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())

    def act_probabilistic(self, observation: torch.Tensor):
        # epsilon greedy:
        first_term = self.eps_max * (self.exploration_steps - self.training_step) / self.exploration_steps
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.size_action)
        else:
            self.Q.eval()
            Q = self.Q(observation)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=1)
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
        self.output = nn.Linear(96, size_action)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.output(x.view(x.size(0), -1))
