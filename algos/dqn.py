import torch
import torch.nn as nn
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import binomial
from numpy.random import choice

from algos.models import CNNQNetwork
from algos.models import VGGQNetwork
from algos.algos_utils import phi

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

        self.phi = phi
        self.Q = CNNQNetwork(h=self.image_h,
                             w=self.image_w,
                             channels=self.frame_skip,
                             size_action=self.size_action,
                             kernel_size=self.kernel_size,
                             stride=self.stride)
        self.Q_tar = CNNQNetwork(h=self.image_h,
                                 w=self.image_w,
                                 channels=self.frame_skip,
                                 size_action=self.size_action,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride)
        # self.Q = VGGQNetwork(size_action=self.size_action)
        # self.Q_tar = VGGQNetwork(size_action=self.size_action)

        self.optimizer_Q = torch.optim.RMSprop(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

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

