from algos.bdqn import BDQN
from algos.buffer import ReplayBufferBootstrap

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import datetime
import copy
import retro
import os
import matplotlib.pyplot as plt
import time
from torch.distributions.bernoulli import Bernoulli

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)
device = torch.device("cuda:0")


# simulation setup
config = {
    'env_name': 'Castlevania-aria-of-sorrow-2ndboss_2',
    'image_h': 160,
    'image_w': 240,
    'size_action': 8,  # 7 basic button + 1 combined button (up + B)
    'kernel_size': 4,
    'stride': 4,
    'frame_skip': 2,
    'double_q': True,
    'num_heads': 10,
    'mask_prob': 0.4,
    'lr': 0.0001,
    'copy_steps': 1000,
    'discount': 0.99,
    'batch_size': 32,
    'replay_buffer_size': 50000,
    'steps_before_train': 1000,
    'seed': 1,
    'max_episode': 10000,
}

# prepare environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "envs"))
print(config['env_name'] in retro.data.list_games(inttype=retro.data.Integrations.ALL))
env = retro.make(config['env_name'], inttype=retro.data.Integrations.ALL)

action_map = [0, 4, 5, 6, 7, 8, 10, [4, 0]]
# action dims:
# 0 - B (attack)
# 1 - unknown
# 2 - map
# 3 - unknown
# 4 - up
# 5 - down
# 6 - left
# 7 - right
# 8 - A (jump)
# 9 - unknown
# 10 - upper left button
# 11 - unknown

# running simulation
bdqn = BDQN(config)
Q = torch.load('./model/bootstrapQ_{}.pth.tar'.format(config['env_name']))
bdqn.Q.load_state_dict(Q['state_dict'])
Q_tar = torch.load('./model/bootstrapQ_tar_{}.pth.tar'.format(config['env_name']))
bdqn.Q_tar.load_state_dict(Q_tar['state_dict'])
buffer = ReplayBufferBootstrap(config)
train_writer = SummaryWriter(log_dir='tensorboard/bdqn_{env:}_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             env=config['env_name'],
                             date=datetime.datetime.now()))
mask_dist = Bernoulli(torch.tensor([config['mask_prob']]))
sample_shape = torch.tensor([config['num_heads']])

frame_skip = config['frame_skip']
obs = env.reset()
obs_tensor = bdqn.phi(obs)
obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
next_obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)

steps = 0
steps_before_train = config['steps_before_train']
for i_episode in range(config['max_episode']):
    obs = env.reset()
    bdqn.sample_index()
    done = False
    t = 0
    ret = 0.
    while done is False:
        obs_tensor_skip = torch.cat(list(obs_queue)).to('cuda:0')[None, :].double() / 255.0
        action = bdqn.act_probabilistic(obs_tensor_skip)
        action_onehot = np.zeros(12, dtype='int8')
        action_onehot[action_map[action]] = 1

        reward_skip = 0.
        done_skip = False
        # env.render()
        for ii in range(frame_skip):
            obs_queue.append(bdqn.phi(obs))
            next_obs, reward, done, info = env.step(action_onehot)
            if done is True:
                obs_queue = deque([bdqn.phi(obs)] * frame_skip, maxlen=frame_skip)
                next_obs_queue = deque([bdqn.phi(next_obs)] * frame_skip, maxlen=frame_skip)
                break
            next_obs_queue.append(bdqn.phi(next_obs))
            reward_skip += reward
            done_skip = done_skip or done
            obs = copy.deepcopy(next_obs)

        if t > 2:
            buffer.append_memory(obs=torch.cat(list(obs_queue)),
                                 action=torch.from_numpy(np.array([action])).to(device),
                                 reward=torch.from_numpy(np.array([reward_skip])).to(device),
                                 next_obs=torch.cat(list(next_obs_queue)),
                                 done=done_skip,
                                 mask=mask_dist.sample(sample_shape).squeeze()
                                 )
        if steps > steps_before_train:
            bdqn.update(buffer)

        t += 1
        steps += 1
        ret += reward_skip

        if done:
            print("Episode {} return {} (total steps: {})".format(i_episode, ret, steps))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

torch.save({'state_dict': bdqn.Q.state_dict()}, './model/bootstrapQ_{}.pth.tar'.format(config['env_name']))
torch.save({'state_dict': bdqn.Q_tar.state_dict()}, './model/bootstrapQ_tar_{}.pth.tar'.format(config['env_name']))

env.close()
train_writer.close()


def test_model(episodes):
    bdqn = BDQN(config)
    Q = torch.load('./model/bootstrapQ_{}.pth.tar'.format(config['env_name']))
    bdqn.Q.load_state_dict(Q['state_dict'])
    Q_tar = torch.load('./model/bootstrapQ_tar_{}.pth.tar'.format(config['env_name']))
    bdqn.Q_tar.load_state_dict(Q_tar['state_dict'])

    steps = 0
    for i_episode in range(episodes):
        obs = env.reset()
        obs_tensor = bdqn.phi(obs)
        obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
        next_obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
        done = False
        t = 0
        ret = 0.
        while done is False:
            obs_tensor_skip = torch.cat(list(obs_queue)).to('cuda:0')[None, :].double() / 255.0
            action = bdqn.act_deterministic(obs_tensor_skip)
            action_onehot = np.zeros(12, dtype='int8')
            action_onehot[action_map[action]] = 1

            reward_skip = 0.
            done_skip = False
            env.render()
            time.sleep(0.01)
            for ii in range(frame_skip):
                obs_queue.append(bdqn.phi(obs))
                next_obs, reward, done, info = env.step(action_onehot)
                if done is True:
                    obs_queue = deque([bdqn.phi(obs)] * frame_skip, maxlen=frame_skip)
                    next_obs_queue = deque([bdqn.phi(next_obs)] * frame_skip, maxlen=frame_skip)
                    break
                next_obs_queue.append(bdqn.phi(next_obs))
                reward_skip += reward
                done_skip = done_skip or done
                obs = copy.deepcopy(next_obs)

            t += 1
            steps += 1
            ret += reward_skip

            if done:
                print("Testing episode {} return {} (total steps: {})".format(i_episode, ret, steps))
        train_writer.add_scalar('TestPerformance/episodic_return', ret, i_episode)




