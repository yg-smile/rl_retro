from algos.ddqn import DQN
from algos.buffer import ReplayBuffer

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

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)
device = torch.device("cuda:0")


# prepare environment
env_name = 'Castlevania-aria-of-sorrow_3'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "envs"))
print(env_name in retro.data.list_games(inttype=retro.data.Integrations.ALL))
env = retro.make(env_name, inttype=retro.data.Integrations.ALL)

# simulation setup
config = {
    'image_h': 160,
    'image_w': 240,
    'size_action': 8,  # 7 basic button + 1 combined button (up + B)
    'kernel_size': 4,
    'stride': 4,
    'frame_skip': 2,
    'lr': 0.0001,
    'copy_steps': 1000,
    'discount': 0.99,
    'eps_max': 1.,
    'eps_min': 0.05,
    'exploration_steps': 10000,
    'batch_size': 32,
    'replay_buffer_size': 50000,
    'steps_before_train': 1000,
    'seed': 1,
    'max_episode': 10000,
}
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
dqn = DQN(config)
# Q = torch.load('./model/Q.pth.tar')
# dqn.Q.load_state_dict(Q['state_dict'])
# Q_tar = torch.load('./model/Q_tar.pth.tar')
# dqn.Q_tar.load_state_dict(Q_tar['state_dict'])
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/dqn_{env:}_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             env=env_name,
                             date=datetime.datetime.now()))

frame_skip = config['frame_skip']
obs = env.reset()
obs_tensor = dqn.phi(obs)
obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
next_obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)

steps = 0
steps_before_train = config['steps_before_train']
for i_episode in range(config['max_episode']):
    obs = env.reset()
    done = False
    t = 0
    ret = 0.
    while done is False:
        obs_tensor_skip = torch.cat(list(obs_queue)).to('cuda:0')[None, :].double() / 255.0
        action = dqn.act_probabilistic(obs_tensor_skip)
        action_onehot = np.zeros(12, dtype='int8')
        action_onehot[action_map[action]] = 1

        reward_skip = 0.
        done_skip = False
        env.render()
        for ii in range(frame_skip):
            obs_queue.append(dqn.phi(obs))
            next_obs, reward, done, info = env.step(action_onehot)
            if done is True:
                obs_queue = deque([dqn.phi(obs)] * frame_skip, maxlen=frame_skip)
                next_obs_queue = deque([dqn.phi(next_obs)] * frame_skip, maxlen=frame_skip)
                break
            next_obs_queue.append(dqn.phi(next_obs))
            reward_skip += reward
            done_skip = done_skip or done
            obs = copy.deepcopy(next_obs)

        if t > 2:
            buffer.append_memory(obs=torch.cat(list(obs_queue)),
                                 action=torch.from_numpy(np.array([action])).to(device),
                                 reward=torch.from_numpy(np.array([reward_skip])).to(device),
                                 next_obs=torch.cat(list(next_obs_queue)),
                                 done=done_skip)
        if steps > steps_before_train:
            dqn.update(buffer)

        t += 1
        steps += 1
        ret += reward_skip

        if done:
            print("Episode {} return {} (total steps: {})".format(i_episode, ret, steps))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

torch.save({'state_dict': dqn.Q.state_dict()}, './model/Q.pth.tar')
torch.save({'state_dict': dqn.Q_tar.state_dict()}, './model/Q_tar.pth.tar')

env.close()
train_writer.close()


def test_model(episodes):
    dqn = DQN(config)
    Q = torch.load('./model/Q.pth.tar')
    dqn.Q.load_state_dict(Q['state_dict'])
    Q_tar = torch.load('./model/Q_tar.pth.tar')
    dqn.Q_tar.load_state_dict(Q_tar['state_dict'])

    steps = 0
    for i_episode in range(episodes):
        obs = env.reset()
        obs_tensor = dqn.phi(obs)
        obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
        next_obs_queue = deque([obs_tensor] * frame_skip, maxlen=frame_skip)
        done = False
        t = 0
        ret = 0.
        while done is False:
            obs_tensor_skip = torch.cat(list(obs_queue)).to('cuda:0')[None, :].double() / 255.0
            action = dqn.act_deterministic(obs_tensor_skip)
            action_onehot = np.zeros(12, dtype='int8')
            action_onehot[action_map[action]] = 1

            reward_skip = 0.
            done_skip = False
            env.render()
            time.sleep(0.01)
            for ii in range(frame_skip):
                obs_queue.append(dqn.phi(obs))
                next_obs, reward, done, info = env.step(action_onehot)
                if done is True:
                    obs_queue = deque([dqn.phi(obs)] * frame_skip, maxlen=frame_skip)
                    next_obs_queue = deque([dqn.phi(next_obs)] * frame_skip, maxlen=frame_skip)
                    break
                next_obs_queue.append(dqn.phi(next_obs))
                reward_skip += reward
                done_skip = done_skip or done
                obs = copy.deepcopy(next_obs)

            t += 1
            steps += 1
            ret += reward_skip

            if done:
                print("Testing episode {} return {} (total steps: {})".format(i_episode, ret, steps))
        train_writer.add_scalar('Performance/episodic_return', ret, i_episode)




