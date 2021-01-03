# rl_retro
Play GBA games using deep reinforcement learning.


## Castlevania: Aria of Sorrow, first boss

State: 160 * 240 grayscale pixels. Stacked 2 frames
Action: Discrete(8) - 1.attack 2.jump 3-6.up down left right 7.back dash 8.(combined) up+attack.
Reward formulation: negative of boss (Creaking skull) HP loss.
Done condition: boss defeated or Soma dies.

Using double deep Q learning (DDQN) with epsilon greedy exploration. Hyperparameters:
```
network structure: conv2d -> relu -> conv2d -> relu -> conv2d -> relu -> fully connected
discount factor: 0.99
stack frame: 2
optimizer: RMSprop
learning rate: 0.0001
target net copy steps: 1000
replay buffer size: 50000
minibatch size: 32
epsilon max: 1.0
epsilon min: 0.05
epsilon decreasing length: 10000
```

After 160 episodes of play:
<p float="left">
  <img src="https://user-images.githubusercontent.com/49927412/103485231-b7006400-4da9-11eb-9ae9-cc933da2103b.gif" width="350" />
  <img src="https://user-images.githubusercontent.com/49927412/103485485-8d483c80-4dab-11eb-81cf-890d49ce20d0.png" width="350" />
</p>

## Setup
Wrap a piece of game as a gym-like environment:

Step 1. install [gym-retro](https://retro.readthedocs.io/en/latest/getting_started.html). Then download and install the [Integration UI](https://retro.readthedocs.io/en/latest/integration.html#the-integration-ui)

Step 2. obtain the ROM of the game.

Step 3. follow the tutorial in gym-retro on how to integrate a game. In particular, how to define the start state, terminal condition, and reward function, as well as where to look for relevant variables of the game. The integrated game will consists of the ROM of the game, the .state file, data.json, metadata.json, and scenario.json.

### Tips for finding variables
In order to create the terminal condition and reward function we need to access relevant variables of the game play. For example, the HP of the main character, the HP of the enemy, etc. For GBA games these variables are stored in the [RAM](https://problemkaputt.de/gbatek.htm#gbamemorymap). There are many emulators that can access the memory address and values. I would recommend [mGBA](https://mgba.io/) since its Search Memory (under the Tools tab in the main menu) is very easy to use. In addition, it can search a particular value without specifying the width of that value (1 byte, 2 bytes, etc.). This helped my variable searching process so much because it turned out that different boss's HP in Aria of Sorrow can have different byte width. 

There are other great tools such as [BizHawk](https://github.com/TASVideos/BizHawk) can be used for finding variables. But it seems that it can only search RAM portion within 03000000-03007FFF; whereas the variables of Aria of Sorrow are actually stored somewhere in 02000000-0203FFFF.




