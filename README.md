# rl_retro
Play GBA games using deep reinforcement learning.


## Castlevania: Aria of Sorrow, first boss

State: 160 * 240 grayscale pixels.

Action: Discrete(8) - 1.attack 2.jump 3-6.up down left right 7.back dash 8.(combined) up+attack.

Reward formulation: negative of boss (Creaking skull) HP loss.

Done condition: boss defeated or Soma dies.


After 80 episodes of play:
<p float="left">
  <img src="https://user-images.githubusercontent.com/49927412/103398702-b8eccd80-4af2-11eb-85a4-bec2dd5ea14d.gif" width="300" />
</p>
