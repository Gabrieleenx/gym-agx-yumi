# source /opt/Algoryx/AGX-2.30.4.0/setup_env.bash

import torch
from torch import nn
import numpy as np
import gym
from gymEnv import YumiPegInHole
import time
def main():

    yumiGym = YumiPegInHole(max_episode_steps=300)
    yumiGym.init_render(sync_real_time=True)
    obs = yumiGym.reset()
    action = np.zeros(16, dtype=np.float32)
    action[7] = 0.2
    #action[9] = -0.2
    #action[10] = -0.2

    print('Before loop')
    for i in range(1000):
        obs, r, done, _ = yumiGym.step(action)
        yumiGym.render()
        if done:
            yumiGym.reset()


if __name__ == "__main__":
    main() 