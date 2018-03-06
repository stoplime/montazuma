# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import math

import gym
from gym import wrappers, logger

class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.shortMemSize = 10
        self.shortMem = np.zeros((self.shortMemSize, 210, 160, 3), dtype=np.float16)
        self.shortMemIndex = 0
        self.shortMemIsFull = False
        self.novel = np.zeros((210, 160, 3), dtype=np.float16)

        self.frameSkip = 10
        self.frameCount = 0

    def act(self, observation, reward, done):
        self.frameCount += 1
        # print(self.noveltyMask(observation))
        if self.frameCount % self.frameSkip == 0:
            self.noveltyMask(observation)
        self.pushMem(observation)
        # print(observation)
        return self.action_space.sample()

    def pushMem (self, state):
        self.shortMem[self.shortMemIndex] = state
        
        self.shortMemIndex += 1
        if self.shortMemIndex >= self.shortMemSize:
            self.shortMemIndex = 0
            self.shortMemIsFull = True

    def noveltyMask (self, state):
        if len(self.shortMem) == 0:
            self.novel = np.zeros_like(state, dtype=np.float16)
            return
        # if not self.shortMemIsFull:
        #     return np.sum( np.sqrt( np.absolute(self.shortMem[:self.shortMemIndex] - state) ), axis=2 ) / self.shortMemIndex
        self.novel = np.sum( np.sqrt( np.absolute(self.shortMem - state) ), axis=2 ) / self.shortMemIndex

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MontezumaRevenge-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    # logger.set_level(logger.INFO)

    print(args.env_id)
    env = gym.make(args.env_id)
    # print(env.unwrapped.get_action_meanings())

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './tmp'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()