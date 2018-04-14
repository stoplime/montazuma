# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import math
import random

import gym
from gym import wrappers, logger

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

class Agent(object):
    def __init__(self, action_space):
        self.state_shape = (210, 160, 3)
        self.action_space = action_space
        self.memory = deque(maxlen=2000)

        self.frameSkip = 10
        self.frameCount = 0

        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._make_model()

    def _make_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=self.state_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        # print(self.action_space.n)
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, observation, reward, done):
        # if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_space.n)

        # act_values = self.model.predict(observation)
        # print(np.argmax(act_values[0]))
        # return np.argmax(act_values[0])

    def replay(self, batch_size):
        # print("learning")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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
    agent.load("./save/montazuma-dqn.h5")

    batch_size = 32
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        ob = np.expand_dims(ob, axis=0)
        time = 0
        while True:
            time += 1
            action = agent.act(ob, reward, done)
            new_ob, reward, done, _ = env.step(action)
            new_ob = np.expand_dims(new_ob, axis=0)
            agent.remember(ob, action, reward, new_ob, done)

            ob = new_ob
            env.render()
            if done or time >= 1000:
                print("episode: {}/{}, time: {}, e: {:.3}"
                      .format(i, episode_count, time, agent.epsilon))
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                # agent.save("./save/montazuma-dqn.h5")
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()