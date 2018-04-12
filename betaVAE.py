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

import tensorflow as tf

from PIL import Image

sys.path.append("BVAE-tf/bvae")

from models import Darknet19Encoder, Darknet19Decoder
from ae import AutoEncoder

import itertools
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

import cv2

class Agent(object):
    def __init__(self, action_space):
        self.state_shape = (210, 160, 3)
        self.model_shape = (256, 256, 3)
        self.batchSize = 10
        self.latentSize = 100

        self.action_space = action_space
        self.memory = deque(maxlen=2000)

        self.frameSkip = 10
        self.frameCount = 0

        encoder = Darknet19Encoder(self.model_shape, self.batchSize, self.latentSize, 'bvae')
        decoder = Darknet19Decoder(self.model_shape, self.batchSize, self.latentSize)
        self.vae = AutoEncoder(encoder, decoder)
        self.vae.ae.compile(optimizer='adam', loss='mean_absolute_error')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def processImage(self, state, batchsize=1):
        batch = np.reshape(np.float32(state), ([batchsize]+list(self.model_shape)))/255 - 0.5
        return batch
        # return cv2.resize(batch, self.model_shape[:2], interpolation=cv2.INTER_LINEAR)

    def unprocessImage(self, state):
        batch = np.reshape(np.uint8((state+0.5)*255), self.model_shape)
        return batch
        # return cv2.resize(batch, self.state_shape[:2], interpolation=cv2.INTER_LINEAR)

    def act(self, observation, reward, done):
        # if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_space.n)

        # act_values = self.model.predict(observation)
        # # print(np.argmax(act_values[0]))
        # return np.argmax(act_values[0])

    def train(self, batch_images):
        batch_images = self.processImage(batch_images, self.batchSize)

        self.vae.ae.fit(batch_images, batch_images,
                    epochs=1,
                    batch_size=self.batchSize)
    
    def get_latent(self, state):
        return self.vae.encoder.predict(self.processImage(state))

    def get_predict(self, state):
        pred = self.vae.ae.predict(self.processImage(state))
        return self.unprocessImage(pred)

    def replay(self):
        # print("learning")
        # minibatch = random.sample(self.memory, self.batchSize)
        minibatch = itertools.islice(self.memory, self.batchSize)

        batch_images, _, _, _, _ = zip(*minibatch)
        batch_images = np.array(np.squeeze(batch_images, axis=1))
        
        self.train(batch_images)
    
    def load(self, name):
        # TODO: Implement this
        # self.model.load_weights(name)
        pass

    def save(self, name):
        # TODO: Implement this
        # self.model.save_weights(name)
        pass

def convertImage(image):
    uint8Image = np.array(image, dtype=np.uint8)
    pilImage = Image.fromarray(uint8Image)
    resizedImage = pilImage.resize((256, 256))
    numpyImage = np.array(resizedImage, dtype=np.float16)
    return numpyImage

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

    # writer = tf.summary.FileWriter('log')
    # writer.add_graph(tf.get_default_graph())

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        ob = convertImage(ob)
        ob = np.expand_dims(ob, axis=0)
        time = 0
        while True:
            time += 1
            action = agent.act(ob, reward, done)
            new_ob, reward, done, _ = env.step(action)
            new_ob = convertImage(new_ob)
            new_ob = np.expand_dims(new_ob, axis=0)

            agent.remember(ob, action, reward, new_ob, done)
            # if len(agent.memory) > agent.batchSize:
            #     agent.replay()

            ob = new_ob
            env.render()
            if done or time >= 1000:
                print("episode: {}/{}, time: {}"
                      .format(i, episode_count, time))
                if len(agent.memory) > agent.batchSize:
                    agent.replay()
                # agent.save("./save/montazuma-dqn.h5")
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()