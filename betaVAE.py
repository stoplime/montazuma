# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import os
import math
import random

import gym
from gym import wrappers, logger

import tensorflow as tf

from PIL import Image

sys.path.append("BVAE-tf/bvae")

from models import Darknet19Encoder, Darknet19Decoder, ResDarknet19Encoder, ResDarknet19Decoder
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
        self.model_shape = (128, 128, 3)
        self.batchSize = 32
        self.latentSize = 100

        self.action_space = action_space
        self.memory = deque(maxlen=2000)

        encoder = Darknet19Encoder(self.model_shape, self.batchSize, self.latentSize, 'bvae', )
        decoder = Darknet19Decoder(self.model_shape, self.batchSize, self.latentSize)
        self.vae = AutoEncoder(encoder, decoder)
        self.vae.ae.compile(optimizer='adam', loss='mean_absolute_error')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def processImage(self, state):
        batch = np.float32(state)/255 - 0.5
        return batch
        # return cv2.resize(batch, self.model_shape[:2], interpolation=cv2.INTER_LINEAR)

    def unprocessImage(self, state):
        state[state > 0.5] = 0.5 # clean it up a bit
        state[state < -0.5] = -0.5
        _state = np.uint8((state+0.5)*255)
        return np.squeeze(_state)
        # return cv2.resize(batch, self.state_shape[:2], interpolation=cv2.INTER_LINEAR)

    def act(self, observation, reward, done):
        # if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_space.n)

        # act_values = self.model.predict(observation)
        # # print(np.argmax(act_values[0]))
        # return np.argmax(act_values[0])

    def train(self, batch_images):
        batch_images = self.processImage(batch_images)

        self.vae.ae.fit(batch_images, batch_images,
                    epochs=1,
                    batch_size=self.batchSize)
    
    def get_latent(self, state):
        return self.vae.encoder.predict(self.processImage(state))

    def get_predict(self, state):
        tileImage = self.processImage(state)
        pred = self.vae.ae.predict(np.squeeze(np.array([tileImage]*self.batchSize)), batch_size=self.batchSize)
        return self.unprocessImage(pred)[0]

    def replay(self):
        # print("learning")
        # minibatch = random.sample(self.memory, self.batchSize)
        minibatch = itertools.islice(self.memory, self.batchSize)

        batch_images, _, _, _, _ = zip(*minibatch)
        batch_images = np.array(np.squeeze(batch_images, axis=1))
        
        self.train(batch_images)
    
    def load(self, name):
        # TODO: Implement this
        self.vae.ae.load_weights(name)
        pass

    def save(self, name):
        # TODO: Implement this
        self.vae.ae.save_weights(name)
        pass

def convertImage(image):
    uint8Image = np.array(image, dtype=np.uint8)
    pilImage = Image.fromarray(uint8Image)
    resizedImage = pilImage.resize((128, 128), resample=Image.LANCZOS)
    numpyImage = np.array(resizedImage, dtype=np.uint8)
    return numpyImage

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MontezumaRevenge-v0', help='Select the environment to run')
    args = parser.parse_args()

    folder = os.path.join("save", "images_6")
    # model - number - modelType - betaValue - inputShape
    saveFile = "DarkNet19-6-bvae-100-128px"

    load = False
    save = True

    if not os.path.exists(folder):
        os.makedirs(folder)

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
    if load:
        agent.load(os.path.join(folder, saveFile+".h5"))

    # writer = tf.summary.FileWriter('log')
    # writer.add_graph(tf.get_default_graph())

    episode_count = 1000
    maxTime = 1000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        ob = convertImage(ob)
        ob = np.expand_dims(ob, axis=0)
        time = 0
        sampleTime = random.randint(0, maxTime)
        hasSampled = False
        while True:
            time += 1
            action = agent.act(ob, reward, done)
            new_ob, reward, done, _ = env.step(action)
            new_ob = convertImage(new_ob)
            new_ob = np.expand_dims(new_ob, axis=0)

            agent.remember(ob, action, reward, new_ob, done)

            if len(agent.memory) > agent.batchSize and time % agent.batchSize == 0:
                print("episode: {}/{}, time: {}"
                      .format(i+1, episode_count, time))
                agent.replay()
            
            if (done or time == sampleTime) and hasSampled == False:
                # pred = Image.fromarray(agent.get_predict(ob))
                # pred.show()
                pred = agent.get_predict(ob)
                visualize = np.concatenate((np.squeeze(ob), pred), axis=1)
                cvPred = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(folder, "sample_{}.png".format(str(i).zfill(3))), cvPred)
                cv2.imshow("image", cvPred)
                cv2.waitKey(1)
                hasSampled = True

            ob = new_ob
            env.render()
            if done or time >= maxTime:
                print("episode: {}/{}, time: {}"
                      .format(i+1, episode_count, time))
                if len(agent.memory) > agent.batchSize:
                    agent.replay()
                if save:
                    agent.save(os.path.join(folder, saveFile+".h5"))
                break

    # Close the env and write monitor result info to disk
    env.close()