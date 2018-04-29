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

import models
from ae import AutoEncoder

from region import RegionProposal

import itertools
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

import cv2

class Agent(object):
    def __init__(self, action_space, encoderArchitecture=models.Darknet19Encoder, decoderArchitecture=models.Darknet19Decoder, model_shape=(128, 128, 3), batch_size=32, latent_size=128, latentConstraints="bvae", beta=128, capacity=32):
        self.model_shape = model_shape
        self.batchSize = batch_size
        self.latentSize = latent_size

        self.action_space = action_space
        self.memory = deque(maxlen=2000)

        encoder = encoderArchitecture(self.model_shape, self.batchSize, self.latentSize, latentConstraints=latentConstraints, beta=beta, capacity=capacity)
        decoder = decoderArchitecture(self.model_shape, self.batchSize, self.latentSize)
        self.vae = AutoEncoder(encoder, decoder)
        self.vae.ae.compile(optimizer='adam', loss='mean_absolute_error')

        self.regionLibrary = RegionProposal(input_size=(128, 128, 3), region_size=(32, 32), verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.appendleft((state, action, reward, next_state, done))

    def rememberBatch(self, states, action, reward, next_state, done):
        for i, state in enumerate(states):
            self.memory.appendleft((state, action, reward, next_state, done))

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

    def train(self, batch_images, returnPred=False):
        batch_images = self.processImage(batch_images)

        self.vae.ae.fit(batch_images, batch_images,
                    epochs=1,
                    batch_size=self.batchSize)
        if returnPred:
            return self.vae.ae.predict(batch_images, batch_size=self.batchSize)
        return None
    
    def get_latent(self, state):
        return self.vae.encoder.predict(self.processImage(state))

    def get_predict(self, state, preState=None):
        tileImage = self.processImage(state)
        if preState is not None:
            preImage = self.processImage(preState)
            inputBatch = np.squeeze(np.array([tileImage, preImage]))

            minibatch = itertools.islice(self.memory, self.batchSize)
            batch_images, _, _, _, _ = zip(*minibatch)
            batch_images = np.array(np.squeeze(batch_images, axis=1))
            
            regions = self.regionLibrary.MotionRegions(batch_images)
            regions = np.reshape(regions, (-1,)+regions.shape[-3:])

            while regions.shape[0] % self.batchSize != 0:
                print(regions.shape)
                regions = np.concatenate((regions, regions[0:1]), axis=0)
            print("get_pred.regions.shape", regions.shape)
            
            pred = self.vae.ae.predict(regions, batch_size=self.batchSize)
            return regions, self.unprocessImage(pred)
        else:
            inputImage = np.squeeze(np.array([tileImage]*self.batchSize))

            pred = self.vae.ae.predict(inputImage, batch_size=self.batchSize)
            return state, self.unprocessImage(pred)[0]

    def replay(self, regions=False):
        # print("learning")
        # minibatch = random.sample(self.memory, self.batchSize)
        minibatch = itertools.islice(self.memory, self.batchSize)

        batch_images, _, _, _, _ = zip(*minibatch)
        batch_images = np.array(np.squeeze(batch_images, axis=1))

        cv2.imshow("batch", cv2.cvtColor(np.concatenate((*batch_images[:],), axis=1), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if regions:
            regions = self.regionLibrary.MotionRegions(batch_images)

            # print("batch shape", (-1,)+regions.shape[-3:])
            regions = np.reshape(regions, (-1,)+regions.shape[-3:])
        
            self.train(regions)
        else:
            pred = self.train(batch_images, True)
            return batch_images, self.unprocessImage(pred)
    
    def load(self, name):
        # TODO: Implement this
        self.vae.ae.load_weights(name)
        pass

    def save(self, name):
        # TODO: Implement this
        self.vae.ae.save_weights(name)
        pass

def convertImage(image, size=(128, 128)):
    uint8Image = np.array(image, dtype=np.uint8)
    if uint8Image.ndim > 3:
        imgs = []
        for i in range(uint8Image.shape[0]):
            img = uint8Image[i]
            pilImage = Image.fromarray(img)
            resizedImage = pilImage.resize(size, resample=Image.NEAREST)
            numpyImage = np.array(resizedImage, dtype=np.uint8)
            imgs.append(numpyImage)
        return np.array(imgs, dtype=np.uint8)
    pilImage = Image.fromarray(uint8Image)
    resizedImage = pilImage.resize(size, resample=Image.LANCZOS)
    numpyImage = np.array(resizedImage, dtype=np.uint8)
    return numpyImage

def differenceImageV6(img1, img2):
  a = img1-img2
  b = np.uint8(img1<img2) * 254 + 1
  return a * b

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MontezumaRevenge-v0', help='Select the environment to run')
    args = parser.parse_args()

    folder = os.path.join("save", "images_25_1")
    # model - number - modelType - betaValue - latentSize - capacity - inputShape - episodes - regions/whole
    # BetaVaePooless, DarkNet19, StrideDarkNet19, ResDarkNet19
    loadFile = os.path.join("save", "images_25", "BetaEncoder-25-bvae-0_1-128l-20c-128px-1000e-whole")
    loadFileAgent2 = os.path.join("save", "BetaEncoder-25-bvae-0_1-20l-20c-128px-1000e-whole-Diff")
    saveFile = os.path.join(folder, "BetaEncoder-25-bvae-0_1-128l-20c-128px-1000e-whole")
    saveFileAgent2 = os.path.join(folder, "BetaEncoder-25-bvae-0_1-20l-20c-128px-1000e-whole-Diff")

    load = True
    loadAgent2 = True
    save = True
    saveAgent2 = True

    Agent2 = True
    shrinkAgent2 = False

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
    agent = Agent(env.action_space, models.BetaEncoder, models.BetaDecoder, model_shape=(128, 128, 3), batch_size=32, latent_size=128, latentConstraints="bvae", beta=0.1, capacity=20)
    if load:
        agent.load(loadFile+".h5")
    if Agent2:
        diffAgent = Agent(env.action_space, models.BetaEncoder, models.BetaDecoder, model_shape=(128, 128, 3), batch_size=32, latent_size=20, latentConstraints="bvae", beta=0.1, capacity=20)
        if loadAgent2:
            diffAgent.load(loadFileAgent2+".h5")

    # writer = tf.summary.FileWriter('log')
    # writer.add_graph(tf.get_default_graph())

    episode_count = 1000
    maxTime = 10000
    reward = 0
    done = False
    usingRegions = False

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
                print("episode: {}/{}, time: {}".format(i+1, episode_count, time))
                if Agent2:
                    batch, pred = agent.replay(usingRegions)
                    diff = differenceImageV6(batch, pred)

                    if shrinkAgent2:
                        diff = convertImage(diff, size=(32,32))
                    
                    diffAgent.rememberBatch(diff, action, reward, new_ob, done)
                    agentDiff, diffPred = diffAgent.replay(False)

                    if shrinkAgent2:
                        diff = convertImage(diff, size=(128,128))
                        diffPred = convertImage(diffPred, size=(128,128))

                    batch_vis = np.concatenate((*batch,), axis=1)
                    pred_vis = np.concatenate((*pred,), axis=1)
                    diff_vis = np.concatenate((*diff,), axis=1)
                    diffPred_vis = np.concatenate((*diffPred,), axis=1)

                    visualize = np.concatenate((batch_vis, pred_vis, diff_vis, diffPred_vis), axis=0)
                    cvPred = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
                    cv2.imshow("train", cvPred)
                    cv2.waitKey(1)
                else:
                    agent.replay(usingRegions)
            
            if (done or time == sampleTime) and hasSampled == False:
                # pred = Image.fromarray(agent.get_predict(ob))
                # pred.show()
                if usingRegions:
                    state, pred = agent.get_predict(new_ob, ob)
                    stateVisualize = np.concatenate((*state,), axis=1)
                    visualize = np.concatenate((*pred,), axis=1)
                    visualize = np.concatenate((stateVisualize, visualize), axis=0)
                else:
                    state, pred = agent.get_predict(ob)
                    diff = differenceImageV6(state, pred)
                    state = np.squeeze(state)
                    if Agent2:
                        if shrinkAgent2:
                            diff = convertImage(diff, size=(32,32))
                        _, diffPred = diffAgent.get_predict(diff)
                        if shrinkAgent2:
                            diff = convertImage(diff, size=(128,128))
                            diffPred = convertImage(diffPred, size=(128,128))
                        diff = np.squeeze(diff)
                        visualize = np.concatenate((state, pred, diff, diffPred), axis=1)
                    else:
                        visualize = np.concatenate((state, pred), axis=1)
                
                cvPred = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(folder, "sample_{}.png".format(str(i).zfill(4))), cvPred)
                cv2.imshow("image", cvPred)
                cv2.waitKey(1)
                hasSampled = True

            ob = new_ob
            env.render()
            if done or time >= maxTime:
                print("episode: {}/{}, time: {}"
                      .format(i+1, episode_count, time))
                if len(agent.memory) > agent.batchSize:
                    if Agent2:
                        batch, pred = agent.replay(usingRegions)
                        diff = differenceImageV6(batch, pred)
                        if shrinkAgent2:
                            diff = convertImage(diff, size=(32,32))
                        diffAgent.rememberBatch(diff, action, reward, new_ob, done)
                        diffAgent.replay(False)
                    else:
                        agent.replay(usingRegions)
                if save:
                    agent.save(saveFile+".h5")
                    if Agent2 and saveAgent2:
                        diffAgent.save(saveFileAgent2+".h5")
                break

    # Close the env and write monitor result info to disk
    env.close()