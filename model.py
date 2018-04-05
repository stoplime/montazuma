# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from tf.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tf.keras.activations import relu

LRelu = tf.nn.leaky_relu


class autoencoder(object):
    def __init__(self, shape, name='bvae'):
        self.name = name
        self.shape = shape
        self.build()

    def build(self):
        build_encoder()
        build_decoder()
    
    def build_encoder(self):
        self.input = tf.placeholder(dtype=tf.float32(), shape=self.shape)
        self.m = input


    def build_decoder(self):
        pass
    
    def attach_loss(self):
        pass
    
    def train_on_batch(self, input_image):
        pass
    
    


def test():
    with tf.Session() as sess:
        pass


if __name__ == "__main__":
    test()

