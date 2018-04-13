# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import math
import random

from PIL import Image
import cv2

class Visualizer(object):
    def __init__(self, imageShape, numCompare=2):
        self.NumCompare = numCompare
        self.ImageShape = imageShape
        self.Count = [0] * numCompare

        self.Masks = np.zeros((numCompare,) + imageShape, dtype=np.float32)

    def DisplayMask(self):
        # TODO: convert mask from float to uint8 to display 0 to 255
        pass

    def AddImage(self, image, compareNum):
        proccessedImage = self.ProcessImage(image)
        mask = self.FeatureMaskDetect(image)
        self.AppendMask(mask, compareNum)
    
    def ProcessImage(self, image):
        # TODO: add any preprocessing to the image as nessacery
        return image

    def FeatureMaskDetect(self, image):
        mask = np.zeros_like(image)
        # TODO: Add in Color Range filter and append it to the mask
        return mask

    def AppendMask(self, mask, compareNum):
        totalMask = self.Masks[compareNum] * self.Count[compareNum]
        addedTotalMask = totalMask + mask
        averagedMask = addedTotalMask/(self.Count[compareNum]+1)
        self.Masks[compareNum] = averagedMask
        self.Count[compareNum] += 1
