# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import math
import random

import cv2

class RegionProposal(object):
    def __init__(self, input_size=(128, 128, 3), region_size=(64, 64)):
        self.input_size = input_size
        self.region_size = region_size

    def GetMask(self, inputImages):
        imageCompare = inputImages[0]
        averageMask = np.zeros_like(imageCompare)
        for i in range(1, inputImages.Shape[0]):
            diffMask = imageCompare - inputImages[i]
            averageMask = (averageMask * (i-1) + diffMask)/i
        return averageMask

    def BlobDetect(self, mask):
        


