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

class RegionProposal(object):
    def __init__(self, input_size=(128, 128, 3), region_size=(64, 64)):
        self.input_size = input_size
        self.region_size = region_size

    def GetMask(self, inputImages):
        imageCompare = inputImages[0]
        averageMask = np.zeros_like(imageCompare)
        for i in range(1, inputImages.shape[0]):
            diffMask = np.float32(imageCompare - inputImages[i])
            averageMask = averageMask + diffMask
        averageMask = np.clip(averageMask, 0, 1)
        averageMask = np.uint8(averageMask * 255)
        averageMask = np.clip(averageMask, 0, 255)
        averageMask = cv2.cvtColor(averageMask, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('mask', averageMask)
        # cv2.waitKey(0)
        return averageMask

    def BlobDetect(self, mask):
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        print("contours", len(contours))
        centres = []
        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            if moments['m00'] != 0:
                centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
                cv2.circle(mask, centres[-1], 1, (0, 0, 0), -1)
        print("centres", len(centres))
        cv2.imshow('contours', mask)
        cv2.waitKey(0)

    def test(self, inputImages):
        self.BlobDetect(self.GetMask(inputImages))


