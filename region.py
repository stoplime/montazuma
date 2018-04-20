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
    def __init__(self, input_size=(128, 128, 3), region_size=(32, 32), verbose=0):
        self.input_size = input_size
        self.region_size = region_size
        self.verbose = verbose

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
        
        return averageMask

    def BlobDetect(self, mask):
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        if self.verbose == 1:
            print("contours", len(contours))
        centres = []
        for i in range(len(contours)):
            moments = cv2.moments(contours[i])
            if moments['m00'] != 0:
                centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
                cv2.circle(mask, centres[-1], 1, (0, 0, 0), -1)
        if self.verbose == 1:
            print("centres", len(centres))
            cv2.imshow('contours', mask)
            cv2.waitKey(1)
        return centres

    def ClampRegion(self, images, centers):
        if self.verbose == 1:
            print("regions Shape", (images.shape[0],) + (len(centers),) + self.region_size + images.shape[-1:])
        
        regions = np.zeros((images.shape[0],) + (len(centers),) + self.region_size + images.shape[-1:], dtype=np.uint8)

        for i, center in enumerate(centers):
            halfRegion = [int(self.region_size[0]/2), int(self.region_size[1]/2)]
            xMin, xMax = center[0] - halfRegion[0], center[0] + halfRegion[0]
            yMin, yMax = center[1] - halfRegion[1], center[1] + halfRegion[1]
            # clamp the min and max
            if xMin < 0:
                xMax -= xMin
                xMin = 0
            if xMax > images.shape[1]:
                xMin -= xMax-images.shape[1]
                xMax = images.shape[1]
            if yMin < 0:
                yMax -= yMin
                yMin = 0
            if yMax > images.shape[2]:
                yMin -= yMax-images.shape[2]
                yMax = images.shape[2]
            # crop the region
            regions[:, i] = images[:, yMin:yMax, xMin:xMax, :]
        return regions

    def MotionRegions(self, inputImages):
        centers = self.BlobDetect(self.GetMask(inputImages.copy()))
        regions = self.ClampRegion(inputImages, centers)
        
        if self.verbose == 1:
            print("regions.shape", regions.shape)
            regionsSpan = np.concatenate((*regions[:],), axis=2)
            print("regionsSpan.shape", regionsSpan.shape)
            regionsSpan = np.concatenate((*regionsSpan[:],), axis=0)
            print("regionsSpan.shape", regionsSpan.shape)
            cv2.imshow("regions", cv2.cvtColor(regionsSpan, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        return regions


