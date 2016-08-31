#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 22:55:41 2016

@author: nubot
"""
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/kaihong/workspace/gltes")
sys.path.append("/home/nubot/data/workspace/gltes")
from tools import *
from EpilineCalculator import EpilineDrawer,EpilineCalculator

import cv2
def sample(dIc,x,y):
    x,y = np.atleast_1d(x, y)
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)
x,y = 170,267

frames, wGc, K, Zs = loaddata1()
frames = [f.astype('f') for f in frames]
rGc = [relPos(wGc[0], g) for g in wGc]

pis(frames[0])
x,y = plt.ginput(0,-1)[0]
ims = {0:frames}
Ks  = {0:K}
ps  = {0: projective(x,y)}
scale_mat = np.diag([0.5, 0.5, 1])
for level in range(1,5):
    ims[level] = [cv2.pyrDown(im) for im in ims[level-1]]
    Ks[level] = scale_mat.dot(Ks[level-1])
    ps[level] = scale_mat.dot(ps[level-1])
    EpilineDrawer(ims[level], wGc, Ks[level], ps[level][:2].ravel())


dmin,dmax = 0.2, 10
level = 4
cnt = 0
for level in reversed(range(5)):
    ec = EpilineCalculator(ps[level][0], ps[level][1], rGc[-1], Ks[level])
    res, dom = ec.searchEPL(ims[level][0], ims[level][-1], dmin=dmin, dmax=dmax)
    cnt += len(res)
    if len(res) == 0:
        break
    ''' find the basin '''
    peak_pos, = scipy.signal.argrelmax(np.insert(res,0,res[1]),mode='warp') # peak
    peak_pos -= 1
    best_pos = np.argmin(res)
    best_seq = np.searchsorted(peak_pos, best_pos)
    peak_pos = np.insert(peak_pos, best_seq, best_pos)
    left  = peak_pos[np.maximum(0, best_seq-1)]
    right = peak_pos[np.minimum(len(peak_pos)-1, best_seq+1)]
    dmin, d, dmax = dom[left], dom[best_pos], dom[right]
    print level,dmin,d,dmax, len(res)
print 1.0/d, sample(Zs[0],x,y)
print 'visted point in total',cnt