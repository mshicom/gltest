#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:42:37 2016

@author: kaihong
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
pis = plt.imshow
pf = plt.figure
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage
from vtk_visualizer import *
#%%

def loaddata1():
    data = scipy.io.loadmat('data.mat')
    frames, = data['I']
    G, = data['G']
    K = data['K']
    Z, = data['Z']/100.0
    return frames, G, K, Z

def sample(dIc,x,y):
        return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)

def metric(P): return P[:-1]/P[-1]

def homogeneous(P):
    return np.lib.pad(P, ((0,1),(0,0)), mode='constant', constant_values=1)

normalize = lambda x:x/np.linalg.norm(x)

if __name__ == "__main__":
    if 'frames' not in globals() or 1:
        frames, wGc, K, Zs = loaddata1()
        h,w = frames[0].shape[:2]

    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    refid, curid = 0,8
    Iref, G0, Z = frames[refid].astype('f')/255.0, wGc[refid].astype('f'), Zs[refid].astype('f')
    Icur, G1  = frames[curid].astype('f')/255.0, wGc[curid].astype('f')
    Iref3 = np.tile(Iref.ravel(), (3,1))
    Icur3 = np.tile(Icur.ravel(), (3,1))
#%%
    plt.close('all')
    f,(a) = plt.subplots(1,1,num='epiline')
    a.imshow(sim(Iref, Icur))

    cGr = np.dot(np.linalg.inv(wGc[curid]), wGc[refid])
    Rcr, Tcr = cGr[0:3,0:3], cGr[0:3,3]


    u,v = np.meshgrid(range(w), range(h))
    u,v = (u.ravel()-cx)/fx, (v.ravel()-cy)/fy
    pref = np.vstack([u, v, np.ones(w*h) ]).astype('f')
    pref /= np.linalg.norm(pref, axis=0)*10
    plotxyz(np.vstack([pref,Iref3]).T)

    pcur = np.linalg.inv(cGr).dot(homogeneous(pref))[:3]
    plotxyz(np.vstack([pcur, Icur3]).T, hold=True)



