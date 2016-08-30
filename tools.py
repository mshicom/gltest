#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:28:15 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import scipy.io

pis = plt.imshow
pf = plt.figure
def sim(*arg,**kwarg):
    return np.hstack(arg)

def loaddata1():
    data = scipy.io.loadmat('/home/nubot/data/workspace/gltes/data.mat')
    frames, = data['I']
    G, = data['G']
    K = data['K']
    Z, = data['Z']/100.0
    return frames, G, K, Z

def loaddata2():
    with np.load("kfrecord.npz") as data:
        frames = data["frames"]
        G = data["Gs"]
        K = data['K']
        r = data['r']
    return frames, G, K

def loaddata3():
    with np.load("/mnt/workbench/orb_pos.npz") as data:
        frames = data["frames"]
        wGcs = [np.linalg.inv(G) for G in data["cGws"]]
        K = data['K']
    return frames, wGcs, K

def metric(P): return P[:-1]/P[-1]
def skew(e): return np.array([[  0,  -e[2], e[1]],
                              [ e[2],    0,-e[0]],
                              [-e[1], e[0],   0]])
def isScalar(obj):
    return not hasattr(obj, "__len__")

def sample(dIc,x,y):
    x,y = np.atleast_1d(x, y)
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)

def relPos(wG0, wG1):
    return np.dot(np.linalg.inv(wG0), wG1)

def transform(G,P):
    ''' Pr[3,N]   Pr = rGc*Pc'''
    return G[:3,:3].dot(P)+G[:3,3][:,np.newaxis]

def conditions(*args):
    return reduce(np.logical_and, args)

def normalize(P):
    '''normalize N points seperately, dim(P)=3xN'''
    return P/np.linalg.norm(P, axis=0)

def vec(*arg):
    return np.reshape(arg,(-1,1))

def iD(depth):
    return 1.0/depth

inv = np.linalg.inv

from functools import wraps
from time import time
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print 'func:%r took: %2.6f sec' % (f.__name__, te-ts)
        return result
    return wrap

def projective(x, y):
    x,y = np.atleast_1d(x,y)   # scalar to array
    return np.vstack([x.ravel(), y.ravel(), np.ones_like(x)])

def backproject(x, y, K):
    ''' return 3xN backprojected points array, x,y,z = p[0],p[1],p[2]'''
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    x,y = np.atleast_1d(x,y)   # scalar to array
    x,y = x.ravel(), y.ravel()
    return np.vstack([(x-cx)/fx, (y-cy)/fy, np.ones_like(x)])

def trueProj(xr, yr, cGr,K, Zr):
    # xr, yr, cGr, Zr = 0, 0, getG(f1,f0), f0.Z
    zr = sample(Zr, xr, yr)
    pr = backproject(xr, yr, K)*zr
    pc =  K.dot(transform(cGr, pr))
    pc /= pc[2]
    return pc[0],pc[1]

class IndexTracker(object):
    def __init__(self, X, ax=None):
        if ax is None:  # create one
            fig, ax = plt.subplots(1,1)

        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        try:
            self.slices = X.shape[0]
        except:
            self.slices = len(X)

        self.ind = int(0)

        self.im = ax.imshow(self.X[self.ind])
        self.update()
        fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.slices - 1)

        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
