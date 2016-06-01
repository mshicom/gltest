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
        imheight,imwidth = frames[0].shape[:2]

    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    refid, curid = 0,1
    Iref, G0, Z = frames[refid].astype('f')/255.0, wGc[refid].astype('f'), Zs[refid].astype('f')
    Icur, G1  = frames[curid].astype('f')/255.0, wGc[curid].astype('f')
    cGr = np.dot(np.linalg.inv(G1), G0)
    R, T = cGr[0:3,0:3], cGr[0:3,3]

#%% get good pixels
    dI,px,py,pcolor = [],[],[],[]
    for i,im in enumerate([Iref, Icur]):
        dy, dx = np.gradient(im)
        dI.append( np.sqrt(dx**2+dy**2) )
        valid_mask = dI[i]>0.05
        u, v = np.where(valid_mask)
        color = dI[i][valid_mask]
        px.append(u)
        py.append(v)
        pcolor.append(color)

#%% calc the base line to be projected to
    plt.close('all')
    f,(a) = plt.subplots(1,1,num='epiline')
    a.imshow(sim(Iref, Icur))

    cGr = np.dot(np.linalg.inv(wGc[curid]), wGc[refid])
    Rcr, Tcr = cGr[0:3,0:3], cGr[0:3,3]

    def calcEpl(p):
        min_idepth, max_idepth = 0.0, np.inf
        Pc0  = K.dot(Tcr)
        Pinf = K.dot(Rcr.dot(np.linalg.inv(K).dot(np.array([p[0],p[1],1]))))

        a0 = (0.01 - Pinf[2])/Pc0[2]      # Pinf[2] + λ*Pc[2] > 0.01
        a1 = (Pinf[0]-640*Pinf[2])/(640*Pc0[2]-Pc0[0])
        a2 = (Pinf[1]-480*Pinf[2])/(480*Pc0[2]-Pc0[1])
        max_idepth = a0 # np.min([a0, a1, a2])
        Pc = Pinf + max_idepth*Pc0
        if Pinf[2] < 0 or max_idepth < min_idepth:
            print "Both points are invalid"
        Pc = Pc/Pc[2]
        Pinf = Pinf/Pinf[2]

        l = np.cross(Pinf.flat, Pc.flat)
        l = l/np.linalg.norm(l[:2])
        return Pc, Pinf, l, max_idepth

    Pc, Pinf, l, max_idepth = calcEpl([cx,cy])


#    dist = np.empty((480,640))
#    x,y = np.meshgrid(range(640),range(480))
#    dist.flat = l.dot(np.vstack((x.ravel(),y.ravel(),np.ones(640*480))))
#    dist_range = np.max(dist)-np.min(dist)

    p = np.round(plt.ginput(1, timeout=-1)[0])
    Pc1, Pinf1, ld, _ = calcEpl(p)
    a.plot([Pc[0]+640,Pinf[0]+640],
           [Pc[1], Pinf[1]],'b-')  # the complete epi line
    a.plot(p[0], p[1],'*')
    dist_p = np.hstack([p,1]).dot(l)

    ld0 = np.array([Tcr[2]*(p[0]-cx) -fx*Tcr[0],
                    Tcr[2]*(p[1]-cy) -fy*Tcr[1]])
    pi=cv2.linearPolar(Iref,(cx,cy),300,cv2.WARP_FILL_OUTLIERS)