#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:26:41 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
pis = plt.imshow
pf = plt.figure
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage

if __name__ == "__main__":
    imleft = plt.imread('./left0000.jpg')[:,:,0].astype('f').copy()
    imright = plt.imread('./right0000.jpg')[:,:,0].astype('f').copy()
    h,w = imleft.shape[:2]

    #%% get good pixels
    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([imleft, imright]):
        d = scipy.ndimage.filters.gaussian_laplace(im,1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,85)
        dI.append( d.copy() )
        v, u = np.where(valid_mask)
        px.append(u.copy())
        py.append(v.copy())
        pvm.append(valid_mask.copy())

    pis(valid_mask)
    cmin = np.minimum(dI[0].min(), dI[1].min())
    dI[1] += -cmin
    dI[0] += -cmin
    scale = np.maximum(dI[0].max(), dI[1].max())
    dI[1] = dI[1]/scale*256
    dI[0] = dI[0]/scale*256

    for vm,d in zip(pvm, dI):
        dt = np.round(d).astype('int')
        pcolor.append(dt[vm])


#%% construct database
    data = [[[] for _ in range(257)] for _ in range(h)]
    for x,y,c in zip(px[1], py[1], pcolor[1]):
        """put pixels into 256 bins base on their color"""
        data[y][c].append(x)

 #%%

    f,a0 = plt.subplots(1,1,num='query')
    atRow = lambda y: data[y]

    x,y,c = px[0][100], py[0][100], pcolor[0][100]
#    for x,y,c in zip(px[0], py[0], pcolor[0]):
    a0.clear()
    '''show the data slice'''
    a0.plot(dI[0][y,:],'r')
    a0.plot(dI[1][y,:],'b')
    ''' plot target point'''
    a0.vlines(x,0,256,'r')

    p_success = []
    for offset in [0,1,-1,2,-2]:
        plist = atRow(y)[c+offset]
        for cp_x in plist:
            if cp_x < x:   # discard point behind
                continue
            p_success.append(cp_x)
            a0.vlines(cp_x,0,256,'b')

        '''If we don't have a '''
        if not p_success:
            break