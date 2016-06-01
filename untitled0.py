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
    dI,px,py,pcolor = [],[],[],[]
    for i,im in enumerate([imleft, imright]):
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im,1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,85)
        dI.append( d )

        u, v = np.where(valid_mask)
        color = dI[i][valid_mask]
        px.append(u)
        py.append(v)
        pcolor.append(color)
    plt.hist(dI[0].ravel(),256)
    pf()
    pis(valid_mask)
    scale = np.maximum(dI[0].max(), dI[1].max())*256
    for f in dI:
        f /=  scale
#%%
    data = [[[] for _ in range(256)] for _ in range(h)]
    for j in range(h):
        colordata = data[j]
        for i in range(w):
            """put pixels into 256 bins base on their color"""
            colordata[int(dI[0][j,i])].append(i)
