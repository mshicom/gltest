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

from pygco import *
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage
from vtk_visualizer import *
from scipy import weave
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    lim = plt.imread('./left0000.jpg')[:,:,0].astype('u1').copy()
    rim = plt.imread('./right0000.jpg')[:,:,0].astype('u1').copy()
    h,w = lim.shape[:2]

    normalize = lambda x:x/np.linalg.norm(x)
    plt.ion()
    # get good pixels
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)

    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([lim, rim]):
#        d = calcGradient(im)
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im.astype('f'),1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,80)
        dI.append( d.copy() )
        u, v = np.meshgrid(range(w),range(h))
        pixel_mask = reduce(np.logical_and,[valid_mask, u>1, v>1, u<w-2, v<h-2])
        px.append(u[pixel_mask].copy())
        py.append(v[pixel_mask].copy())
        pvm.append(valid_mask.copy())

    pis(valid_mask)

    for vm,d in zip(pvm, dI):
        dt = np.round(d).astype('int')
        pcolor.append(dt[vm])

#% construct database
    data = [[] for _ in range(h)]
    for x,y in zip(px[1], py[1]):
        """put pixels into bins base on their color"""
        data[y].append((x, 0, (y,x)))

    data_cur = [[] for _ in range(h)]
    for x,y in zip(px[0], py[0]):
        """put pixels into bins base on their color"""
        data_cur[y].append((x, 0, (y,x)))
    d_result = np.full_like(lim, -1)

#%%
    max_disp = 150
    def stereo_unaries(img1, img2):
        differences = []
        for disp in np.arange(max_disp+1):
            if disp == 0:
                diff = np.abs(img1 - img2)
            else:
                diff = np.abs(img1[:, disp:] - img2[:, :-disp])
            diff = diff[:, max_disp - disp:]
            differences.append(diff)
        return np.dstack(differences).copy("C")
    unaries = stereo_unaries(lim.astype('i4'), rim.astype('i4')).astype(np.int32)
    n_disps = unaries.shape[2]
    newshape = unaries.shape[:2]

    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.int32).copy("C")
    one_d_cut = cut_simple(unaries, 5 * one_d_topology).reshape(newshape)

    plt.imshow(np.argmin(unaries, axis=2), interpolation='nearest')
    d_result[:,max_disp:] = one_d_cut
    v,u = np.where(np.logical_and(d_result >10,d_result <150))
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(lim[v,u],(3,1))]).T)
#%%
    from skimage.measure import label
    lab_im, lab_cnt = label(pvm[0], background=False,return_num=True, connectivity=2)