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
from vtk_visualizer import *

if __name__ == "__main__":
    imleft = plt.imread('./left0000.jpg')[:,:,0].astype('f').copy()
    imright = plt.imread('./right0000.jpg')[:,:,0].astype('f').copy()
    h,w = imleft.shape[:2]

    normalize = lambda x:x/np.linalg.norm(x)
    plt.ion()
    #%% get good pixels
    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([imleft, imright]):
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im,1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,70)
        dI.append( d.copy() )
        u, v = np.meshgrid(range(w),range(h))
        pixel_mask = reduce(np.logical_and,[valid_mask, u>1, v>1, u<w-2, v<h-2])
        px.append(u[pixel_mask].copy())
        py.append(v[pixel_mask].copy())
        pvm.append(valid_mask.copy())

    pis(valid_mask)
    cmin = np.minimum(dI[0].min(), dI[1].min())
    dI[1] += -cmin
    dI[0] += -cmin
    scale = int(np.ceil(np.maximum(dI[0].max(), dI[1].max())))

    for vm,d in zip(pvm, dI):
        dt = np.round(d).astype('int')
        pcolor.append(dt[vm])


#%% construct database
    data = [[[] for _ in range(scale+1)] for _ in range(h)]
    for x,y,c in zip(px[1], py[1], pcolor[1]):
        """put pixels into bins base on their color"""
        data[y][c].append(x)

 #%%
    x_off, y_off = np.meshgrid(range(-2,3),range(-2,3))
    def calcPatchScore(y, x0, x_can):
        dl =  imleft[y+y_off,    x0+x_off]
        dr = imright[y+y_off, x_can+x_off]
#        ncc = 1+np.dot(normalize(dl.ravel()-dl.mean()),
#                     normalize(dr.ravel()-dr.mean()))
        ssd = np.linalg.norm((dl-dr).ravel())/25.0
        return ssd
    from operator import itemgetter

    def test_calcPatchScore():
        x0, y = (786, 25)
        score = [calcPatchScore(y, x0, x_can) for x_can in range(2,w-2)]
        plt.subplot(2,1,1)
        plt.plot(score)
        plt.subplot(2,1,2)
        plt.plot( imleft[y,:],'r')
        plt.plot(imright[y,:],'b')
        plt.vlines(x0,0,255,'r')
        plt.hlines(imleft[y,x0],2,w-2)
        result, element = min(enumerate(score), key=itemgetter(1))
        plt.vlines(result+2,0,255,'g',linewidths=3)

    f,(a0,a1) = plt.subplots(2,1,num='query')
    fi, b0 = plt.subplots(1,1,num='i')
    b0.imshow(imleft)

#    x,y,c = px[0][1000], py[0][1000], pcolor[0][1000]

    start = 0
    debug = False
    d_result = np.full_like(imleft, np.inf)
    for x,y,c in zip(px[0][start:], py[0][start:], pcolor[0][start:]):
        if debug:
            pf(fi.number)
            b0.plot(x,y,'r.')
            pf(f.number)
            a0.clear()
            a1.clear()
            '''show the data slice'''
            a0.plot(dI[0][y,:],'r')
            a0.plot(dI[1][y,:],'b')
            ''' plot target point'''
            a0.vlines(x,0,scale,'r')
            a1.plot( imleft[y,:],'r')
            a1.plot(imright[y,:],'b')
            a1.vlines(x,0,255,'r')
        min_score = np.inf
        '''consider all points in neighbouring gradient-level as candiates'''
        for offset in [0,1,-1,2,-2,3,-3]:
            plist = data[y][np.clip(c+offset, 0, scale)]
            for cp_x in plist:

                ''' discard points in negative or out-of-range disparity'''
                dis = x-cp_x
                if dis < 15 or dis > 120 :   # discard points in negative  or out-of-range disparity
                    continue

                ''' discard points different to much in intensity'''
                if np.abs(imleft[y,x]-imright[y,cp_x]) > 15 :
                    continue

                if debug:
                    a0.vlines(cp_x,0,scale,'b','dashed')
                    a1.vlines(cp_x,0,255,'b','dashed')

                ''' discard points with poor score'''
                score = calcPatchScore(y,x,cp_x)
                if score > 3:
                    continue

                if score < min_score:   # only keep the best match
                    min_score = score
                    d_result[y,x] = x-cp_x

        if debug:
            print 'depth at %d with score:%f' % (cp_x, min_score)
            a0.vlines(x-d_result[y,x],0,scale,'g',linewidths=3)
            a1.vlines(x-d_result[y,x],0,255,'g',linewidths=3)

            plt.pause(0.01)
            plt.waitforbuttonpress()

    v,u = np.where(d_result != 0)
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(imleft[v,u],(3,1))]).T)
