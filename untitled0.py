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

    normalize = lambda x:x/np.linalg.norm(x)
    plt.ion()
    #%% get good pixels
    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([imleft, imright]):
        d = scipy.ndimage.filters.gaussian_laplace(im,1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,85)
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



    def calcPatchScore(y, x0, x_can):
        x_off, y_off = np.meshgrid(range(-2,3),range(-2,3))
        dl =  imleft[y+y_off,    x0+x_off]
        dr = imright[y+y_off, x_can+x_off]
        ncc = np.dot(normalize(dl.ravel()-dl.mean()),
                     normalize(dr.ravel()-dr.mean()))
        return ncc
    def test_calcPatchScore():
        x0, y = (302,180)
        score = [calcPatchScore(y, x0, x_can) for x_can in range(2,w-2)]
        plt.subplot(2,1,1)
        plt.plot(score)
        plt.subplot(2,1,2)
        plt.plot( imleft[y,2:-2],'r')
        plt.plot(imright[y,2:-2],'b')
        plt.vlines(x0,0,255,'r')

    f,(a0,a1) = plt.subplots(2,1,num='query')
    fi, b0 = plt.subplots(1,1,num='i')
    b0.imshow(imleft)

#    x,y,c = px[0][1000], py[0][1000], pcolor[0][1000]
    start = 1000
    for x,y,c in zip(px[0][start:], py[0][start:], pcolor[0][start:]):
        pf(fi.number)
        b0.plot(x,y,'r.')
        pf(f.number)
        a0.clear()
        a1.clear()
        '''show the data slice'''
        a0.plot(dI[0][y,:],'r')
        a0.plot(dI[1][y,:],'b')
        ''' plot target point'''
        a0.vlines(x,0,256,'r')
        a1.plot( imleft[y,:],'r')
        a1.plot(imright[y,:],'b')
        a1.vlines(x,0,256,'r')
        p_success = []
        '''consider all points in neighbouring gradient-level as candiates'''
        for offset in [0,1,-1,2,-2]:
            plist = data[y][c+offset]
            for cp_x in plist:
                '''do the match'''
                if x-cp_x < 0:   # discard point in negative disparity
                    continue
                p_success.append((calcPatchScore(y,x,cp_x), cp_x))
                a0.vlines(cp_x,0,256,'b')
                a1.vlines(cp_x,0,256,'b')
        if p_success:
            p_d = max(p_success, key=lambda x:x[0])
            print 'depth at %d, c:%f' % (p_d[1],p_d[0])
            a0.vlines(p_d[1],0,256,'g')
            a1.vlines(p_d[1],0,256,'g')
        else:
            print 'failed to find a match'

        plt.pause(0.01)
        plt.waitforbuttonpress()