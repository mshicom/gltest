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
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im,2)
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

    data_l = [[[] for _ in range(scale+1)] for _ in range(h)]
    for x,y,c in zip(px[0], py[0], pcolor[0]):
        """put pixels into bins base on their color"""
        data_l[y][c].append(x)
#%% DP
    import itertools
    f = plt.figure(num='query')
    gs = plt.GridSpec(2,2)
    al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
    ab = f.add_subplot(gs[1,:])
    ab.autoscale()
    vec = lambda x:np.reshape(x,(-1,1))
    for y in range(7,h):
        al.clear(); ar.clear();ab.clear()
        al.imshow(imleft); ar.imshow(imright)
        pl,pr = [],[]
        '''obtain and plot the row data'''
        for ptt in itertools.chain(data_l[y]):
            for p in ptt:
                al.plot(p, y,'r.',ms=3)
                pl.append((p, imleft[y, p]))
        for ptt in itertools.chain(data[y]):
            for p in ptt:
                ar.plot(p, y,'b.',ms=3)
                pr.append((p, imright[y,p]))
        if pl and pr:
            pl.sort(key=lambda x:x[0])
            pr.sort(key=lambda x:x[0])
            pl = zip(*pl)
            pr = zip(*pr)


            '''DP 1st step: get all matching error array and corresponding dispairity value'''
            vl = np.array([imleft[y,x] for x in pl[0]])
            vr = np.array([imright[y,x] for x in pr[0]])
            ''' use broacasting to get MxN array,
            rows for sequential target points(in left image),
            colums for candidate matching points (in right image)'''
            Edata = np.abs(vec(vl)-vr)
            M,N = (len(pl[0]), len(pr[1]))

            vl = np.array(pl[0])    # x coordinates
            vr = np.array(pr[0])
            dis = vec(vl)-vr        # corresponding dispairity value for array Edata
            Edata[dis<0] = np.inf   # negative disparity should not be considered

            '''DP 2nd step: calculate regularise term'''
            S = np.empty_like(Edata)
            Best_rec = np.empty_like(Edata,'i8')
            S[0] = Edata[0]
            Best_rec[0] = range(N)
            for i in xrange(1, M):
                ab.clear()
                ab.plot(pr[0],pr[1],'b*-')
                ab.plot(pl[0][:i+1],pl[1][:i+1],'r*-')

                ''' non-smooth punishment '''
                Ereg = (vec(dis[i]) - dis[i-1])**2/(vl[i]-vl[i-1])   # NxN array, costs for dispairity jumps from last point to this point
                Etotal = vec(S[i-1]) + 10*Ereg           # matching cost + jump cost
                best_idx = np.nanargmin(Etotal, axis=0)   # Nx1 array, shortest path to current N choose
                S[i] = Edata[i] + Etotal[range(N),best_idx] #
                Best_rec[i] = best_idx

                '''DP 3rd step: backtrace to readout the optimum'''
                res = np.nanargmin(S[i]) # get the final best
                ab.plot([pl[0][i],pr[0][i]], [pl[1][res],pr[1][res]],'g-')
                for j in xrange(i-1, 0, -1):
                    '''given the state of parent step, lookup the waypoint to it'''
                    res = Best_rec[j+1][res]
                    ab.plot([pl[0][i],pr[0][i]], [pl[1][res],pr[1][res]],'g-')

                plt.pause(0.01)
                plt.waitforbuttonpress()




 #%% one-by-one matching
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
        x0, y = (786, 205)
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



#    x,y,c = px[0][1000], py[0][1000], pcolor[0][1000]

    start = 0
    debug = False
    d_result = np.full_like(imleft, 0)
    if debug:
        f,(a0,a1) = plt.subplots(2,1,num='query')
        fi, b0 = plt.subplots(1,1,num='i')
        b0.imshow(imleft)

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
        for offset in [0,1,-1,2,-2]:
            plist = data[y][np.clip(c+offset, 0, scale)]
            for cp_x in plist:

                ''' discard points in negative or out-of-range disparity'''
                dis = x-cp_x
                if dis < 15 or dis > 120 :
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
                ''' only keep the best match'''
                if score < min_score:
                    min_score = score
                    d_result[y,x] = x-cp_x

        if debug:
            if min_score != np.inf:
                print 'depth at %d with score:%f' % (cp_x, min_score)
                a0.vlines(x-d_result[y,x],0,scale,'g',linewidths=3)
                a1.vlines(x-d_result[y,x],0,255,'g',linewidths=3)

            plt.pause(0.01)
            plt.waitforbuttonpress()
    pis(d_result)

    v,u = np.where(d_result != 0)
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(imleft[v,u],(3,1))]).T)
#%%
    y = 205; x = 305
    f = imleft[205,:]
    g = imright[205,:]
    df = scipy.ndimage.filters.gaussian_gradient_magnitude(f,2)
    dg = scipy.ndimage.filters.gaussian_gradient_magnitude(g,2)
    plt.subplot(2,1,1)
    plt.plot(f,'r')
    plt.plot(g,'b')
    plt.subplot(2,1,2)
    plt.plot(df,'r')
    plt.plot(dg,'b')

    pf()
    plt.subplot(2,1,1)
    plt.plot(f)

    zc = np.where(np.diff(np.signbit(df)))
    for i in zc:
        plt.vlines(i+1,0,255)
    plt.subplot(2,1,2)
    plt.plot(df)