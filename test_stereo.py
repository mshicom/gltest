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
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)


    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([imleft, imright]):
        d = calcGradient(im)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,80)
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
    data = [[] for _ in range(h)]
    for x,y in zip(px[1], py[1]):
        """put pixels into bins base on their color"""
        data[y].append(x)

    data_l = [[] for _ in range(h)]
    for x,y in zip(px[0], py[0]):
        """put pixels into bins base on their color"""
        data_l[y].append(x)

    x_off, y_off = np.meshgrid(range(-2,3),range(-2,3))
    def calcPatchScore(y, x0, x_can):
        dl =  imleft[y+y_off,    x0+x_off]
        dr = imright[y+y_off, x_can+x_off]
#        ncc = 1-np.dot(normalize(dl.ravel()-dl.mean()),
#                     normalize(dr.ravel()-dr.mean()))
        ssd = np.linalg.norm((dl-dr).ravel())/25
        return ssd

#%% DP debug class
    f = plt.figure(num='query')
    gs = plt.GridSpec(2,2)
    al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
    ab = f.add_subplot(gs[1,:])
    ab.autoscale()
    vec = lambda x:np.reshape(x,(-1,1))

    class Cost:
        def __init__(self, x, v, d_list, pre_idx=None, d_costs=None):
            self.x = x
            self.v = v
            self.d_list = d_list
            self.pre_idx = pre_idx
            self.d_costs = d_costs

    debug = False
#    d_result = np.full_like(imleft, -1)

    for y in range(0,h):
#        al.clear(); ar.clear();ab.clear()
#        al.imshow(imleft); ar.imshow(imright)
        pl,pr = [],[]
        '''obtain and plot the row data'''
        for p in data_l[y]:
#            al.plot(p, y,'r.',ms=3)
            pl.append((p, imleft[y, p], dI[0][y, p]))
        for p in data[y]:
#            ar.plot(p, y,'b.',ms=3)
            pr.append((p, imright[y,p], dI[1][y, p]))
        if pl and pr:

            pl = zip(*pl)
            pr = zip(*pr)

            M = len(pl[0])

            dis = vec(pl[0])-pr[0]        # corresponding dispairity value for array Edata

            vl = np.array(pl[1])
            vr = np.array(pr[1])

            States = []
            for p_idx in xrange(M):
#                ab.clear()
#                ab.plot(pr[0],vr,'b*-')
#                ab.plot(pl[0][:p_idx+1], vl[:p_idx+1],'r*-')

                mask =  dis[p_idx] > 10       # maximum disparity
                d_idx = np.where(mask)[0]
                d_list = np.hstack([0, dis[p_idx, d_idx]])        # 0 for occluded

                x = pl[0][p_idx]
                n = d_list.size

                '''matching cost for each possible dispairty value of x2 '''
#                Edata = np.hstack([10, vl[p_idx] - vr[d_idx]])**2      # 10 for occlusion cost
                cost = [10]+[calcPatchScore(y,x, x-d_can) for d_can in d_list[1:]]
                Edata = np.array(cost)

                if len(States) == 0:
                    cur_state = Cost(x, vl[p_idx], d_list, range(n), Edata)
                    States.append(cur_state)
                else:
                    '''For each value of x2 determine the cost with each value of x1 '''
                    last_state = States[-1]
                    weight = 1/(x - last_state.x)
                    jumps = vec(d_list) - last_state.d_list
                    Ereg = np.where(jumps<2, 1, 5)
                    Ereg[0] = 0        # from non-occluded to occluded
                    Ereg[:,0] = 0      # from occluded to non-occluded
                    Etotal = last_state.d_costs + 100*Ereg   # matching cost + jump cost

                    '''choose the n best path'''
                    best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                    total_cost = Edata + Etotal[range(n), best_idx]

                    cur_state = Cost(x, vl[p_idx], d_list, best_idx, total_cost)
                    States.append(cur_state)

            '''backtrace to readout the optimum'''
            res = np.nanargmin(cur_state.d_costs)                # get the final best
            for j in reversed(xrange(len(States))):
                '''given the state of parent step, lookup the waypoint to it'''
                if res != 0:
#                    ab.plot([States[j].x,  pr[0][res-1]],
#                            [States[j].v,     vr[res-1]],'g-')
                    d_result[y,States[j].x] = States[j].x - pr[0][res-1]
                res = States[j].pre_idx[res]

#            plt.pause(0.01)
#            plt.waitforbuttonpress()
        print y


    v,u = np.where(d_result >10)
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(imleft[v,u],(3,1))]).T)
    exit


#%% DP debug numpy
    f = plt.figure(num='query')
    gs = plt.GridSpec(2,2)
    al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
    ab = f.add_subplot(gs[1,:])
    ab.autoscale()
    vec = lambda x:np.reshape(x,(-1,1))
    for y in range(125,h):
        al.clear(); ar.clear();ab.clear()
        al.imshow(imleft); ar.imshow(imright)
        pl,pr = [],[]
        '''obtain and plot the row data'''
        for p in data_l[y]:
            al.plot(p, y,'r.',ms=3)
            pl.append((p, imleft[y, p], dI[0][y, p]))
        for p in data[y]:
            ar.plot(p, y,'b.',ms=3)
            pr.append((p, imright[y,p], dI[1][y, p]))
        if pl and pr:
            pl.sort(key=lambda x:x[0])
            pr.sort(key=lambda x:x[0])
            pl = zip(*pl)
            pr = zip(*pr)
            '''DP 1st step: get all matching error array and corresponding
               dispairity value, use broacasting to get MxN array,
               rows for sequential target points(in left image),
               colums for candidate matching points (in right image)'''
            M,N = (len(pl[0]), len(pr[1]))
            vl = np.array(pl[0])    # x coordinates
            vr = np.array(pr[0])
            dis = vec(vl)-vr        # corresponding dispairity value for array Edata

#            vl = np.array(pl[2])
#            vr = np.array(pr[2])
#            Edata = 0.5*(vec(vl)-vr)**2
            vl = np.array(pl[1])
            vr = np.array(pr[1])
            Edata = (vec(vl)-vr)**2
            Edata[dis<0] = np.inf   # negative disparity should not be considered

            '''DP 2nd step: calculate regularise term'''
            S = np.empty_like(Edata)
            Best_rec = np.empty_like(Edata,'i8')
            S[0] = Edata[0]
            Best_rec[0] = range(N)
            for i in xrange(1, M):
                ab.clear()
                ab.plot(pr[0],vr,'b*-')
                ab.plot(pl[0][:i+1],vl[:i+1],'r*-')

                ''' non-smooth punishment '''
                Ereg = (vec(dis[i]) - dis[i-1])**2/(pl[0][i]-pl[0][i-1])   # NxN array, costs for dispairity jumps from last point to this point
                Etotal = S[i-1] + 10*Ereg           # matching cost + jump cost
                best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                S[i] = Edata[i] + Etotal[range(N),best_idx]
                Best_rec[i] = best_idx

                '''DP 3rd step: backtrace to readout the optimum'''
                res = np.nanargmin(S[i]) # get the final best
                ab.plot([pl[0][i],  pr[0][res]],
                        [vl[i],vr[res]],'g-')
                for j in xrange(i-1, -1, -1):
                    '''given the state of parent step, lookup the waypoint to it'''
                    res = Best_rec[j+1][res]
                    ab.plot([pl[0][j],  pr[0][res]],
                            [vl[j],vr[res]],'-')

                plt.pause(0.01)
                plt.waitforbuttonpress()

#%% fast DP
    d_result = np.full_like(imleft, -1)
    vec = lambda x:np.reshape(x,(-1,1))
    def dpProcess():
        for y in range(h):
            pl,pr = [],[]
            '''obtain and plot the row data'''
            for p in data_l[y]:
                pl.append((p, imleft[y, p], dI[0][y, p]))

            for p in data[y]:
                pr.append((p, imright[y,p], dI[1][y, p]))

            if pl and pr:
                pl.sort(key=lambda x:x[0])
                pr.sort(key=lambda x:x[0])
                pl = zip(*pl)
                pr = zip(*pr)

                '''DP 1st step: get all matching error array and corresponding
                   dispairity value, use broacasting to get MxN array,
                   rows for sequential target points(in left image),
                   colums for candidate matching points (in right image)'''
                M,N = (len(pl[0]), len(pr[1]))
                vl = np.array(pl[0])    # x coordinates
                vr = np.array(pr[0])
                dis = vec(vl)-vr        # corresponding dispairity value for array Edata

                vl = np.array(pl[2])
                vr = np.array(pr[2])
                Edata = 0.5*(vec(vl)-vr)**2
                vl = np.array(pl[1])
                vr = np.array(pr[1])
                Edata += 0.5*(vec(vl)-vr)**2
                Edata[dis<0] = np.inf   # negative disparity should not be considered

                '''DP 2nd step: calculate regularise term'''
                S = np.empty_like(Edata)
                Best_rec = np.empty_like(Edata,'i8')
                S[0] = Edata[0]
                Best_rec[0] = range(N)
                for i in xrange(1, M):
                    ''' non-smooth punishment '''
                    Ereg = (vec(dis[i]) - dis[i-1])**2/(pl[0][i]-pl[0][i-1])   # NxN array, costs for dispairity jumps from last point to this point
                    Etotal = S[i-1] + 10*Ereg           # matching cost + jump cost
                    best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                    S[i] = Edata[i] + Etotal[range(N),best_idx] #
                    Best_rec[i] = best_idx

                '''DP 3rd step: backtrace to readout the optimum'''
                res = np.nanargmin(S[i])        # result of final step
                d_result[y,pl[0][i]] = pl[0][i]-pr[0][res]
                for j in xrange(i-1, -1, -1):
                    '''given the state of parent step, lookup the waypoint to it'''
                    res = Best_rec[j+1][res]
                    d_result[y,pl[0][j]] = pl[0][j]-pr[0][res]
            print y
    dpProcess()





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

