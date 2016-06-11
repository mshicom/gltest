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
snormalize = lambda x:x/np.linalg.norm(x, axis=0)
vec = lambda x:np.reshape(x,(-1,1))
class Scaler:
    def __init__(self, vmin, vmax, levels):
        self.min, self.max, self.levels = (vmin, vmax, levels)
        self.a = levels/(vmax-vmin)
        self.b = -self.a*vmin

    def __call__(self, value):
        return self.a*value+self.b

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
    cGr = np.dot(np.linalg.inv(wGc[curid]), wGc[refid])
    Rcr, Tcr = cGr[0:3,0:3], cGr[0:3,3]
    rGc = np.dot(np.linalg.inv(wGc[refid]), wGc[curid])
    Rrc, Trc = rGc[0:3,0:3], rGc[0:3,3]

    u,v = np.meshgrid(range(w), range(h))
    u,v = (u.ravel()-cx)/fx, (v.ravel()-cy)/fy
    pref = np.vstack([u, v, np.ones(w*h) ]).astype('f')
    pt = pref*Z.ravel()
    p0 = np.vstack([pt,Iref3])

    pref /= np.linalg.norm(pref, axis=0)*6
    p1 = np.vstack([pref,Iref3])

#    plotxyz(np.vstack([pref,Iref3]).T)

    pcur = rGc.dot(homogeneous(pref))[:3]
    p2 = np.vstack([pcur,Icur3])
#    plotxyz(np.vstack([pcur, Icur3]).T, hold=True)

    vis = get_vtk_control()
    vis.RemoveAllActors()
    vis.AddPointCloudActor(np.hstack([p0,p1,p2]).T)
    vis.AddLine([0,0,0], Trc)

    p = (182,286)#(202,299)
    ps = np.array([(p[0]-cx)/fx,(p[1]-cy)/fy,1])*Z[p[1],p[0]]
    vis.AddLine([0,0,0], ps)
    vis.AddLine(Trc, ps)
#%% calculate the
    '''define vectors correspond to 4 image corners '''
    corners = [[0,0],[0,h],[w,h],[w,0]]
    corners = [normalize(np.array([(cn[0]-cx)/fx,
                                   (cn[1]-cy)/fy,
                                   1])) for cn in corners]

    '''generate new coordinate system'''
    ax_z = normalize(Trc)                          # vector pointed to camera Cur serve as z axis
    ax_y = normalize(np.cross(ax_z, corners[0]))   # top-left corner serves as temperary x axis
    ax_x = normalize(np.cross(ax_y, ax_z))
    M = np.vstack([ax_x,ax_y,ax_z])

    '''transform the vector to new coordinate and then calculate the vector
       angle wrt. to x axis'''
    new_ps = [M.dot(cn) for cn in corners]
    angles = [np.rad2deg(np.arctan2(p[1], p[0])) for p in new_ps]

    '''re-adjust the x,y axis so that all pixel lies on the same half-plane'''
    ax_min = np.argmin(angles)
    ax_y = normalize(np.cross(ax_z, corners[ax_min]))   # top-left corner serves as temperary x axis
    ax_x = normalize(np.cross(ax_y, ax_z))
    M = np.vstack([ax_x,ax_y,ax_z])
    new_ps = [M.dot(cn) for cn in corners]
    angles = [np.rad2deg(np.arctan2(p[1], p[0])) for p in new_ps]
    print angles

    if 0:
        phi,theta = np.meshgrid(range(78), range(10,170))
        phi = np.deg2rad(phi.ravel())
        theta = np.deg2rad(theta.ravel())
        pxyz = np.vstack([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)])
        pxyz = M.T.dot(pxyz)
        vis.AddPointCloudActor(pxyz.T)


#%% generate target point
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)

    grad = calcGradient(Iref)
    grad_threshold = np.percentile(grad,80)

    u, v = np.meshgrid(range(w),range(h))
    ub, vb = (u-cx)/fx, (v-cy)/fy


    mask_ref = reduce(np.logical_and,[grad>grad_threshold, u>1, v>1, u<w-2, v<h-2])
    puv_ref = np.array(np.where(mask_ref)).T

    pts = np.vstack([ub[mask_ref], vb[mask_ref], np.ones(mask_ref.sum())])

    grad_ref = grad[mask_ref]
    grad_scaler = Scaler(grad_ref.min(), grad_ref.max(), 255)
    grad_ref = vec(grad_scaler(grad_ref))

    '''calc angle'''
    pvp = M.dot(pts)
    ang_ref =  np.rad2deg(np.arctan2(pvp[1,:], pvp[0,:]))
    ang_ref[ang_ref<0] += 360
    ang_scaler = Scaler(ang_ref.min(), ang_ref.max(), 360)
    ang_ref = vec(ang_scaler(ang_ref))

    ang_ref_z =  vec(np.rad2deg(np.arctan2(np.linalg.norm(pvp[:2,:],axis=0), pvp[2,:])))
    ang_ref_z[ang_ref_z<0] += 360

    '''fill the data structure'''
    data = [[] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_ref, ang_ref,ang_ref_z, grad_ref):
        """put pixels into bins base on their color"""
        data[int(np.round(a))].append((tuple(p),double(az)))

    if 0:
        rec_im = np.zeros((361,361))
        for p,a,az,g in zip(puv_ref, ang_ref,ang_ref_z, grad_ref):
           rec_im[int(np.round(az)),int(np.round(a))]=1
        pis(rec_im[57:140,:])


#%%
    grad = calcGradient(Icur)
    mask_cur = reduce(np.logical_and,[grad>grad_threshold, u>1, v>1, u<w-2, v<h-2])
    puv_cur = np.array(np.where(mask_cur)).T
    pts_cur = np.vstack([ub[mask_cur], vb[mask_cur], np.ones(mask_cur.sum())])
    pts_cur = M.dot(Rrc.dot(pts_cur))
    ang_cur = np.rad2deg(np.arctan2(pts_cur[1,:], pts_cur[0,:]))
    ang_cur[ang_cur<0] += 360
    ang_cur = vec(ang_scaler(ang_cur))
    ang_cur_z =  vec(np.rad2deg(np.arctan2(np.linalg.norm(pts_cur[:2,:],axis=0), pts_cur[2,:])))
    ang_cur_z[ang_cur_z<0] += 360

    grad_cur = grad[mask_cur]
    grad_cur = vec(grad_scaler(grad_cur))


    data_cur = [[] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_cur, ang_cur,ang_cur_z, grad_cur):
        """put pixels into bins base on their color"""
        if a > 360 or g > 255:
            continue
        data_cur[int(np.round(a))].append((tuple(p),double(az)))
#%% demo: points on the scanline
    if 1:
        def trueProj(x, y, G=cGr):
            p0 = np.array([(x-cx)/fx, (y-cy)/fy, 1.0])*Z[int(y),int(x)]
            p =  K.dot(G[0:3,0:3].dot(p0)+G[0:3,3])
            p /= p[2]
            return p[0],p[1]

        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        ar,ac = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        for a in range(65,ang_scaler.levels+1):
            ac.clear(); ar.clear();ab.clear()
            ar.imshow(Iref); ac.imshow(Icur)
            pr,pc = [],[]

            for pt in data_cur[a]:
                p = pt[0]
                ac.plot(p[1],p[0],'r.')
                pc.append((double(pt[1]), Icur[p], p))

            for pt in data[a]:
                p = pt[0]
                tx,ty = trueProj(p[1],p[0])
                ac.plot(tx,ty,'g.')
                ar.plot(p[1],p[0],'b.')
                pr.append((double(pt[1]), Iref[p], p))

            if pc:
                pc.sort(key=lambda x:x[0])
                pc = zip(*pc)
                ab.plot(pc[0],pc[1],'r*-')
            if pr:
                pr.sort(key=lambda x:x[0])
                pr = zip(*pr)
                ab.plot(pr[0],pr[1],'b*-')

            plt.pause(0.01)
            plt.waitforbuttonpress()
#%% DP stereo
    f = plt.figure(num='query')
    gs = plt.GridSpec(2,2)
    al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
    ab = f.add_subplot(gs[1,:])
    ab.autoscale()
    vec = lambda x:np.reshape(x,(-1,1))

    class Cost:
        def __init__(self, x, v, p, d_list, pre_idx=None, d_costs=None):
            self.x = x
            self.v = v
            self.d_list = d_list
            self.pre_idx = pre_idx
            self.d_costs = d_costs
            self.p = p

    d_result = np.full_like(Icur, -1)
    for y in range(ang_scaler.levels+1):
        al.clear(); ar.clear();ab.clear()
        al.imshow(Icur); ar.imshow(Iref)
        pl,pr = [],[]
        '''obtain and plot the row data'''
        for p,az in data_cur[y]:
            al.plot(p[1], p[0],'r.',ms=3)
            pl.append((az, Icur[p],p))
        for p,az in data[y]:
            ar.plot(p[1], p[0],'b.',ms=3)
            pr.append((az, Iref[p],p))
        if pl and pr:
            pl.sort(key=lambda x:x[0])
            pr.sort(key=lambda x:x[0])
            pl = zip(*pl)
            pr = zip(*pr)

            M = len(pl[0])

            dis = vec(pl[0])-pr[0]        # corresponding dispairity value for array Edata

            vl = np.array(pl[1])
            vr = np.array(pr[1])

            States = []
            for p_idx in xrange(M):
                ab.clear()
                ab.plot(pr[0],vr,'b*-')
                ab.plot(pl[0][:p_idx+1], vl[:p_idx+1],'r*-')

                mask =  dis[p_idx] > 0       # maximum disparity
                d_idx = np.where(mask)[0]
#                d_list = np.hstack((dis[p_idx, d_idx], -1))
                d_list = dis[p_idx, d_idx]
                x = pl[0][p_idx]
                n = d_list.size
                if n<1:
                    continue

                '''matching cost for each possible dispairty value of x2 '''
#                Edata = np.hstack(((vl[p_idx] - vr[d_idx])**2, 0.04))
                Edata = (vl[p_idx] - vr[d_idx])**2
                if len(States) == 0:
                    cur_state = Cost(x, vl[p_idx], pl[2][p_idx], d_list, range(n), Edata)
                    States.append(cur_state)
                else:
                    '''For each value of x2 determine the cost with each value of x1 '''
                    last_state = States[-1]
                    weight = 1/(x - last_state.x)
                    Ereg = (vec(d_list) - last_state.d_list)**2
#                    Ereg[-1] = 0
#                    Ereg[:,-1] = 0
                    Etotal = last_state.d_costs + 0.1*Ereg* weight   # matching cost + jump cost

                    '''choose the n best path'''
                    best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                    total_cost = Edata + Etotal[range(n), best_idx]

                    cur_state = Cost(x, vl[p_idx], pl[2][p_idx], d_list, best_idx, total_cost)
                    States.append(cur_state)

            '''backtrace to readout the optimum'''
            res = np.nanargmin(cur_state.d_costs)                # get the final best
            for j in reversed(xrange(len(States))):
                '''given the state of parent step, lookup the waypoint to it'''
                d_result[pl[2][p_idx]] = res
                ab.plot([States[j].x,  pr[0][res]],
                        [States[j].v,     vr[res]],'g-')
                res = States[j].pre_idx[res]

#                plt.pause(0.01)
#                plt.waitforbuttonpress()
        print y
    pf()
    pis(d_result)

#%% exam the depth calculation

    positive_range = lambda x: x if x>0 else x+2*np.pi
    def calcAngle(x, y, G=None):
        p0 = np.array([(x-cx)/fx, (y-cy)/fy, 1.0])
        if not G is None:
            p0 = G[0:3,0:3].dot(p0)
        p = M.dot(p0)
        theta = positive_range(np.arctan2(p[1], p[0]))
        phi = positive_range(np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2]))
        return theta, phi

    def calcDepth(ar, ac):
        B = np.linalg.norm(Trc)
        c = np.pi-ac
        b = ac-ar
        return B*np.sin(c)/np.sin(b)

    def test_calcDepth():
        f,a = plt.subplots(1, 1, num='test_depth')
        a.imshow(sim(Iref, Icur))
        while 1:
            plt.pause(0.01)
            pref = np.round(plt.ginput(1, timeout=-1)[0])
            a.plot(pref[0], pref[1],'r.',ms=2)
            pcur = trueProj(pref[0], pref[1])
            a.plot(pcur[0]+640, pcur[1],'b.',ms=2)
            a_ref = calcAngle(pref[0], pref[1])
            a_cur = calcAngle(pcur[0], pcur[1], rGc)
            prange = calcDepth(a_ref[1], a_cur[1])
            P = snormalize(np.array([(pref[0]-cx)/fx, (pref[1]-cy)/fy, 1.0]))*prange
            print 'Ground truth:%f, estimated:%f' % (P[2], Z[pref[1],pref[0]])
    test_calcDepth()