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
    refid, curid = 8,0
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

    p = (202,299)
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

    def calcAngle(p, M):
        pp = M.dot(p)
        angle = np.rad2deg(np.arctan2(pp[1], pp[0]))
        return angle+360 if angle<0 else angle

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
    data = [[[] for _ in range(grad_scaler.levels+1)] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_ref, ang_ref,ang_ref_z, grad_ref):
        """put pixels into bins base on their color"""
        data[int(np.round(a))][int(np.round(g))].append((p,az))

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

    data_cur = [[[] for _ in range(grad_scaler.levels+1)] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_cur, ang_cur,ang_cur_z, grad_cur):
        """put pixels into bins base on their color"""
        if a > 360 or g > 255:
            continue
        data_cur[int(np.round(a))][int(np.round(g))].append((p,az))

#%%
    def calcEpl(p):
        min_idepth, max_idepth = 0.0, np.inf
        Pc0  = K.dot(Trc)
        Pinf = K.dot(Rrc.dot(np.linalg.inv(K).dot(np.array([p[0],p[1],1]))))

        a0 = (0.01 - Pinf[2])/Pc0[2]      # Pinf[2] + λ*Pc[2] > 0.01
        a1 = (Pinf[0]-640*Pinf[2])/(640*Pc0[2]-Pc0[0])
        a2 = (Pinf[1]-480*Pinf[2])/(480*Pc0[2]-Pc0[1])
        max_idepth = np.min([a0, a1, a2])
        Pc = Pinf + max_idepth*Pc0
        if Pinf[2] < 0 or max_idepth < min_idepth:
            print "Both points are invalid"
        Pc = Pc/Pc[2]
        Pinf = Pinf/Pinf[2]

        l = np.cross(Pinf.flat, Pc.flat)
        l = l/np.linalg.norm(l[:2])
        return Pc, Pinf, l, max_idepth



    f,(al,ar) = plt.subplots(1,2,num='query')
    start = 20000
    import itertools
    for p,an,az,g in zip(puv_cur[start:], ang_cur[start:], ang_cur_z[start:], grad_cur[start:]):
        an = int(np.round(an))
        al.clear(); ar.clear()
        al.imshow(Iref); ar.imshow(Icur)
        ar.plot(p[1],p[0],'r.')

        Pc, Pinf, _, max_idepth = calcEpl([p[1],p[0]])
        al.plot(Pinf[0], Pinf[1],'r*')
        if max_idepth>0:
            al.plot([Pc[0],Pinf[0]], [Pc[1],Pinf[1]],'r')
        for off in [0]:
            a = an + off
            if a<0 or a>ang_scaler.levels:
                continue

            pcan = data[a]
            for ptt in itertools.chain(pcan):
                for pt in ptt:
                    if pt[1] < az:
                        uv = pt[0]
                        al.plot(uv[1],uv[0],'b.',ms=2)

#            for off in [0,-1,1,2,-2]:
#                pcan = data[a][np.clip(int(np.round(g))+off,0,255)]
#                for pt in pcan:
#                    if pt[1] < az:
#                        uv = pt[0]
#                        al.plot(uv[1],uv[0],'b.')
        plt.pause(0.01)
        plt.waitforbuttonpress()





