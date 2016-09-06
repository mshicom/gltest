#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy import ndimage
import scipy.sparse
import matplotlib.pyplot as plt

from tools import *

from EpilineCalculator import EpilineCalculator,EpilineDrawer

from vtk_visualizer import plotxyzrgb,plotxyz

import cv2
import scipy.sparse

def getIncidenceMat(px, py, shape, makelist=False):
    mask_im = np.full(shape, False,'bool')
    mask_im[py, px] = True

    node_cnt = len(px)
    id_LUT = np.empty_like(mask_im, 'i4')
    id_LUT[mask_im] = range(node_cnt)      # lookup-table of index number for valid pixels

    edges = []
    nbr_list = [[] for _ in xrange(node_cnt)]
    nbr_cnt = np.zeros(node_cnt,'i')
    for p_id in range(node_cnt):
        p_x, p_y = px[p_id], py[p_id]
        degree = 0
        nbr = nbr_list[p_id]
        '''diagonal edge'''
        if 0:
            if mask_im[p_y-1, p_x+1]:
                edges.append([p_id, id_LUT[p_y-1, p_x+1]]); degree += 1
                nbr.append(id_LUT[p_y-1, p_x+1])
            if mask_im[p_y-1, p_x-1]:
                edges.append([p_id, id_LUT[p_y-1, p_x-1]]); degree += 1
                nbr.append(id_LUT[p_y-1, p_x-1])

        if mask_im[p_y-1, p_x]:
            edges.append([p_id, id_LUT[p_y-1,  p_x ]]); degree += 1
            nbr.append(id_LUT[p_y-1, p_x])
        if mask_im[p_y, p_x-1]:
            edges.append([p_id, id_LUT[ p_y,  p_x-1]]); degree += 1
            nbr.append(id_LUT[p_y, p_x-1])
        nbr_cnt[p_id] = degree

    edge_cnt = len(edges)
    row_ind = np.tile(np.arange(edge_cnt)[:,np.newaxis],2).ravel()  # [0,0,1,1,...] edge idx, each node-pair share the same row
    col_ind = np.array(edges).ravel()                               # [i1,o1,i2,o2,...] node linear idx, i for in, o for out
    data = np.tile(np.array([-1,1]), edge_cnt)                      # in -1 out 1

    incidence_matrix = scipy.sparse.csr_matrix((data,(row_ind,col_ind)), (len(edges),node_cnt),'i4') # each row represent an edge
    if makelist:
        enode_out = np.array(zip(*edges)[0])
        return incidence_matrix, nbr_list, nbr_cnt, enode_out
    else:
        return incidence_matrix


def gen_warpf(x,y, Iref, I, cGr, K):
    shape = Iref.shape

    ec = EpilineCalculator(x, y, cGr, K)
    vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(shape)
    Pinf, Pe = ec.Pinf, ec.Pe
    mask_out = ~valid_mask

    dx0 = Pe[0]*Pinf[2]-Pe[2]*Pinf[0]
    dy0 = Pe[1]*Pinf[2]-Pe[2]*Pinf[1]
    v_ref = sample(Iref, x, y)
    def warp_d(d):
        """
        linearize the error fuction arround the current depth estimate
        """
        d = d.ravel()
        denom = Pinf[2]+ d*Pe[2]
        p_x = (Pinf[0] + d*Pe[0])/denom
        p_y = (Pinf[1] + d*Pe[1])/denom

        # warp images, interplate with calculated pixel coorindates
        Iw = ndimage.map_coordinates(I, (p_y, p_x), order=1, mode='nearest', cval=np.nan) #
        err = Iw - v_ref       # 'time' derivative'

        # calculate the derivative of pixel position(u,v) w.r.t depth(z)
        Iy1 = ndimage.map_coordinates(I, (p_y+1, p_x), order=1, mode='nearest', cval=np.nan) #
        Iy0 = ndimage.map_coordinates(I, (p_y-1, p_x), order=1, mode='nearest', cval=np.nan) #
        Ix1 = ndimage.map_coordinates(I, (p_y, p_x+1), order=1, mode='nearest', cval=np.nan) #
        Ix0 = ndimage.map_coordinates(I, (p_y, p_x-1), order=1, mode='nearest', cval=np.nan) #

        gIwy,gIwx = 0.5*(Iy1-Iy0), 0.5*(Ix1-Ix0)
        Ig = (gIwx.ravel()*dx0 + gIwy.ravel()*dy0)/denom**2

#        err[mask_out] = 0
#        Ig[mask_out] = 0
        return err, Ig, Iw
    return warp_d


def gen_prox_g(warp_d, d, epsilon, tau):
    d0 = d.copy()
    err, Ig, Iw = warp_d(d0)
    b = err - d0*Ig
    th = tau*Ig**2 + epsilon

    def prox_g(d, tau):
        r = b + Ig*d                # := err + Ig*(d-d0)
        idx1 = np.where(r > th)
        idx2 = np.where(r < -th)
        idx3 = np.where(np.abs(r) <= th)
        d[idx1] -= tau*Ig[idx1]
        d[idx2] += tau*Ig[idx2]
        d[idx3] = (d[idx3] - tau/epsilon*Ig[idx3]*b[idx3] ) / (1+tau/epsilon*Ig[idx3]**2)
        return d
    return prox_g


frames, wGc, K0, Z = loaddata1() #
frames = [np.ascontiguousarray(f, 'f')/255 for f in frames]
rGcs = [relPos(wGc[0], g) for g in wGc]
#EpilineDrawer(frames[0:], wGc[0:], K0)
dx,dy = np.gradient(frames[0])
grad = np.sqrt(dx**2 + dy**2)
y0,x0 = np.where(grad>np.percentile(grad, 80))
#y0,x0 = np.mgrid[0:480, 0:640]
#y0,x0 = y0.ravel(), x0.ravel()

I, nbr_list, nbrs_cnt, enode_out = getIncidenceMat(x0,y0, frames[0].shape, True)
L = I.transpose().dot(I)
D = L.diagonal()
A = scipy.sparse.diags(D) - L
'''setup neighbor LUT'''
import pandas as pd
nbrs = pd.DataFrame(nbr_list, dtype=int).fillna(-1).values
node_edge = np.abs(I).T

pyr_ims = {0:frames}
Ks  = {0:K0}
ps = {0:(x0,y0)}
scale_mat = np.diag([0.5, 0.5, 1])

for level in range(1,6):
    pyr_ims[level] = [cv2.pyrDown(im) for im in pyr_ims[level-1]]
    Ks[level] = scale_mat.dot(Ks[level-1])
    ps[level] = (0.5*ps[level-1][0], 0.5*ps[level-1][1])

#%%

tau = 0.025
sigma = 0.5
epsilon = 1e-2

cur_id = -1
level = 5
alpha = 3*0.5**level
im0,im1 = pyr_ims[level][0],pyr_ims[level][cur_id]
K = Ks[level]
x,y = ps[level]
rGc = rGcs[cur_id]
def dual_prox(prox):
    return lambda u, sigma: u - sigma * prox(u / sigma, 1 / sigma)

soft_thresholding = lambda x, gamma: np.maximum(0, 1 - gamma / np.maximum(np.abs(x), 1E-10)) * x

amp = lambda eFlow: np.sqrt(node_edge.dot(eFlow**2))     # 1xN, norm(▽x) of each nodes
normalize = lambda eFlow: eFlow / np.maximum(amp(eFlow)[enode_out], 1e-10) #1xM, scale each edge
F   = lambda eFlow: alpha * np.sum(amp(eFlow))

prox_f = lambda eFlow, tau: soft_thresholding(amp(eFlow), alpha * tau)[enode_out] * normalize(eFlow)
prox_fs = dual_prox(prox_f)
warp_d = gen_warpf(x,y, im0, im1, rGc, K)
# Proximity operators G
def huber(x, epsilon):
    err_abs = np.abs(x)
    result = np.where(err_abs>=epsilon, err_abs-0.5*epsilon, err_abs**2/(2*epsilon))
    return result
def G(d):
    err, Ig, Iw = warp_d(d)
    return np.sum(huber(err.ravel(), epsilon))

d = np.full_like(x, 0.3, 'f')  # 1xN, nodes
p = I.dot(d)        # 1xM flows on edges = ▽x
d1 = d.copy()
for warps in range(200):
    prox_g = gen_prox_g(warp_d, d, epsilon, tau)

    for iterations in range(20):
        d_old = d.copy()

        p += sigma * I.dot(d1)
        p = prox_fs(p, sigma)
#        norm_n = np.sqrt(node_edge.dot(p**2))
#        p /= np.maximum(norm_n[enode_out], 1) # reprojection

        d -= tau * I.T.dot(p)
        d = prox_g(d, tau)

        d = np.clip(d, iD(5.0), iD(0.1))        # make sure depth stays positive
        d1 = 2*d - d_old
#    pim.set_data(d)
#
#        print F(I.dot(d)),G(d)
    p3d = backproject(x.ravel(), y.ravel(), K)/d.ravel()
#    p3dc = np.vstack([p3d, np.tile(frames[0].ravel()*255,(3,1))])

    plotxyz(p3d.T)
    plt.pause(0.0001)