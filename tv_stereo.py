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

# Gradient and divergence with periodic boundaries
def gradient(x):    # x(w,h) -> g(w,h,2)
    g = np.zeros( (2,) + x.shape )
    g[0] = np.roll(x, -1, axis=0) - x
    g[1] = np.roll(x, -1, axis=1) - x
    return g

def divergence(p):
    px = p[0]   # p(w,h,2) -> x(w,h)
    py = p[1]
    resx = px - np.roll(px, 1, axis=0)
    resy = py - np.roll(py, 1, axis=1)
    return -(resx + resy)

def MakeLinearOperator(shape, weight=None):
    r"""
    Sparse matrix approximation of gradient operator on image plane.
    Use forward differences inside image, backward differences at left/bottom border

    :param shape: image size (tuple of ints)

    Returns:

    :Kx,Ky: sparse matrices for gradient in x- and y-direction
    """
    M = shape[0]
    N = shape[1]

    x,y = np.meshgrid(np.arange(0,N), np.arange(0,M))
    linIdx = np.ravel_multi_index((y,x), x.shape)    # linIdx[y,x] = linear index of (x,y) in an array of size MxN


    i = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,:-1], (-1,1) )) )  # row indices
    j = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,1:], (-1,1) )) )   # column indices
    v = np.vstack( (np.ones( (M*(N-1),1) )*-1, np.ones( (M*(N-1),1) )) )                       # values
    i = np.vstack( (i, np.reshape(linIdx[:,-1], (-1,1) ), np.reshape(linIdx[:,-1], (-1,1) )) )  # row indices
    j = np.vstack( (j, np.reshape(linIdx[:,-1], (-1,1) ), np.reshape(linIdx[:,-2], (-1,1) )) )   # column indices
    v = np.vstack( (v, np.ones( ((M),1) )*1, np.ones( ((M),1) )*-1) )                       # values
    Kx = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    i = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[:-1,:], (-1,1) )) )
    j = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[1:,:], (-1,1) )) )
    v = np.vstack( (np.ones( ((M-1)*N,1) )*-1, np.ones( ((M-1)*N,1) )) )
    i = np.vstack( (i, np.reshape(linIdx[-1,:], (-1,1) ), np.reshape(linIdx[-1,:], (-1,1) )) )
    j = np.vstack( (j, np.reshape(linIdx[-1,:], (-1,1) ), np.reshape(linIdx[-2,:], (-1,1) )) )
    v = np.vstack( (v, np.ones( ((N),1) )*1, np.ones( ((N),1) )*-1) )
    Ky = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    if weight is not None:
        g = scipy.sparse.diags(weight.flatten(), 0)
        L = scipy.sparse.vstack([g*Kx, g*Ky])          # L.shape=(2MN, MN)
    else:
        L = scipy.sparse.vstack([Kx, Ky])

    tau = 1.0/np.array(np.abs(L).sum(axis=0)).ravel()
    sigma = 1.0/np.array(np.abs(L).sum(axis=1)).ravel()
    tau[np.isnan(tau)] = 0
    sigma[np.isnan(sigma)] = 0

    return L, sigma, tau

# Proximity operators G
def huber(x, epsilon):
    err_abs = np.abs(x)
    result = np.where(err_abs>=epsilon, err_abs-0.5*epsilon, err_abs**2/(2*epsilon))
    return result

def gen_warpf(Iref, I, cGr, K):
    shape = Iref.shape
    y, x = np.mgrid[0.:shape[0], 0.:shape[1]]
    ec = EpilineCalculator(x.ravel(), y.ravel(), cGr, K)
    vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(shape)
    Pinf, Pe = ec.Pinf, ec.Pe
    mask_out = np.reshape(~valid_mask, shape)

    dx0 = Pe[0]*Pinf[2]-Pe[2]*Pinf[0]
    dy0 = Pe[1]*Pinf[2]-Pe[2]*Pinf[1]
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
        Iw = np.reshape(Iw, shape)
        err = Iw - Iref       # 'time' derivative'

        # calculate the derivative of pixel position(u,v) w.r.t depth(z)
        gIwy,gIwx = np.gradient(Iw)  # ∇I|π○g(zX) = image gradient after transform
        Ig = gIwx.ravel()*dx0/denom**2 + gIwy.ravel()*dy0/denom**2
        Ig = Ig.reshape(shape)

        err[mask_out] = 0
        Ig[mask_out] = 0
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

# Minimization of F(K*x) + G(x)

# Proximity operators F
amp = lambda u: np.sqrt(np.sum(u ** 2, axis=0))
normalize = lambda u: u / np.tile((np.maximum(amp(u), 1e-10))[np.newaxis, :, :], (2, 1, 1))
def soft_thresholding(x, gamma):
    return np.maximum(0, 1 - gamma / np.maximum(np.abs(x), 1E-10)) * x

def dual_prox(prox):
    return lambda u, sigma: u - sigma * prox(u / sigma, 1 / sigma)

def solver(im0, im1, rGc, K, alpha, epsilon, d=None, p=None):
    tau = 0.025
    sigma = 0.5

    F   = lambda u: alpha * np.sum(amp(u))
    prox_f = lambda u, tau: np.tile(soft_thresholding(amp(u), alpha * tau)[np.newaxis, :, :], (2, 1, 1)) * normalize(u)
    prox_fs = dual_prox(prox_f)

    warp_d = gen_warpf(im0, im1, rGc, K)
    L,_,_ = MakeLinearOperator(im0.shape)
    def G(d):
        err, Ig, Iw = warp_d(d)
        return np.sum(huber(err.ravel(), epsilon))

    if d is None:
        d = np.full_like(im0, 0.01, 'f')
    if p is None:
        p = gradient(d)

    d1 = d.copy()
    f = pf(num='tv')

    pim = pis(im0,cmap='jet')
    for warps in range(20):
        prox_g = gen_prox_g(warp_d, d, epsilon, tau)

        for iterations in range(200):
            d_old = d.copy()

            p.flat += sigma * L.dot(d1.ravel())
            p = prox_fs(p, sigma)

            d.flat -= tau * L.T.dot(p.ravel())
            d = prox_g(d, tau)

            d = np.clip(d, iD(5.0), iD(0.1))        # make sure depth stays positive
            d1 = 2*d - d_old
        pim.set_data(d)
        plt.pause(0.0001)
#        print F(gradient(d)),G(d)
    return d,p


frames, wGc, K0, Z = loaddata1() #
frames = [np.ascontiguousarray(f, 'f')/255 for f in frames]
rGc = [relPos(wGc[0], g) for g in wGc]
EpilineDrawer(frames[0:], wGc[0:], K0)
pyr_ims = {0:frames}
Ks  = {0:K0}
scale_mat = np.diag([0.5, 0.5, 1])

for level in range(1,6):
    pyr_ims[level] = [cv2.pyrDown(im) for im in pyr_ims[level-1]]
    Ks[level] = scale_mat.dot(Ks[level-1])

alpha = 3
cur_id = -1
for level in reversed(range(6)):
    im0,im1 = pyr_ims[level][0],pyr_ims[level][cur_id]
    K = Ks[level]
    if level==5:
        d = np.full_like(im0, 0.01, 'f')
        p = np.zeros((2,)+d.shape,'f')
    else:
        d_,p_ = d.copy(), p.copy()
        d = cv2.pyrUp(d_)
        p = np.zeros((2,)+d.shape,'f')
        for i in range(2):
            p[i] = cv2.pyrUp(p_[i])
    d,p = solver(im0, im1, rGc[cur_id], K, alpha*0.5**level, 1e-2, d, p)

    y,x = np.mgrid[0.0:im0.shape[0], 0.0:im0.shape[1]]
    p3d = backproject(x.ravel(), y.ravel(), K)/d.ravel()
#    plotxyz(p3d.T)
    p3dc = np.vstack([p3d, np.tile(im0.ravel()*255,(3,1))])
    plotxyzrgb(p3dc.T)
