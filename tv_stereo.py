#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================
Total variation denoising using Chambolle Pock
==============================================
"""
# Author: Samuel Vaiter <samuel.vaiter@ceremade.dauphine.fr>
from __future__ import division

print __doc__

import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

from tools import *

from pyprox import dual_prox, admm
from pyprox.operators import soft_thresholding
from EpilineCalculator import EpilineCalculator

from vtk_visualizer import plotxyzrgb,plotxyz

import cv2

# Gradient and divergence with periodic boundaries
def gradient(x):    # x(w,h) -> g(w,h,2)
    g = np.zeros((x.shape[0], x.shape[1], 2))
    g[:, :, 0] = np.roll(x, -1, axis=0) - x
    g[:, :, 1] = np.roll(x, -1, axis=1) - x
    return g

def divergence(p):
    px = p[:, :, 0]   # p(w,h,2) -> x(w,h)
    py = p[:, :, 1]
    resx = px - np.roll(px, 1, axis=0)
    resy = py - np.roll(py, 1, axis=1)
    return -(resx + resy)


# Proximity operators G
def huber(x, epsilon):
    err_abs = np.abs(x)
    result = np.where(err_abs>=epsilon, err_abs-0.5*epsilon, err_abs**2/(2*epsilon))
    return result

def gen_warpf(Iref, I, cGr, K):
    shape = Iref.shape
    y, x = np.mgrid[0.:shape[0], 0.:shape[1]]
    ec = EpilineCalculator(x.ravel(), y.ravel(), cGr, K)
    Pinf, Pe = ec.Pinf, ec.Pe

    dx0 = Pe[0]*Pinf[2]-Pe[2]*Pinf[0]
    dy0 = Pe[1]*Pinf[2]-Pe[2]*Pinf[1]
    def warp_d(d):
        """
        linearize the error fuction arround the current depth estimate
        """
        d = d.ravel()
        denom = Pinf[2]+ d*Pe[2]
        p_x,p_y = ec.XYfromD(d)
#        p_x = (Pinf[0] + d*Pe[0])/denom
#        p_y = (Pinf[1] + d*Pe[1])/denom

        # warp images, interplate with calculated pixel coorindates
        Iw = ndimage.map_coordinates(I, (p_y, p_x), order=1, mode='nearest', cval=np.nan) #
        Iw = np.reshape(Iw, shape)
        err = Iw - Iref       # 'time' derivative'

        # calculate the derivative of pixel position(u,v) w.r.t depth(z)
        # calculate the full derivative of error-function(r) w.r.t depth(z)
        gIwy,gIwx = np.gradient(Iw)  # ∇I|π○g(zX) = image gradient after transform
        Ig = gIwx.ravel()*dx0/denom**2 + gIwy.ravel()*dy0/denom**2
        Ig = Ig.reshape(shape)

        return err, Ig, Iw
    return warp_d


def gen_prox_g(warp_d, d, epsilon):
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
amp = lambda u: np.sqrt(np.sum(u ** 2, axis=2))
normalize = lambda u: u / np.tile((np.maximum(amp(u), 1e-10))[:, :, np.newaxis], (1, 1, 2))
def solver(im0, im1, rGc, K, alpha, epsilon, d=None, p=None):
    tau = 0.025
    sigma = 0.5

    F   = lambda u: alpha * np.sum(amp(u))
    prox_f = lambda u, tau: np.tile(soft_thresholding(amp(u), alpha * tau)[:, :, np.newaxis], (1, 1, 2)) * normalize(u)
    prox_fs = dual_prox(prox_f)

    warp_d = gen_warpf(im0, im1, rGc, K)
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
        prox_g = gen_prox_g(warp_d, d, epsilon)

        for iterations in range(200):
            d_old = d.copy()
            p = prox_fs(p + sigma * gradient(d1), sigma)
            d = prox_g(d - tau * divergence(p), tau)

            d = np.maximum(0.01, d)        # make sure depth stays positive
            d1 = 2*d - d_old
        pim.set_data(d)
        plt.pause(0.0001)
#        print F(gradient(d)),G(d)
    return d,p


frames, wGc, K0, Z = loaddata1() #
frames = [np.ascontiguousarray(f, 'f')/255 for f in frames]
rGc = [relPos(wGc[0], g) for g in wGc]

pyr_ims = {0:frames}
Ks  = {0:K0}
scale_mat = np.diag([0.5, 0.5, 1])

for level in range(1,6):
    pyr_ims[level] = [cv2.pyrDown(im) for im in pyr_ims[level-1]]
    Ks[level] = scale_mat.dot(Ks[level-1])

alpha = 3
for level in reversed(range(6)):
    im0,im1 = pyr_ims[level][0],pyr_ims[level][-1]
    K = Ks[level]
    if level==5:
        d = np.full_like(im0, 0.01, 'f')
        p = np.zeros(d.shape+(2,),'f')
    else:
        d_,p_ = d.copy(), p.copy()
        d = cv2.pyrUp(d_)
        p = np.zeros(d.shape+(2,),'f')
        for i in range(2):
            p[:,:,i] = cv2.pyrUp(p_[:,:,i])
    d,p = solver(im0, im1, rGc[-1], K, alpha*0.5**level, 1e-2, d, p)

    y,x = np.mgrid[0.0:im0.shape[0], 0.0:im0.shape[1]]
    p3d = backproject(x.ravel(), y.ravel(), K)/d.ravel()
    p3dc = np.vstack([p3d, np.tile(im0.ravel()*255,(3,1))])
    plotxyzrgb(p3dc.T)
#    plt.waitforbuttonpress()