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
import scipy.io
from vtk_visualizer import *

from scipy import weave




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
def skew(e): return np.array([[  0,  -e[2], e[1]],
                              [ e[2],    0,-e[0]],
                              [-e[1], e[0],   0]])
def isScalar(obj):
    return not hasattr(obj, "__len__")

def relPos(wG0, wG1):
    return np.dot(np.linalg.inv(wG0), wG1)


def transform(G,x):
    return G[:3,:3].dot(x)+G[:3,3]

def conditions(*args):
    return reduce(np.logical_and, args)

normalize = lambda x:x/np.linalg.norm(x)
snormalize = lambda x:x/np.linalg.norm(x, axis=0)
vec = lambda x:np.reshape(x,(-1,1))
inv = np.linalg.inv

if __name__ == "__main__":
    if 'frames' not in globals() or 1:
        frames, wGc, K, Zs = loaddata1()
        h,w = frames[0].shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

#%%

    def backproject(x, y, K=K):
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        if isScalar(x):
            return np.array([(x-cx)/fx, (y-cy)/fy, 1])
        else:
            x,y = x.ravel(), y.ravel()
            return np.array([(x-cx)/fx, (y-cy)/fy, np.ones(len(x))])


    def trueProj(xr, yr, cGr, Zr):
        pr = backproject(xr, yr)*Zr[yr,xr]
        if isScalar(xr):
            pc =  K.dot(cGr[0:3,0:3].dot(pr)+cGr[0:3,3])
        else:
            pc =  K.dot(cGr[0:3,0:3].dot(pr)+cGr[0:3,3][:,np.newaxis])
        pc /= pc[2]
        return pc[0],pc[1]


    import scipy.signal
    def scharr(im):
        im = im.astype('f')
        kernel_h = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
        kernel_v = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
        dx = scipy.signal.convolve2d(im, kernel_h, boundary='symm', mode='same')
        dy = scipy.signal.convolve2d(im, kernel_v, boundary='symm', mode='same')
        mod = np.sqrt(dx**2+dy**2)
        orin = np.arctan2(dy,dx)
        return mod, orin

    class Frame(object):
        __slots__ = ['im', 'wGc',               \
                     'px','py','p_cnt','P',     \
                     'nbrs','v','grad','orin',
                     'Z']
        def __init__(self, img, wGc=np.eye(4), Z=None, gthreshold=None):
            self.im = img.copy()
            self.wGc = wGc.copy()
            if not Z is None:
                self.Z = Z.copy()

            '''extract sailent points'''
            self.extractPts(gthreshold)


        def extractPts(self, gthreshold=None):
            ''' 1.extract pixels with significant gradients'''
            h,w = self.im.shape

            grad,orin = scharr(self.im)
            self.grad, self.orin = grad,orin

            if gthreshold is None:
                gthreshold = np.percentile(grad, 80)
            u, v = np.meshgrid(range(w),range(h))
            mask = reduce(np.logical_and, [grad>gthreshold, u>1, v>1, u<w-2, v<h-2]) # exclude border pixels
            y,x = np.where(mask)
            self.py, self.px = y,x
            self.p_cnt = len(x)

            ''' 2. corresponding back-projected 3D point'''
            self.P = np.vstack([(x-cx)/fx,
                                (y-cy)/fy,
                                np.ones(self.p_cnt)])
            ''' 3. patch pixels'''
            patt = [(y,x),(y-2,x),(y-1,x+1),(y,x+2),(y+1,x+1),(y+2,x),(y+1,x-1),(y,x-2),(y-1,x-1)]
            self.v = np.vstack([self.im[ind].astype('i2') for ind in patt]).T
            ''' 4. Neighbors Info'''
            self.nbrs = self.setNeighborsInfo(mask)


        def setNeighborsInfo(self, mask_image):
            px, py = self.px, self.py
            node_cnt = self.p_cnt

            edges_forward = [[] for _ in range(node_cnt)]
            edges_backward = [[] for _ in range(node_cnt)]

            id_LUT = np.empty_like(mask_image, 'i4')
            id_LUT[py,px] = range(node_cnt)      # lookup-table of index number for valid pixels
            for p_id, (p_x,p_y) in enumerate(zip(px, py)):
                fcoord = [(p_y-1,p_x),(p_y,p_x-1),(p_y-1,p_x-1),(p_y-1,p_x+1)]
                fnbrs = [id_LUT[coord] for coord in fcoord if mask_image[coord]]
                if p_id-1 not in fnbrs:
                    fnbrs.append(p_id-1)
                edges_forward[p_id].extend(fnbrs)

                bcoord = [(p_y+1,p_x),(p_y,p_x+1),(p_y+1,p_x+1),(p_y+1,p_x-1)]
                bnbrs = [id_LUT[coord] for coord in bcoord if mask_image[coord]]
                if p_id+1 not in bnbrs:
                    fnbrs.append(p_id+1)
                edges_backward[p_id].extend(bnbrs)
            return edges_forward, edges_backward


        def calcPtsAngle(self, M):
            p0 = self.wGc[0:3,0:3].dot(self.P)
            p = M.dot(p0)
            theta = positive_range(np.arctan2(p[1], p[0]))
            phi = positive_range(np.arctan2(np.sqrt(p[0]**2 + p[1]**2), p[2]))
            return theta, phi

        def ProjTo(self, f):
            assert(self.Z is not None)
            assert(isinstance(f, Frame))
            G = np.dot(np.linalg.inv(f.wGc), self.wGc)
            P0 = self.P*self.Z[self.py, self.px]
            P1 = K.dot(G[0:3,0:3].dot(P0)+G[0:3,3][:,np.newaxis])
            return P1[:2]/P1[2]


    def calcF(rGc, K=K):
        ''' xr'*F*xc = 0 '''
        R,t = rGc[:3,:3],rGc[:3,3]
        rFc = inv(K.T).dot(skew(t)).dot(R).dot(inv(K))
        return rFc

    def calcEpl(xr,yr,rGc,K=K):
        ''' pc = pr_inf + depth*dxy '''
        cGr = inv(rGc)
        Rcr,Tcr = cGr[:3,:3],cGr[:3,3]
        pr = np.array([xr, yr, 1 if isScalar(xr) else np.ones(len(xr))])
        pr_inf = metric(K.dot(Rcr.dot(inv(K).dot(pr))))  # <= projection of points at infinity

        rFc = calcF(rGc)
        a,b,c = np.hsplit(pr.T.dot(rFc), 3)    # a*xc+b*yc+c=0
        norm = np.sqrt(a**2+b**2)
        dxy = np.hstack([np.sign(Tcr[0])*np.abs(b/norm), -np.sign(Tcr[1])*np.abs(a/norm)]).T  # TODO: principle?
        return pr_inf, dxy

    def checkEplRange(pr_inf, dxy):
        # pr_inf.x/y + d*dxy = {0,w}/{0,h}
        d = np.vstack([-pr_inf[0]/dxy[0], (w-pr_inf[0])/dxy[0], -pr_inf[1]/dxy[1], (h-pr_inf[1])/dxy[1]])
        d.sort(axis=0)
        d_min, d_max = d[1],d[2]
        pb1, pb2 = pr_inf+d_min*dxy, pr_inf+d_max*dxy # the two boundary point that is closest to pr_inf

        """ \2| 2\|3    Region 1 is the valid image area,
            --+---+--   suppose the epiline is going from pr_inf to bottom-right
             \| 1 |\    then pb1 & pb2 are the interception points of epi-line and the 4 extended image border
            --+---+--   If pr_inf is in Region 3 & 4, then the epiline does not touch the image area
             3|\4 |4 \

        if   isInValidReigion(pr_inf):            k_min,k_max = 0,d_max
        elif isInValidReigion(pb1) and d_min>0: k_min,k_max = d_min, d_max
        else:  k_min,k_max = 0,0 """
        isInValidReigion = lambda p: conditions(p[0]>0,p[0]<w,p[1]>0,p[1]<h)
        cond = [ isInValidReigion(pr_inf),      # pr_inf is in Region 1, so everything is fine
                 np.logical_and(isInValidReigion(pb1), d_min>0)] # pr_inf is in Region 2
        k_max = np.select( cond, [d_max, d_max], default = 0)
        k_min = np.select( cond[1], [d_min] )


    def test_calcEpl():
        f,a = plt.subplots(1, 1, num='test_F')
        a.imshow(sim(f0.im, f1.im))
        while 1:
            plt.pause(0.01)
            pref = np.round(plt.ginput(1, timeout=-1)[0])
            a.plot(pref[0], pref[1],'r.')

            cGr = relPos(f1.wGc, f0.wGc)
            pcur = trueProj(pref[0], pref[1], cGr=cGr, Zr=f0.Z)
            a.plot(pcur[0]+640, pcur[1],'b.')

            pb,pe,dxy,dm = calcEpl(pref[0], pref[1], inv(cGr))
            a.plot([pb[0]+640,pe[0]+640], [pb[1],pe[1]],'g-')

    def getG(f0,f1):
        return np.dot(inv(f0.wGc), f1.wGc)

    def triangulate(xr,yr,xc,yc,rGc):
        Rrc,Trc = rGc[:3,:3],rGc[:3,3]
        Baseline = np.linalg.norm(Trc)      # Edge C

        Base0 = Trc/Baseline      # epipolar: a unit vector pointed to the other camerea
        ray0 = snormalize(backproject(xr,yr))
        phi0 = np.arccos(ray0.dot(Base0))   # Angle a

        Base1 = -Rrc.T.dot(Base0)
        ray1 = snormalize(backproject(xc,yc))
        phi1 = np.arccos(ray1.dot(Base1))   # Angle b

        c = np.pi-phi1-phi0
        Range_r = Baseline*np.sin(phi1)/np.sin(c)     # Edge B = Edge C/sin(c)*sin(b)
        depth_r = ray0[2]*Range_r
        return depth_r

    ''' set up matching Frame'''
    refid = 4
    fs = []
    seq = [refid, 0,2,6,9]
    for fid in seq:
        f = Frame(frames[fid], wGc[fid], Z=Zs[fid])
        fs.append(f)
    f0,f1 = fs[0],fs[1]

    ''' plot all image'''
    if 1:
        fig = plt.figure(num='all image'); fig.clear()
        gs = plt.GridSpec(len(seq)-1,2)
        a =  fig.add_subplot(gs[:,0])
        b = [fig.add_subplot(gs[sp,1]) for sp in range(len(seq)-1)]

        ''' base image'''
        fref = fs[0]
        a.imshow(fref.im, interpolation='none')
        a.plot(fref.px, fref.py, 'r.' )

        ''' matching images'''
        for i,sp in enumerate(b):
            fcur = fs[i+1]
            sp.imshow(fcur.im, interpolation='none')
            sp.plot(fcur.px, fcur.py, 'b.', ms=2)
            pref = fref.ProjTo(fcur)
            sp.plot(pref[0], pref[1], 'r.', ms=2)
        fig.tight_layout()

    pinf,pcls,dxy,dmax = calcEpl(f0.px,f0.py, getG(f0,f1))
#    for p in xrange(f0.p_cnt):
#        pass

