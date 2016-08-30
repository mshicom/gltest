#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:50:58 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

from tools import loaddata1
def sample(dIc,x,y):
    x,y = np.atleast_1d(x, y)
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)

def normalize(P):
    '''normalize N points seperately, dim(P)=3xN'''
    return P/np.linalg.norm(P, axis=0)

def projective(x, y):
    x,y = np.atleast_1d(x,y)   # scalar to array
    return np.vstack([x.ravel(), y.ravel(), np.ones_like(x)])

def transform(G,P):
    ''' Pr[3,N]   Pr = rGc*Pc'''
    return G[:3,:3].dot(P)+G[:3,3][:,np.newaxis]

def vec(*arg):
    return np.reshape(arg,(-1,1))

inv = np.linalg.inv

def conditions(*args):
    return reduce(np.logical_and, args)

def backproject(x, y, K):
    ''' return 3xN backprojected points array, x,y,z = p[0],p[1],p[2]'''
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    x,y = np.atleast_1d(x,y)   # scalar to array
    x,y = x.ravel(), y.ravel()
    return np.vstack([(x-cx)/fx, (y-cy)/fy, np.ones_like(x)])

pis = plt.imshow
pf = plt.figure
def sim(*arg,**kwarg):
    return np.hstack(arg)

def skew(e):
    return np.array([[  0,  -e[2], e[1]],
                     [ e[2],    0,-e[0]],
                     [-e[1], e[0],   0]])

def calcF(rGc, K):
    ''' xr'*F*xc = 0 '''
    R,t = rGc[:3,:3],rGc[:3,3]
    rFc = inv(K.T).dot(skew(t)).dot(R).dot(inv(K))
    return rFc

def relG(wG0, wG1):
    return np.dot(np.linalg.inv(wG0), wG1)

class EpilineCalculator(object):
    ''' 1. suppose (X, X') are the Ref and Cur image pixel pairs,
        given the relative camera pos (R,T), X'=(R,T).dot(X), we have a ray X'∈R3:
            X' = K*(R*inv(K)*X*z + T)
               = K*R*inv(K)*X + 1/z*K*T
               =    Pinf[1:3] +   λ*Pe[1:3]  (λ:=1/z)
        The projected image point x'∈R2 of X' will be:
            x' = (Pinf[1:2] + λ*Pe[1:2])/(Pinf[3]+ λ*Pe[3])     (eq.1)
        but what we want is x' in this form:
            x' =  Pinf[1:2]/Pinf[3] + λ*dxy[1:2]                (eq.2a)
               =  Pinf[1:2]/Pinf[3] + v*normalize(dxy[1:2])     (eq.2b)
        v is disparity in pixels along the epi-polar line.
        Putting eq.1 & eq.2a together and solve for dxy:
              dxy[1:2] = 1/(Pinf[3]+ λ*Pe[3]) * (-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2])   (eq.3)
        so normalize(dxy) = normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]),  if (Pinf[3]+ λ*Pe[3])>0,
                          = -normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]), otherwise.
        2.We need to make sure (Pinf[3]+ λ*Pe[3])>0, which means the triangulated
        3D point will be in front of the Cur camera, not behind it. Also λ is
        non-negative, this leads to 2 cases:
          a.If camera Ref is in front of Cur (i.e. Pe[3]>0), we have:
                      λ > 0 > -Pinf[3]/Pe[3], so λ_max=inf
            everything is fine;
          b.Otherwise camera Ref is behind Cur (i.e. Pe[3]<0), then
                      -Pinf[3]/Pe[3] > λ > 0, then λ_max=-Pinf[3]/Pe[3].
            which limits the minimum depth Z ( or maximum inverse depth).
        3. Also from eq.2 we have:
                        v*normalize(dxy[1:2]) = λ*dxy[1:2]
        if λ is given, then
                        λ/(Pinf[3] + λ*Pe[3])*dxy_norm = v,          (eq.4)
        equivalently, if v is given, then
                        λ = a*Pinf[3]/(1- a*Pe[3]),  a:=v/dxy_norm   (eq.5)
        with two special cases:
            z_max=inf -> λ_min=0   -> v_min=0,
            z_min=0   -> λ_max=inf -> v_max = dxy_norm/Pe[3] (i.e. 1-a*Pe[3]=0)
        4. if x' is given, reorganize eq.1:
            x'[1:2]*Pe[3]*λ - Pe[1:2]*λ = Pinf[1:2] - x'[1:2]*Pinf[3]
            λ = (Pinf[1:2] - x'[1:2]*Pinf[3])/(x'[1:2]*Pe[3] - Pe[1:2])
    '''
    def __init__(self, xr, yr, rGc, K):
        # xr, yr, rGc, K = f0.px, f0.py, getG(f0,f1), K
        xr,yr = np.atleast_1d(xr,yr)

        Rrc,Trc = rGc[:3,:3],rGc[:3,3]
        Rcr,Tcr = Rrc.T, -Rrc.T.dot(Trc)

        Pr = projective(xr, yr)                                 # 3xN
        Pe0 = vec(K.dot(Trc))                                   # 3x1
        dxy_local = normalize(-Pe0[2]/Pr[2]*Pr[:2]+Pe0[:2])     # 2xN

        Pinf = (K.dot(Rcr.dot(inv(K)))).dot(Pr)
        nPinf = Pinf[:2]/Pinf[2]
        Pe = vec(K.dot(Tcr))

        dxy_raw = -Pe[2]/Pinf[2]*Pinf[:2]+Pe[:2]              # 2xN,
        dxy_norm = np.linalg.norm(dxy_raw, axis=0)              # N
        dxy = dxy_raw/dxy_norm                                  # 2xN
        self.dxy = dxy
        self.dxy_local = dxy_local
        self.nPinf = nPinf
        self.Pb = inv(K).dot(Pr)
        self.Pe = Pe
        self.Pinf = Pinf

        self.VfromD = lambda  d,ind=slice(None): np.where(d==np.inf, dxy_norm[ind]/Pe[2], d/(Pinf[2,ind] + d*Pe[2])*dxy_norm[ind])
        self.VfromX = lambda xc,ind=slice(None): (xc - nPinf[0,ind])/dxy[0,ind]
        self.VfromY = lambda yc,ind=slice(None): (yc - nPinf[1,ind])/dxy[1,ind]

        self.DfromV = lambda  v,ind=slice(None): v*Pinf[2,ind]/(dxy_norm[ind] - v*Pe[2])
        self.DfromX = lambda xc,ind=slice(None): (Pinf[0,ind] - xc*Pinf[2,ind])/(xc*Pe[2]-Pe[0])
        self.DfromY = lambda yc,ind=slice(None): (Pinf[1,ind] - yc*Pinf[2,ind])/(yc*Pe[2]-Pe[1])

        self.XYfromV = lambda v,ind=slice(None): ( nPinf[0,ind] + v*dxy[0,ind], nPinf[1,ind] + v*dxy[1,ind])
        self.XYfromD = lambda d,ind=slice(None): ((Pinf[0,ind]+d*Pe[0])/(Pinf[2,ind]+d*Pe[2]), \
                                                  (Pinf[1,ind]+d*Pe[1])/(Pinf[2,ind]+d*Pe[2]))
        self.XYfromV_local=lambda v,ind=slice(None): (Pr[0,ind]+v*dxy_local[0,ind], Pr[1,ind]+v*dxy_local[1,ind])

        def Triangulate(xc, yc, ind=slice(None)):
            ''' pixel -> ray -> 2 angles -> rules of sine -> edge length -> depth'''
            Baseline = np.linalg.norm(Trc)      # Edge C
            Base0 = Trc/Baseline      # epipolar: a unit vector pointed to the other camerea
            ray0 = normalize(backproject(xr[ind], yr[ind], K))
            phi0 = np.arccos(ray0.T.dot(Base0))   # Angle a

            Base1 = -Rrc.T.dot(Base0)
            ray1 = normalize(backproject(xc, yc, K))
            phi1 = np.arccos(ray1.T.dot(Base1))   # Angle b

            c = np.pi-phi1-phi0
            Range_r = Baseline*np.sin(phi1)/np.sin(c)     # Edge B/sin(b)= Edge C/sin(c)
            Z = ray0[2]*Range_r
            return Z
        self.ZfromXY = Triangulate

        def getLimits(shape, dmin=0.0, dmax=1e6):
            '''There are in total 5 constraints in the epiline parameters λ and v:
                a. Valid image region: 0<x<w, 0<y<h;
                b. Expected search range indicated by [dmin, dmax];
                c. if Pe[3]>0, v_max=dxy_norm/Pe[3] for λ=np.inf
                d. if Pe[3]<0, Ref is behind Cur, λ<-Pinf[2]/Pe[2] to ensure the resulting 3D point
                    will be in front of the Cur camera,
                e. total epiline length is no less than 4
            '''
            # a. valid border is trimmed a little bit, i.e. 5 pixels, for
            h,w = shape
            tx = self.VfromX(vec(5, w-5))
            tx = np.where(dxy[0]>0, tx, np.roll(tx,1,axis=0))   # tx[0,1] := [v_xmin,v_xmax]
            ty = self.VfromY(vec(5, h-5))
            ty = np.where(dxy[1]>0, ty, np.roll(ty,1,axis=0))   # ty[0,1] := [v_ymin,v_ymax]\
            v_xmin,v_xmax,v_ymin,v_ymax = tx[0],tx[1],ty[0],ty[1]
            vmax = np.minimum(v_xmax, v_ymax)
            vmin = np.maximum(v_xmin, v_ymin)
            valid_mask = conditions(v_xmin<v_ymax, v_xmax>v_ymin, vmax>0)

            # c
            if Pe[2]<0:
                dmax = np.clip(dmax, dmin, (1e-3-Pinf[2])/Pe[2])
            # b
            vmin = np.maximum(vmin, self.VfromD(dmin))
            vmax = np.minimum(vmax, self.VfromD(dmax))
            # d
            if Pe[2]>0:
                vmax = np.minimum(vmax, dxy_norm/Pe[2])
            d_min, d_max = self.DfromV(vmin), self.DfromV(vmax)
            # e
            valid_mask = conditions(valid_mask, (vmax-vmin)>1)

            return vmin,vmax, d_min, d_max, valid_mask
        self.getLimits = getLimits

        def searchEPL(imr, imc, win_width=3, index=None, dmin=0.0, dmax=1e6):
            if index is None:
                index = range(len(xr))
            index = np.atleast_1d(index)

            offset = np.arange(-win_width, win_width+1)
            vmin,vmax, d_min, d_max, valid_mask = getLimits(imr.shape, dmin, dmax)
            res,dom = [],[]
            for p_id in index:
                if not valid_mask[p_id]:
                    res.append([])
                    dom.append([])
                    continue
                sam_min,sam_max = np.floor(vmin[p_id]), np.ceil(vmax[p_id])
                sam_cnt = int(sam_max-sam_min+1)
                dom.append(self.DfromV(np.arange(sam_min, sam_max+1), p_id))
                err = np.empty(sam_cnt, 'f')

                ref_pos = vec(xr[p_id], yr[p_id]) - vec(dxy_local[:,p_id])*offset
                cur_pos = vec(nPinf[:,p_id]) + vec(dxy[:,p_id])*np.arange(np.floor(vmin[p_id])-win_width, np.ceil(vmax[p_id])+win_width+1)
                ref = sample(imr, ref_pos[0], ref_pos[1])
                cur = sample(imc, cur_pos[0], cur_pos[1])
                for i in xrange(sam_cnt):
                    diff = ref - cur[i:i+2*win_width+1]
                    err[i] = np.sum(diff**2)
                res.append(err)
            if len(index)==1:
                return res[0],dom[0]
            else:
                return res,dom
        self.searchEPL = searchEPL

class EpilineDrawer(object):
    def __init__(self, frames, wGc, K, p0=None):

        self.ind = 0
        self.ecs = {}
        slices= len(frames)-2

        f = plt.figure()
        gs = plt.GridSpec(2,2)
        a1,a2 = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        a3 = f.add_subplot(gs[1,:])
        #plt.tight_layout()

        a1.imshow(frames[0], interpolation='none', aspect=1)
        a1.set_title('pick a point in this image')
        a1.autoscale(False)

        i2 = a2.imshow(frames[1], interpolation='none', aspect=1)
        a2.autoscale(False)

        if p0 is None:
            xr, yr = np.round(plt.ginput(1, timeout=-1)[0])
        else:
            xr, yr = p0
        a1.plot(xr, yr,'r.')
        a1.set_title(' ')

        line1, = a2.plot([],'r-')
        dots, = a2.plot([],'b.',ms=5)
        Z = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        res,dom = {},{}

        for index in range(1, len(frames)):
            rGc = relG(wGc[0], wGc[index])
            ec = EpilineCalculator(xr, yr, rGc, K)
            res[index], dom[index] = ec.searchEPL(frames[0].astype('f'), frames[index].astype('f'),3,0)
            self.ecs[index] = ec
        curv1, = a3.plot(res[1])

        def update(index):
            i2.set_data(frames[index])
            a2.set_title('frame %s' % index)

            rGc = relG(wGc[0], wGc[index])
            ec = self.ecs[index]
            dom_, res_ = dom[index], res[index]
            vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(frames[index].shape)
            if valid_mask:
                pmin = ec.XYfromV(vmin)
                pmax = ec.XYfromV(vmax)
                line1.set_data([pmin[0],pmax[0]],[pmin[1],pmax[1]])
                cGr = inv(rGc)
                pcur = K.dot(transform(cGr, backproject(xr, yr, K)*Z))
                pcur /= pcur[2]
                dots.set_data(pcur[0],pcur[1])
                curv1.set_data(dom_, res_)
                a3.set_xlim(dom_[0], dom_[-1])
            else:
                print 'epiline not valid'
                line1.set_data([],[])
                dots.set_data([],[])
                curv1.set_data([],[])

            f.canvas.draw()

        def onscroll(event):
            if event.button == 'down':
                self.ind = np.clip(self.ind + 1, 0, slices)
            else:
                self.ind = np.clip(self.ind - 1, 0, slices)
            update(self.ind+1)

        update(self.ind+1)
        f.canvas.mpl_connect('scroll_event', onscroll)


if __name__ == "__main__":
    frames, wGc, K, _ = loaddata1()
    e=EpilineDrawer(frames, wGc, K)
