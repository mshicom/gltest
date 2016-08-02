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



def metric(P): return P[:-1]/P[-1]
def skew(e): return np.array([[  0,  -e[2], e[1]],
                              [ e[2],    0,-e[0]],
                              [-e[1], e[0],   0]])
def isScalar(obj):
    return not hasattr(obj, "__len__")
def toArray(x):
    return np.array([x]) if isScalar(x) else x

def sample(dIc,x,y):
    if isScalar(x):
        x,y = np.array([x]),np.array([y])
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)

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
        zr = sample(Zr, xr, yr)
        pr = backproject(xr, yr)*zr
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
            self.im = img.astype('f')/255.0
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

        def searchEPL(self, f1, win_width=4):

            # xr, yr, win_width = pref[0], pref[1],4
            px, py = self.px, self.py
            node_cnt = self.p_cnt
            rGc = relPos(self.wGc, f1.wGc)

            pb,dm,dxy,dxy_local = calcEpl(px, py, rGc)
            best = np.empty((2,node_cnt),'f')


            for n in xrange(node_cnt):
                print n
                p0 = np.array([px[n],py[n]])

                ref_pos = vec(p0) - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1) # TODO: correct the sign
                ref_samples = sample(self.im, ref_pos[0], ref_pos[1])

                sam_cnt = np.floor(dm[n])
                cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(-win_width, sam_cnt+win_width)
                cur_samples = sample(f1.im, cur_pos[0], cur_pos[1])
                err = np.empty(sam_cnt,'f')
#                if 0:
#                for i in xrange(int(sam_cnt)):
#                    diff = ref_samples - cur_samples[i:i+2*win_width+1]
#                    vm = ~np.isnan(diff)
#                    err[i] = np.minimum(diff[vm]**2, 1**2).sum()/vm.sum()
#                best[:,n] = cur_pos[:,np.nanargmin(err)+win_width]
#                else:
                code = r'''
                for(int i=0; i<(int)sam_cnt; i++ )
                {
                    float diff = 0;
                    for(size_t j=0; j<2*%d+1;j++ )
                    {
                        float cur = CUR_SAMPLES1(i+j);
                        diff += std::isnan(cur)? 0 : std::fabs( REF_SAMPLES1(j) - cur);
                    }
                    ERR1(i) = diff;
                }
                ''' % win_width
                weave.inline(code,
                   ['ref_samples','cur_samples','err', 'sam_cnt'],
                    compiler='gcc',headers=['<chrono>','<cmath>'],
                    extra_compile_args=['-std=gnu++11 -msse2 -O3'],
                    verbose=2 )
                best[:,n] = cur_pos[:,np.nanargmin(err)+win_width]
            return best

        def makePC(self, depth, vmin=0, vmax=10):
            vm_ind, = np.where(conditions(depth>vmin,  depth<vmax))
            p3d = self.P*depth
            I = np.tile(self.im[self.py, self.px]*255,(3,1))
            P = np.vstack([p3d, I]).T
            return P[vm_ind,:]      # Nx6





    def calcF(rGc, K=K):
        ''' xr'*F*xc = 0 '''
        R,t = rGc[:3,:3],rGc[:3,3]
        rFc = inv(K.T).dot(skew(t)).dot(R).dot(inv(K))
        return rFc

    def calcEpl(xr,yr,rGc,K=K):
        ''' pc = Pinf + depth*dxy '''
        cGr = inv(rGc)
        Rcr,Tcr = cGr[:3,:3],cGr[:3,3]
        pr = np.array([xr, yr, 1 if isScalar(xr) else np.ones(len(xr))])
        Pinf = K.dot(Rcr.dot(inv(K).dot(pr)))  # <= projection of points at infinity
        Pinf /= Pinf[2]
        Pe = K.dot(Tcr)

        rFc = calcF(rGc)
        a,b,c = np.hsplit(pr.T.dot(rFc), 3)    # a*xc+b*yc+c=0
        norm = np.sqrt(a**2+b**2)
        dxy = np.hstack([np.sign(Tcr[0])*np.abs(b/norm), -np.sign(Tcr[1])*np.abs(a/norm)]).T  # TODO: principle?

        x_limit = np.maximum(-Pinf[0]/dxy[0], (w-Pinf[0])/dxy[0])   # Pinf.x + x_limit*dx = {0,w}
        y_limit = np.maximum(-Pinf[1]/dxy[1], (h-Pinf[1])/dxy[1])   # Pinf.y + y_limit*dy = {0,h}
        dinv_max = np.minimum(x_limit, y_limit)
        Pe = Pinf[:2] + dinv_max*dxy

        Trc = rGc[:3,3]
        a,b,c = np.hsplit(rFc.dot(Pinf).T, 3)    # a*xc+b*yc+c=0
        norm = np.sqrt(a**2+b**2)
        dxy_local = np.hstack([np.sign(Trc[0])*np.abs(b/norm), -np.sign(Trc[1])*np.abs(a/norm)]).T  # TODO: principle?

#        '''P = Pinf + λ*Pe'''
#        az = -Pinf[2]/Pe[2]                         # Pz = Pinf[2] + λ*Pe[2] > 0
#        ax = (Pinf[0]-w*Pinf[2])/(w*Pe[2]-Pe[0])    # 0 < fx*(Px/Pz) + cx < w
#        ay = (Pinf[1]-h*Pinf[2])/(h*Pe[2]-Pe[1])    # 0 < fy*(Py/Pz) + cx < h
#
##        inf = np.full_like(az, np.inf)
#        max_idepth0, min_idepth0 = (inf, az) if Pe[2]>0 else (az, 0)
##        A = -cx/fx
##        if Pe[2]>0:
##            denominator = Pe[0]-A*Pe[2]
##            ax = (A*Pinf[2]-Pinf[0])/denominator
##            max_idepth0, min_idepth0 = (inf, az) if denominator>0 else (az, 0)
#
#        max_idepth = np.min(np.vstack([ax, ay, az]),axis=0)
#        Pe = Pinf + max_idepth*Pe
#
#        Pe = Pe/Pe[2]
#        Pinf = Pinf/Pinf[2]

        return Pinf[:2],dinv_max, dxy, dxy_local



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

            pb,dmax,dxy,dxy_local = calcEpl(pref[0], pref[1], inv(cGr))
            a.plot([pb[0]+640,pe[0]+640], [pb[1],pe[1]],'g-')
            a.plot([pcur[0], pcur[0]+100*dxy_local[0]], [pcur[1],pcur[1]+100*dxy_local[1]],'b-')

    def getG(f0,f1):
        return np.dot(inv(f0.wGc), f1.wGc)

    def triangulate(xr,yr,xc,yc,rGc):
        Rrc,Trc = rGc[:3,:3],rGc[:3,3]
        Baseline = np.linalg.norm(Trc)      # Edge C

        Base0 = Trc/Baseline      # epipolar: a unit vector pointed to the other camerea
        ray0 = snormalize(backproject(xr,yr))
        phi0 = np.arccos(ray0.T.dot(Base0))   # Angle a

        Base1 = -Rrc.T.dot(Base0)
        ray1 = snormalize(backproject(xc,yc))
        phi1 = np.arccos(ray1.T.dot(Base1))   # Angle b

        c = np.pi-phi1-phi0
        Range_r = Baseline*np.sin(phi1)/np.sin(c)     # Edge B = Edge C/sin(c)*sin(b)
        depth_r = ray0[2]*Range_r
        return depth_r

    ''' set up matching Frame'''
    refid = 4
    fs = []
    seq = [refid, 0]
    for fid in seq:
        f = Frame(frames[fid], wGc[fid], Z=Zs[fid])
        fs.append(f)
    f0,f1 = fs[0],fs[1]
    match = f0.searchEPL(f1,4)
    d = triangulate(f0.px,f0.py,match[0],match[1], getG(f0,f1))
    plotxyzrgb(f0.makePC(d, 0, 5))

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


#    pinf,pcls,dxy = calcEpl(f0.px,f0.py, getG(f0,f1))
#    c = savgol_coeffs(5,2,pos=2,deriv=1,use='dot')

    def eplMatch(xr, yr, f0, f1, win_width = 3, debug=False):
        # xr, yr, win_width,debug = pref[0], pref[1],4,True
        rGc = relPos(f0.wGc, f1.wGc)
        pb,dm,dxy,dxy_local = calcEpl(xr, yr, rGc)

        p0 = np.array([xr,yr])

        ref_pos = vec(p0) - vec(dxy_local)*np.arange(-win_width, win_width+1) # TODO: correct the sign
        ref_samples = sample(f0.im, ref_pos[0], ref_pos[1])

        sam_cnt = np.floor(dm).astype('i')
        cur_pos = vec(pb) + vec(dxy)*np.arange(-win_width, sam_cnt+win_width)
        cur_samples = sample(f1.im, cur_pos[0], cur_pos[1])

        err = np.full(sam_cnt,np.inf,'f')
        if 0:
            for i in xrange(sam_cnt):
                err[i] = np.sum(np.minimum(ref_samples - cur_samples[i:i+2*win_width+1], 1)**2)
            best = np.nanargmin(err)
        else:
            code = r'''
            int N = (int)sam_cnt;
            for(int i=0; i<N; i++ )
            {
                float diff = 0;
                for(size_t j=0; j<2*(size_t)win_width+1;j++ )
                    diff += std::fabs( REF_SAMPLES1(j) - CUR_SAMPLES1(i+j));
                ERR1(i) = diff;
            }
//            std::vector<float> vec(err, err+N);
//            int winner = vec.begin() - std::min_element(vec.begin(), vec.end());
//            return_val = Py::new_reference_to(Py::Int(winner));
            '''
            weave.inline(code,
               ['ref_samples','cur_samples','err','sam_cnt','win_width'],
                compiler='gcc',headers=['<algorithm>','<cmath>','<vector>'],
                extra_compile_args=['-std=gnu++11 -msse2 -O3'],
                verbose=2  )
            best = np.nanargmin(err)

        if debug:
            f = plt.figure(num='epl match')
            gs = plt.GridSpec(2,2)
            al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
            ab = f.add_subplot(gs[1,:])

            al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')
            pt = trueProj(ref_pos[0],ref_pos[1], cGr=inv(rGc), Zr=f0.Z)
            ar.plot(pt[0], pt[1],'b.')
            al.plot(ref_pos[0], ref_pos[1],'g.'); al.plot(xr, yr,'r.')
            ar.plot(cur_pos[0], cur_pos[1],'g.')
            ab.plot(err)
            ab.vlines(best,0,1)
            ar.plot(cur_pos[0,best+win_width], cur_pos[1,best+win_width],'r*')
        return best

    def test_EPLMatch():
        f = plt.figure(num='epl match')
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')

        pref = plt.ginput(1, timeout=-1)[0]
        best0 = eplMatch(pref[0], pref[1], f0, f1, win_width=3, debug=True)




