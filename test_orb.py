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

def loaddata2():
    with np.load("kfrecord.npz") as data:
        frames = data["frames"]
        G = data["Gs"]
        K = data['K']
        r = data['r']

    return frames, G, K

def metric(P): return P[:-1]/P[-1]
def skew(e): return np.array([[  0,  -e[2], e[1]],
                              [ e[2],    0,-e[0]],
                              [-e[1], e[0],   0]])
def isScalar(obj):
    return not hasattr(obj, "__len__")

def sample(dIc,x,y):
    x,y = np.atleast_1d(x, y)
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, cval=np.nan)

def relPos(wG0, wG1):
    return np.dot(np.linalg.inv(wG0), wG1)

def transform(G,P):
    ''' Pr[3,N]   Pr = rGc*Pc'''
    return G[:3,:3].dot(P)+G[:3,3][:,np.newaxis]

def conditions(*args):
    return reduce(np.logical_and, args)

def normalize(P):
    '''normalize N points seperately, dim(P)=3xN'''
    return P/np.linalg.norm(P, axis=0)

def vec(*arg):
    return np.reshape(arg,(-1,1))

inv = np.linalg.inv

from functools import wraps
from time import time
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print 'func:%r took: %2.6f sec' % (f.__name__, te-ts)
        return result
    return wrap

if __name__ == "__main__":
    frames, wGc, K, Zs = loaddata1()
#    frames, wGc, K = loaddata2()
    h,w = frames[0].shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    def projective(x, y):
        x,y = np.atleast_1d(x,y)   # scalar to array
        return np.vstack([x.ravel(), y.ravel(), np.ones_like(x)])

    def backproject(x, y, K=K):
        ''' return 3xN backprojected points array, x,y,z = p[0],p[1],p[2]'''
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
        x,y = np.atleast_1d(x,y)   # scalar to array
        x,y = x.ravel(), y.ravel()
        return np.vstack([(x-cx)/fx, (y-cy)/fy, np.ones_like(x)])


    def trueProj(xr, yr, cGr, Zr):
        # xr, yr, cGr, Zr = 0, 0, getG(f1,f0), f0.Z
        zr = sample(Zr, xr, yr)
        pr = backproject(xr, yr)*zr
        pc =  K.dot(transform(cGr, pr))
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
            self.im = np.ascontiguousarray(img.astype('f')/255.0)
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
            border_width = 4
            mask = reduce(np.logical_and, [grad>gthreshold,
                                           w-border_width>u, u>=border_width,
                                           h-border_width>v, v>=border_width]) # exclude border pixels
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
            rGc = relPos(self.wGc, f1.wGc)
            return searchEPL(self.px, self.py, self.im, f1.im, rGc, win_width)

        def makePC(self, depth, vmin=0, vmax=10):
            vm_ind, = np.where(conditions(depth>vmin,  depth<vmax))
            p3d = self.P*depth
            I = np.tile(self.im[self.py, self.px]*255,(3,1))
            P = np.vstack([p3d, I]).T
            return P[vm_ind,:]      # Nx6

    def getG(f0,f1):
        '''return 1G0, which p1 = 1G0 * p0  '''
        return np.dot(inv(f0.wGc), f1.wGc)

    def calcF(rGc, K=K):
        ''' xr'*F*xc = 0 '''
        R,t = rGc[:3,:3],rGc[:3,3]
        rFc = inv(K.T).dot(skew(t)).dot(R).dot(inv(K))
        return rFc

    def calcEpl(xr,yr,rGc,K=K):
        ''' suppose (X, X') are the left and right image pixel pairs,
            given the relative camera pos (R,T), X'=(R,T).dot(X), we have a ray X'∈R3:
                X' = K*(R*inv(K)*X*z + T)
                   = K*R*inv(K)*X + 1/z*K*T
                   =    Pinf[1:3] +   λ*Pe[1:3]  (λ=1/z)
            The projected image point x'=[a,b,1] will be:
                x' = (Pinf[1:2] + λ*Pe[1:2])/(Pinf[3]+ λ*Pe[3]) (eq.1)
            but what we want is x' in this form:
                x' =  Pinf[1:2]/Pinf[3] + λ*dxy[1:2]   (eq.2)
            putting eq.1 & eq.2 together and solve for dxy, we get:
                dxy[1:2] = 1/(Pinf[3]+ λ*Pe[3]) * (-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2])
            so normalize(dxy) = normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]),  if (Pinf[3]+ λ*Pe[3])>0,
                              = -normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]), otherwise.
            for (Pinf[3]+ λ*Pe[3])>0 ,we have:
                λ > 0 > -Pinf[3]/Pe[3],  if Pe[3]>0;
                -Pinf[3]/Pe[3] > λ > 0,  otherwise

        '''
        # xr,yr,rGc = pref[0], pref[1], getG(f0,f1)
        # xr,yr,rGc = f0.px, f0.py, getG(f0,f1)
        xr,yr = np.atleast_1d(xr,yr)

        Rrc,Trc = rGc[:3,:3],rGc[:3,3]
        Rcr,Tcr = Rrc.T, -Rrc.T.dot(Trc)

        Pr = projective(xr, yr)     # 3xN
        Pe0 = vec(K.dot(Trc))           # 3x1
        dxy_local = normalize(-Pe0[2]/Pr[2]*Pr[:2]+Pe0[:2])  # 2xN

        H = K.dot(Rcr.dot(inv(K)))
        Pinf = H.dot(Pr)  # <= projection of points at infinity, Pinf=K.dot(Rcr.dot(backproject(xr, yr))
        Pe1 = vec(K.dot(Tcr))
        dxy = normalize(-Pe1[2]/Pinf[2]*Pinf[:2]+Pe1[:2])   # 2xN

        x_limit = np.maximum(-Pinf[0]/dxy[0], (w-Pinf[0])/dxy[0])   # Pinf.x + x_limit*dx = {0,w}
        y_limit = np.maximum(-Pinf[1]/dxy[1], (h-Pinf[1])/dxy[1])   # Pinf.y + y_limit*dy = {0,h}
        dinv_max = np.minimum(x_limit, y_limit)             # N

        return Pinf[:2]/Pinf[2], dinv_max, dxy, dxy_local

    def calcInvDepth(xr,yr,xc,yc,cGr,K=K):
        '''Once we have x'[1:2], puting it in eq.1 and solve for λ, we get:
                λ = (Pinf[1:2] - Pinf[3]*x'[1:2])/(Pe[3]*x'[1:2]-Pe[1:2])
        '''
        # xr,yr,cGr,(xc,yc) = f0.px, f0.py, getG(f1,f0),trueProj(f0.px, f0.py, getG(f1,f0), Zr=f0.Z)
        xr,yr,xc,yc = np.atleast_1d(xr,yr,xc,yc)
        Rcr,Tcr = cGr[:3,:3],cGr[:3,3]

        H = K.dot(Rcr.dot(inv(K)))
        Pinf = H.dot(projective(xr, yr))  # Pinf=K.dot(Rcr.dot(backproject(xr, yr))
        Pe = vec(K.dot(Tcr))
        dinv = (Pinf[0] - Pinf[2]*xc)/(Pe[2]*xc-Pe[0])     # or (Pinf[1] - Pinf[2]*yc)/(Pe[2]*yc-Pe[0])
        return dinv

    def test_calcEpl():
        f,a = plt.subplots(1, 1, num='test_F')
        a.imshow(sim(f0.im, f1.im))

        pref = np.round(plt.ginput(1, timeout=-1)[0])
        a.plot(pref[0], pref[1],'r.')

        cGr = relPos(f1.wGc, f0.wGc)
        Z = np.linspace(0.1, 5.0, 40)
        pcur = K.dot(transform(cGr, backproject(pref[0], pref[1])*Z))
        pcur /= pcur[2]
        a.plot(pcur[0]+640, pcur[1],'b.')

        pb,dmax,dxy,dxy_local = calcEpl(pref[0], pref[1], inv(cGr))
        pe = pb+dmax*dxy
        a.plot([pb[0]+640,pe[0]+640], [pb[1],pe[1]],'g-')
#        a.plot([pcur[0], pcur[0]+100*dxy_local[0]], [pcur[1],pcur[1]+100*dxy_local[1]],'b-')
        plt.pause(0.01)

    def test_calcInvDepth():
        cGr = getG(f1,f0)
        tx,ty = trueProj(f0.px, f0.py, cGr, Zr=f0.Z)
        td = sample(f0.Z, f0.px, f0.py)
        d = 1.0/calcInvDepth(f0.px, f0.py,tx,ty,cGr)
        print np.allclose(td, d)

    @timing
    def searchEPL(px, py, imr, imc, rGc, win_width=4, debug=False):
        # px, py, imr, imc, rGc, win_width, debug = pref[0], pref[1], f0.im, f1.im, getG(f0,f1), 4, True
        px,py = np.atleast_1d(px, py)

        node_cnt = len(px)
        pb,dm,dxy,dxy_local = calcEpl(px, py, rGc)
        best = np.empty(node_cnt,'i')
        cost = np.empty(node_cnt,'f')
        if 0 or debug:
            for n in xrange(node_cnt):
                print n
                p0 = vec(px[n],py[n])

                ref_pos = p0 - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1) # TODO: why minus?
                ref_samples = sample(imr, ref_pos[0], ref_pos[1])

                sam_cnt = np.floor(dm[n])
                if sam_cnt<1:
                    best[n] = -1
                    continue

                cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(-win_width, sam_cnt+win_width)
                cur_samples = sample(imc, cur_pos[0], cur_pos[1])
                err = np.empty(sam_cnt,'f')

                for i in xrange(int(sam_cnt)):
                    diff = ref_samples - cur_samples[i:i+2*win_width+1]
                    err[i] = (diff*diff).sum()
                best[n] = np.nanargmin(err)  #np.argpartition(err, 5)
                cost[n] = err[best[n]]

                if debug:
                    ''' obtain or create plotting handle'''
                    f = plt.figure(num='epl match')
                    if f.axes:
                        al,ar,ab = tuple(f.axes)
                    else:
                        gs = plt.GridSpec(2,2)
                        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
                        ab = f.add_subplot(gs[1,:])
                        al.imshow(imr, interpolation='none'); ar.imshow(imc, interpolation='none')
                    ''' '''
                    al.plot(ref_pos[0], ref_pos[1],'g.'); al.plot(px, py,'r.')
                    pm = pb+best*dxy
                    ar.plot(cur_pos[0], cur_pos[1],'g.');ar.plot(pm[0],pm[1],'r*');
                    ab.plot(err)
                    ab.vlines(best,0,1)

                    if not f0.Z is None:
                        pt = trueProj(ref_pos[0],ref_pos[1], cGr=inv(rGc), Zr=f0.Z)
                        ar.plot(pt[0], pt[1],'b.')
                        ab.vlines((pt[0][win_width]-pb[0])/dxy[0],0,1,'g')
        else:
            scode = r'''
            inline float sample(const float* const mat, const int width, const float x, const float y)
            {
                	int ix = (int)x;
                	int iy = (int)y;
                	float dx = x - ix;
                	float dy = y - iy;
                	float dxdy = dx*dy;
                	const float* bp = mat +ix+iy*width;
                	float res =   dxdy * bp[1+width]
                				+ (dy-dxdy) * bp[width]
                				+ (dx-dxdy) * bp[1]
                				+ (1-dx-dy+dxdy) * bp[0];
                	return res;
            } '''
            code = r'''
            size_t M = node_cnt;
            const int win_width = %(win_width)d;
            const int win_size = 2*win_width+1;
            std::vector<float> ref_samples(2*win_width+1);
            std::vector<float> cur_samples(1000);
            std::vector<float> err(1000);

            // std::raise(SIGINT);

            // foreach pixel in ref image
            for(size_t p_id=0; p_id<M; p_id++){

                /* 1. Sampling the Intensity */
                int sample_size = std::floor(DM1(p_id));
                if(sample_size<1){
                    BEST1(p_id) = -1;
                    COST1(p_id) = std::numeric_limits<float>::infinity();
                    continue;   // discard pixel whose epi-line length is 0

                }
                cur_samples.resize(sample_size+2*win_width);
                err.resize(sample_size);
                for(int i=0; i<sample_size+2*win_width; i++)
                    cur_samples[i] = sample(imb, width,
                                            PB2(0,p_id)+(i-win_width)*DXY2(0,p_id),
                                            PB2(1,p_id)+(i-win_width)*DXY2(1,p_id));

                for(int pos=0; pos<win_size; pos++)
                    ref_samples[pos] = sample(ima,width,
                                            PX1(p_id)-(pos-win_width)*DXY_LOCAL2(0,p_id),
                                            PY1(p_id)-(pos-win_width)*DXY_LOCAL2(1,p_id));

                /* 2. go through all the points in cur */
                float min_diff = std::numeric_limits<float>::infinity();
                for(int i=0; i<sample_size; i++ ){
                    // speed up: check diff in central pixel, skip if too large
                    //if( std::fabs(ref_samples[win_width]-cur_samples[i+win_width])> 30/255.0)
                    // continue;

                    float diff = 0;
                    for(int j=0; j<win_size;j++ ){
                        float err = ref_samples[j] - cur_samples[i+j];
                        diff += err*err;
                    }
                    err[i] = diff;
                    #if 0
                        if (diff<min_diff){
                            min_diff = diff;
                            BEST1(p_id) = i;
                        }
                    #endif
                }
                COST1(p_id) = min_diff;
                /* 3. find the best N element */
                #if 1
                    auto result = std::min_element(std::begin(err), std::end(err));
                    BEST1(p_id) = std::distance(std::begin(err), result);
                    COST1(p_id) = *result;
                #endif
            }
            '''% {'win_width': win_width }
            ima,imb,width = imr, imc, imr.shape[1]
            weave.inline(code, ['ima','imb','width','node_cnt','pb','dm','dxy','dxy_local','px','py','best','cost'],#
                support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<map>','<csignal>'],
                compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])
#        best[cost>0.1] = -1
        res = pb+best*dxy
        return res, cost

    def test_EPLMatch():
        f = plt.figure(num='epl match'); f.clear()
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')

        pref = plt.ginput(1, timeout=-1)[0]
        best0 = searchEPL(pref[0], pref[1], f0.im, f1.im, getG(f0,f1), win_width=4, debug=True)
        ar.plot(f1.px,f1.py,'b.',ms=2)
        plt.pause(0.01)


    def searchEPLMulti(px, py, imr, imc, rGc, N=5, win_width=4, debug=False):
        # px, py, imr, imc, rGc, N, win_width, debug = pref[0], pref[1], f0.im, f1.im, getG(f0,f1), 5, 4, True
        # px, py, imr, imc, rGc, N, win_width, debug = f0.px, f0.py, f0.im, f1.im, getG(f0,f1), 5, 4, False
        px,py = np.atleast_1d(px, py)

        node_cnt = len(px)
        pb,dm,dxy,dxy_local = calcEpl(px, py, rGc)
        best = np.full((node_cnt, N),-1, 'i')
        cost = np.empty_like(best, 'f')
        best_cnt = np.zeros(node_cnt,'i')

        for n in xrange(node_cnt):
            print n
            p0 = vec(px[n],py[n])

            ref_pos = p0 - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1) # TODO: correct the sign
            ref_samples = sample(imr, ref_pos[0], ref_pos[1])

            sam_cnt = np.floor(dm[n])
            if sam_cnt<1:
                best[n] = -1
                continue

            cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(-win_width, sam_cnt+win_width)
            cur_samples = sample(imc, cur_pos[0], cur_pos[1])
            err = np.empty(sam_cnt,'f')

            for i in xrange(int(sam_cnt)):
                diff = ref_samples - cur_samples[i:i+2*win_width+1]
                err[i] = (diff*diff).sum()
            extrema_id, = scipy.signal.argrelmin(err)
            extrema_id = extrema_id[np.argsort(err[extrema_id])]
            can_cnt = np.minimum(len(extrema_id), N)
            best_cnt[n] = can_cnt
            best[n,:can_cnt] = extrema_id[:can_cnt]
            cost[n,:can_cnt] = err[best[n,:can_cnt]]

        if debug:
            f = plt.figure(num='epl match multi')
            if f.axes:
                al,ar,ab = tuple(f.axes)
            else:
                gs = plt.GridSpec(2,2)
                al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
                ab = f.add_subplot(gs[1,:])
                al.imshow(imr, interpolation='none'); ar.imshow(imc, interpolation='none')
#                        pt = trueProj(ref_pos[0],ref_pos[1], cGr=inv(rGc), Zr=f0.Z)
#                        ar.plot(pt[0], pt[1],'b.')
            al.plot(ref_pos[0], ref_pos[1],'g.'); al.plot(px, py,'r.')
            result = best[best!=-1]
            pm = pb+result*dxy
            ar.plot(cur_pos[0], cur_pos[1],'g.');ar.plot(pm[0],pm[1],'r*')
            ab.plot(err)
            ab.vlines(result,0,1)
        return best,cost,best_cnt

    def test_EPLMatchMulti():
        f = plt.figure(num='epl match multi'); f.clear()
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')
        ar.plot(f1.px,f1.py,'b.',ms=2)

        pref = plt.ginput(1, timeout=-1)[0]
        best,cost,c_cnt = searchEPLMulti(pref[0], pref[1], f0.im, f1.im, getG(f0,f1), 5, win_width=4, debug=True)

        plt.pause(0.01)

    def triangulate(xr,yr,xc,yc,rGc):
        Rrc,Trc = rGc[:3,:3],rGc[:3,3]
        Baseline = np.linalg.norm(Trc)      # Edge C

        Base0 = Trc/Baseline      # epipolar: a unit vector pointed to the other camerea
        ray0 = normalize(backproject(xr,yr))
        phi0 = np.arccos(ray0.T.dot(Base0))   # Angle a

        Base1 = -Rrc.T.dot(Base0)
        ray1 = normalize(backproject(xc,yc))
        phi1 = np.arccos(ray1.T.dot(Base1))   # Angle b

        c = np.pi-phi1-phi0
        Range_r = Baseline*np.sin(phi1)/np.sin(c)     # Edge B = Edge C/sin(c)*sin(b)
        depth_r = ray0[2]*Range_r
        return depth_r




#%%
    ''' set up matching Frame'''
    refid = 0
    fs = []
    seq = [refid, 9]
    for fid in seq:
        try:
            f = Frame(frames[fid], wGc[fid],Z=Zs[fid])
        except:
            f = Frame(frames[fid], wGc[fid])
        fs.append(f)
    f1,f0 = fs[0],fs[1]

    ''' plot all image'''
    if 0:
        fig = plt.figure(num='all image'); fig.clear()
        gs = plt.GridSpec(len(seq)-1,2)
        a =  fig.add_subplot(gs[:,0])
        b = [fig.add_subplot(gs[sp,1]) for sp in range(len(seq)-1)]

        ''' base image'''
        fref = fs[0]
        a.imshow(fref.im, interpolation='none')
        a.plot(fref.px, fref.py, 'r.', ms=2)

        ''' matching images'''
        for i,sp in enumerate(b):
            fcur = fs[i+1]
            sp.imshow(fcur.im, interpolation='none')
            sp.plot(fcur.px, fcur.py, 'b.', ms=2)
#            pref = fref.ProjTo(fcur)
#            sp.plot(pref[0], pref[1], 'r.', ms=2)
        fig.tight_layout()

    if 0:
        test_calcEpl()
        test_EPLMatch()

    if 0:
        match,err = f0.searchEPL(f1, 3)
        d = triangulate(f0.px,f0.py,match[0],match[1], getG(f0,f1))
        plotxyzrgb(f0.makePC(d, 0, 5))
        if not f0.Z is None and 0:
            dt = f0.Z[f0.py,f0.px]
            plotxyzrgb(f0.makePC(dt, 0, 5),hold=True)
            err = np.abs(dt-d)

        '''filtering'''
        if 0:
            d[conditions(d>10,d<1)] = np.nan
            d_result = np.full_like(f0.im, np.nan,'f')
            d_result[f0.py, f0.px] = d
            df = scipy.ndimage.filters.generic_filter(d_result, np.nanmedian, size=15)
            plotxyzrgb(f0.makePC(df[f0.py, f0.px], 0, 5))

    if 0:
        best,cost,best_cnt = searchEPLMulti(f0.px, f0.py, f0.im, f1.im, getG(f0,f1))
        for it in range(1,7):
            if np.mod(it,2):
                p_seq = xrange(f0.p_cnt)
                nbrs = f0.nbrs[0]
            else:
                p_seq = reversed(xrange(f0.p_cnt))
                nbrs = f0.nbrs[1]
            for p_id in p_seq:
                for nbr_id in nbrs[i]:
                    pass



#%%
    @timing
    def dt(f, q=None, p=None):
        f = f.astype('f').copy()
        n = f.shape[0]

        q = q.astype('f').copy() if not q is None else np.arange(n, dtype='f')
        p = p.astype('f').copy() if not p is None else np.arange(n, dtype='f')
        m = p.shape[0]
        d = np.empty(m,'f')
        d_arg = np.empty(m,'i')

        if 0:
            v_id = np.zeros((n+1),'i')
            z = np.full((n+1),np.inf,'f')
            z[0] = -np.inf
            k = 0
            square = lambda x:x*x
            for q_id in range(1,n):
                s = ( (f[q_id]+square(q[q_id]))-(f[v_id[k]]+square(q[v_id[k]])) )/(2*q[q_id]-2*q[v_id[k]])
                while (s <= z[k]):
                    k -= 1
                    s = ( (f[q_id]+square(q[q_id]))-(f[v_id[k]]+square(q[v_id[k]])) )/(2*q[q_id]-2*q[v_id[k]])
                k += 1
                v_id[k] = q_id
                z[k] = s

            k = 0
            for p_id in nditer(np.argsort(p)):
                while z[k+1] < p[p_id]:
                    k += 1
                d[p_id] = square(p[p_id]-q[v_id[k]]) + f[v_id[k]]
                d_arg[p_id] = v_id[k]
        else:
            scode = r'''
                template <class T>
                    inline T square(const T &x) { return x*x; };

                #define INF std::numeric_limits<float>::infinity()
                void dt(float *f, float *p, int n,
                        float *q, int m,
                        float *d, int *d_arg)
                {
                    int *v_id = new int[n+1];
                    float *z = new float[n+1];
                    int k = 0;
                    v_id[0] = 0;
                    z[0] = -INF;
                    z[1] = +INF;
                    for (int q_id = 1; q_id <= n-1; q_id++) {
                        float s = 0.5*((f[q_id]+square(q[q_id]))-(f[v_id[k]]+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
                        while (s <= z[k]) {
                            k--;
                            s = 0.5*((f[q_id]+square(q[q_id]))-(f[v_id[k]]+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
                        }
                        k++;
                        v_id[k] = q_id;
                        z[k] = s;
                        z[k+1] = +INF;
                    }
                    k = 0;
                    for (int p_id = 0; p_id <= m-1; p_id++) {
                        while (z[k+1] < p[p_id])
                            k++;
                        d[p_id] = square(p[p_id]-q[v_id[k]]) + f[v_id[k]];
                        d_arg[p_id] = v_id[k];
                    }

                    delete [] v_id;
                    delete [] z;
                }'''
            code = r'''
              //std::raise(SIGINT);
              dt(&F1(0), &P1(0), n,
                 &Q1(0), m,
                 &D1(0), &D_ARG1(0));
             dt(f,p,n,q,m,d,d_arg);
            '''
            weave.inline(code,['d','d_arg','f','n','m','p','q'],
                         support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<stdio.h>','<csignal>'],
                         compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])
        return d,d_arg
    def test_dt():
       cost,best_id = dt(np.arange(10,0,-1),np.arange(10), np.arange(3)*3)