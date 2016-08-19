#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:42:37 2016

@author: kaihong
"""
from __future__ import division
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

from scipy import weave,sparse


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

def iD(depth):
    return 1.0/depth

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
        def __init__(self, img, wGc=np.eye(4), Z=None):
            self.im = np.ascontiguousarray(img.astype('f')/255.0)
            self.wGc = wGc.copy()
            self.Z = Z
            self.py, self.px = None,None

        def extractPts(self, gthreshold=None):
            ''' 1.extract pixels with significant gradients'''
            h,w = self.im.shape

#            grad,orin = scharr(self.im)
#            self.grad, self.orin = grad,orin
            dx,dy = np.gradient(self.im)
            grad = np.sqrt(dx**2 + dy**2)
            self.grad, self.orin = grad, np.arctan2(dy,dx)

            if gthreshold is None:
                gthreshold = np.percentile(grad, 80)
            u, v = np.meshgrid(range(w),range(h))
            border_width = 20
            mask = reduce(np.logical_and, [w-border_width>u, u>=border_width,
                                           h-border_width>v, v>=border_width]) #, grad>gthreshold exclude border pixels
#            mask = conditions(189>u, u>107, 293>v, v>225) #, grad>gthreshold exclude border pixels

            y,x = np.where(mask)
            self.py, self.px = y,x
            self.p_cnt = len(x)
            self.v = []
            ''' 2. corresponding back-projected 3D point'''
            self.P = np.vstack([(x-cx)/fx,
                                (y-cy)/fy,
                                np.ones(self.p_cnt)])
            ''' 3. patch pixels'''
#            patt = [(y,x),(y-2,x),(y-1,x+1),(y,x+2),(y+1,x+1),(y+2,x),(y+1,x-1),(y,x-2),(y-1,x-1)]
#            self.v = np.vstack([self.im[ind].astype('i2') for ind in patt]).T
            ''' 4. Neighbors Info'''
#            self.nbrs = self.calcNeighborsInfo(mask)
            I = self.getIncidenceMat(False)
            D = (I.transpose().dot(I)).diagonal()
            self.trimPts(D!=0)


        def trimPts(self, mask, d=None):
            var_list = (self.px, self.py, self.P)
            self.px, self.py, self.P = (np.compress(mask,dump,axis=-1) for dump in var_list)
            self.p_cnt = self.px.shape[0]

            if not d is None:
                return d[mask]

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

        def searchEPL(self, f1, dmin=None,dmax=None,win_width=4):
            if self.px is None:
                self.extractPts()
            rGc = relPos(self.wGc, f1.wGc)
            d,var = searchEPL(self.px, self.py, self.im, f1.im, rGc, dmin, dmax, win_width, False)
            self.v = searchEPL.vlist
            return d,var

        def makePC(self, depth, vmin=0, vmax=10):
            vm_ind, = np.where(conditions(depth>vmin,  depth<vmax))
            p3d = self.P*depth
            I = np.tile(self.im[self.py, self.px]*255,(3,1))
            P = np.vstack([p3d, I]).T
            return P[vm_ind,:]      # Nx6

        def getPtsMask(self):
            mask_im = np.full_like(self.im, False,'bool')
            mask_im[self.py, self.px] = True
            return mask_im

        def calcNeighborsInfo(self):
            mask_image = self.getPtsMask()
            px, py = self.px, self.py
            node_cnt = self.p_cnt

            edges_forward = [[] for _ in range(node_cnt)]
            edges_backward = [[] for _ in range(node_cnt)]

            id_LUT = np.empty_like(mask_image, 'i4')
            id_LUT[py,px] = range(node_cnt)      # lookup-table of index number for valid pixels
            for p_id, (p_x,p_y) in enumerate(zip(px, py)):
                fcoord = [(p_y-1,p_x),(p_y,p_x-1)] #,(p_y-1,p_x-1),(p_y-1,p_x+1)
                fnbrs = [id_LUT[coord] for coord in fcoord if mask_image[coord]]
                if p_id-1 not in fnbrs:
                    fnbrs.append(p_id-1)
                edges_forward[p_id].extend(fnbrs)

                bcoord = [(p_y+1,p_x),(p_y,p_x+1)]#,(p_y+1,p_x+1),(p_y+1,p_x-1)
                bnbrs = [id_LUT[coord] for coord in bcoord if mask_image[coord]]
                if p_id+1 not in bnbrs:
                    fnbrs.append(p_id+1)
                edges_backward[p_id].extend(bnbrs)
            return edges_forward, edges_backward

        def getIncidenceMat(self, makelist=False):
            mask_im = self.getPtsMask()

            node_cnt = self.p_cnt
            id_LUT = np.empty_like(mask_im, 'i4')
            id_LUT[mask_im] = range(node_cnt)      # lookup-table of index number for valid pixels

            edges = []
            nbr_list = [[] for _ in xrange(node_cnt)]
            nbr_cnt = np.zeros(node_cnt,'i')
            for p_id in range(node_cnt):
                p_x, p_y = self.px[p_id], self.py[p_id]
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

            incidence_matrix = sparse.csr_matrix((data,(row_ind,col_ind)), (len(edges),node_cnt),'i4') # each row represent an edge
            if makelist:
                enode_out = np.array(zip(*edges)[0])
                return incidence_matrix, nbr_list, nbr_cnt, enode_out
            else:
                return incidence_matrix

        def getId(self, x,y):
            return np.where(conditions(self.px==x,self.py==y))[0]

    def getG(f0,f1):
        '''return 1G0, which p1 = 1G0 * p0  '''
        return np.dot(inv(f0.wGc), f1.wGc)

    def calcF(rGc, K=K):
        ''' xr'*F*xc = 0 '''
        R,t = rGc[:3,:3],rGc[:3,3]
        rFc = inv(K.T).dot(skew(t)).dot(R).dot(inv(K))
        return rFc

    class EpilineCalculator(object):
        def __init__(self, xr, yr, rGc, K):
            ''' 1. suppose (X, X') are the left and right image pixel pairs,
                given the relative camera pos (R,T), X'=(R,T).dot(X), we have a ray X'∈R3:
                    X' = K*(R*inv(K)*X*z + T)
                       = K*R*inv(K)*X + 1/z*K*T
                       =    Pinf[1:3] +   λ*Pe[1:3]  (λ:=1/z)
                The projected image point x' of X' will be:
                    x' = (Pinf[1:2] + λ*Pe[1:2])/(Pinf[3]+ λ*Pe[3])     (eq.1)
                but what we want is x' in this form:
                    x' =  Pinf[1:2]/Pinf[3] + λ*dxy[1:2]                (eq.2a)
                       =  Pinf[1:2]/Pinf[3] + v*normalize(dxy[1:2])     (eq.2b)
                putting eq.1 & eq.2a together and solve for dxy, we get:
                      dxy[1:2] = 1/(Pinf[3]+ λ*Pe[3]) * (-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2])   (eq.3)
                so normalize(dxy) = normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]),  if (Pinf[3]+ λ*Pe[3])>0,
                                  = -normalize(-Pe[3]/Pinf[3]*Pinf[1:2] + Pe[1:2]), otherwise.
                2.for (Pinf[3]+ λ*Pe[3])>0, i.e. point is in front of the cur camera, we have:
                    if Pe[3]>0,  λ > 0 > -Pinf[3]/Pe[3], then λ_max=inf;
                    if Pe[3]<0,  -Pinf[3]/Pe[3] > λ > 0, then λ_max=-Pinf[3]/Pe[3].

                3. Also from eq.2 we have:
                    λ*dxy[1:2] = v*normalize(dxy[1:2]),
                if λ is given, then
                    λ/(Pinf[3] + λ*Pe[3])*dxy_norm = v,             (eq.4)
                equivalently, if v is given, then
                    λ = a*Pinf[3]/(1- a*Pe[3]),  a:=v/dxy_norm   (eq.5)

                4. if x' is given, from eq.1:
                    x'[1:2]*Pe[3]*λ - Pe[1:2]*λ = Pinf[1:2] - x'[1:2]*Pinf[3]
                    λ = (Pinf[1:2] - x'[1:2]*Pinf[3])/(x'[1:2]*Pe[3] - Pe[1:2])
            '''
            xr,yr = np.atleast_1d(xr,yr)

            Rrc,Trc = rGc[:3,:3],rGc[:3,3]
            Rcr,Tcr = Rrc.T, -Rrc.T.dot(Trc)

            Pr = projective(xr, yr)                                 # 3xN
            Pe0 = vec(K.dot(Trc))                                   # 3x1
            dxy_local = normalize(-Pe0[2]/Pr[2]*Pr[:2]+Pe0[:2])     # 2xN

            Pinf = (K.dot(Rcr.dot(inv(K)))).dot(Pr)
            nPinf = Pinf[:2]/Pinf[2]
            Pe1 = vec(K.dot(Tcr))

            dxy_raw = -Pe1[2]/Pinf[2]*Pinf[:2]+Pe1[:2]              # 2xN
            dxy_norm = np.linalg.norm(dxy_raw, axis=0)              # N
            dxy = dxy_raw/dxy_norm                                  # 2xN

            self.VfromD = lambda      d,ind=slice(None): d/(Pinf[2,ind] + d*Pe1[2])*dxy_norm[ind]
            self.VfromX = lambda     xc,ind=slice(None): (xc - nPinf[0,ind])/dxy[0,ind]
            self.VfromY = lambda     yc,ind=slice(None): (yc - nPinf[1,ind])/dxy[1,ind]

            self.DfromV = lambda      v,ind=slice(None): v*Pinf[2,ind]/(dxy_norm[ind] - v*Pe1[2])
            self.DfromX = lambda     xc,ind=slice(None): (Pinf[0,ind] - xc*Pinf[2,ind])/(xc*Pe1[2]-Pe1[0])
            self.DfromY = lambda     yc,ind=slice(None): (Pinf[1,ind] - yc*Pinf[2,ind])/(yc*Pe1[2]-Pe1[1])

            self.XYfromV = lambda     v,ind=slice(None): ( nPinf[0,ind] + v*dxy[0,ind], nPinf[1,ind] + v*dxy[1,ind])
            self.XYfromD = lambda     d,ind=slice(None): ((Pinf[0,ind]+d*Pe1[0])/(Pinf[2,ind]+d*Pe1[2]), \
                                                          (Pinf[1,ind]+d*Pe1[1])/(Pinf[2,ind]+d*Pe1[2]))
            self.XYfromV_local=lambda v,ind=slice(None): (Pr[0,ind]+v*dxy_local[0,ind], Pr[1,ind]+v*dxy_local[1,ind])

            def getDlimt():
                limit0 = -Pinf[2]/Pe1[2] if Pe1[2]<0 else np.inf
                limit1 = np.maximum(self.DfromX(5), self.DfromX(w-5))   # border is extended a little bit, i.e. 5 pixels
                limit2 = np.maximum(self.DfromY(5), self.DfromY(h-5))
                return reduce(np.minimum, [limit0, limit1, limit2])
            self.getDlimt = getDlimt

            self.dxy = dxy
            self.dxy_local = dxy_local
            self.nPinf = nPinf


    def test_EpilineCalculator():
        f,a = plt.subplots(1, 1, num='test_EpilineCalculator')
        a.imshow(sim(f0.im, f1.im))

        pref = np.round(plt.ginput(1, timeout=-1)[0])
        a.plot(pref[0], pref[1],'r.')

        cGr = getG(f1,f0)
        Z = np.linspace(0.5, 10.0, 40)
        pcur = K.dot(transform(cGr, backproject(pref[0], pref[1])*Z))
        pcur /= pcur[2]
        a.plot(pcur[0]+640, pcur[1],'b.')

        ec = EpilineCalculator(pref[0], pref[1], getG(f0,f1), K) #
        pmin = ec.XYfromV(0)
        pmax = ec.XYfromD(ec.getDlimt())
        a.plot([pmin[0]+640,pmax[0]+640], [pmin[1],pmax[1]],'g-')

        plt.pause(0.01)

        ec2 = EpilineCalculator(f0.px, f0.py, getG(f0,f1), K) #
        tx,ty = trueProj(f0.px, f0.py, cGr, Zr=f0.Z)
        td = 1.0/sample(f0.Z, f0.px, f0.py)
        d = ec2.DfromX(tx); assert( np.allclose(td, d) )
        v = ec2.VfromD(td); assert( np.allclose(v, ec2.VfromX(tx)) )
        xy = ec2.XYfromD(td); assert( np.allclose(xy[0], tx) and np.allclose(xy[1], ty))

    @timing
    def searchEPL(px, py, imr, imc, rGc, dmin=0, dmax=np.inf, win_width=4, debug=False):
        # px, py, imr, imc, rGc, win_width, debug, dmin, dmax= pref[0], pref[1], f0.im, f1.im, getG(f0,f1), 4, True, None,None
        px,py = np.atleast_1d(px, py)

        ec = EpilineCalculator(px, py, rGc, K)
        dmax = np.minimum(dmax, ec.getDlimt())                  # Nx1
        vmax = np.floor(ec.VfromD(dmax)).astype('i4')           # Nx1
        vmin = np.ceil( ec.VfromD(dmin)).astype('i4')           # Nx1
        pb,dxy,dxy_local = ec.nPinf, ec.dxy, ec.dxy_local
        var = (dmin-dmax)/(vmax-vmin)

        node_cnt = len(px)
        best = np.empty(node_cnt,'i')
        searchEPL.rlist, searchEPL.vlist,searchEPL.ec = [],[],ec

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
            }
            inline void convolution(const std::vector<float> &ref,
                                const std::vector<float> &cur,
                                float *out, int &argmin)
            {
                float min_diff = std::numeric_limits<float>::infinity();

                for(size_t i=0; i <= cur.size()-ref.size(); i++ ){
                    float diff = 0;
                    for(size_t j=0; j < ref.size();j++ ){
                        float err = ref[j] - cur[i+j];
                        diff += err*err;
                    }
                    out[i] = diff;

                    if (diff < min_diff){
                        min_diff = diff;
                        argmin = i;
                    }
                }
            }
            '''
        ima,imb,width = imr, imc, imr.shape[1]
        if 1 or debug:
            code = r'''
                const int win_width = %(win_width)d;
                const int win_size = 2*win_width+1;
                std::vector<float> ref_samples(win_size);
                std::vector<float> cur_samples(1000);

                // std::raise(SIGINT);

                /* 1. Sampling the Intensity */
                cur_samples.resize(sam_cnt+2*win_width);

                for(int i=0; i<sam_cnt+2*win_width; i++)
                    cur_samples[i] = sample(&IMB1(0), width,
                                            PB2(0,p_id)+(sam_min+i-win_width)*DXY2(0,p_id),
                                            PB2(1,p_id)+(sam_min+i-win_width)*DXY2(1,p_id));

                for(int pos=0; pos<win_size; pos++)
                    ref_samples[pos] = sample(&IMA1(0),width,
                                            PX1(p_id)-(pos-win_width)*DXY_LOCAL2(0,p_id),
                                            PY1(p_id)-(pos-win_width)*DXY_LOCAL2(1,p_id));

                /* 2. go through all the points in cur */
                convolution(ref_samples, cur_samples, &ERR1(0), BEST1(p_id));
                BEST1(p_id) += sam_min;
                '''% {'win_width': win_width }

            for p_id in xrange(node_cnt):
                sam_min,sam_max = int(vmin[p_id]), int(vmax[p_id])
                sam_cnt = int(sam_max-sam_min+1)
                err = np.empty(sam_cnt, 'f')

#                if 0:
#                    p0 = vec(px[p_id],py[p_id])
#                    ref_pos = p0 - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1)
#                    ref_samples = sample(imr, ref_pos[0], ref_pos[1])
#                    cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(sam_min-win_width, sam_max+win_width+1)
#                    cur_samples = sample(imc, cur_pos[0], cur_pos[1])
#
#                    for i in xrange(sam_cnt):
#                        diff = ref_samples - cur_samples[i:i+2*win_width+1]
#                        err[i] = (diff*diff).sum()
#                    mininum = np.nanargmin(err)
#                    best[n] = sam_min+mininum  #np.argpartition(err, 5)
#                    cost[n] = err[mininum]
#                else:
                weave.inline(code, ['ima','imb','width',
                                    'p_id','sam_min','sam_max','sam_cnt',
                                    'pb','dxy','dxy_local','px','py',
                                    'best','err'],#
                            support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<map>','<csignal>','<set>'],
                            compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])

                searchEPL.vlist.append(err)
                searchEPL.rlist.append((sam_min,sam_max))

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
                    p0 = vec(px[p_id],py[p_id])
                    ref_pos = p0 - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1)
                    cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(sam_min-win_width, sam_max+win_width+1)

                    al.plot(ref_pos[0], ref_pos[1],'g.'); al.plot(px, py,'r.')
                    pm = pb+best*dxy
                    ar.plot(cur_pos[0], cur_pos[1],'g.');ar.plot(pm[0],pm[1],'ro');
                    ab.plot(err)
                    ab.vlines(best[p_id]-sam_min,0,1)

#                    if not f0.Z is None:
#                        pt = trueProj(ref_pos[0],ref_pos[1], cGr=inv(rGc), Zr=f0.Z)
#                        ar.plot(pt[0], pt[1],'b.')
#                        ab.vlines((pt[0][win_width]-pb[0])/dxy[0]-sam_min,0,1,'g')
        else:
            code = r'''
            size_t M = node_cnt;
            const int win_width = %(win_width)d;
            const int win_size = 2*win_width+1;
            std::vector<float> ref_samples(win_size);
            std::vector<float> cur_samples(1000);
            std::vector<float> err(1000);

            // std::raise(SIGINT);

            // foreach pixel in ref image
            for(size_t p_id=0; p_id<M; p_id++){

                /* 1. Sampling the Intensity */
                int sam_min = std::floor(VMIN1(p_id));
                int sam_max = std::ceil(VMAX1(p_id));
                int sample_size = sam_max-sam_min+1;
                /*if(sample_size<1){
                    BEST1(p_id) = -1;
                    COST1(p_id) = std::numeric_limits<float>::infinity();
                    continue;   // discard pixel whose epi-line length is 0

                }*/
                cur_samples.resize(sample_size+2*win_width);
                err.resize(sample_size);
                for(int i=0; i<sample_size+2*win_width; i++)
                    cur_samples[i] = sample(&IMB1(0), width,
                                            PB2(0,p_id)+(sam_min+i-win_width)*DXY2(0,p_id),
                                            PB2(1,p_id)+(sam_min+i-win_width)*DXY2(1,p_id));

                for(int pos=0; pos<win_size; pos++)
                    ref_samples[pos] = sample(&IMA1(0),width,
                                            PX1(p_id)-(pos-win_width)*DXY_LOCAL2(0,p_id),
                                            PY1(p_id)-(pos-win_width)*DXY_LOCAL2(1,p_id));

                /* 2. go through all the points in cur */
                convolution(ref_samples, cur_samples, &err[0], BEST1(p_id), COST1(p_id));
                BEST1(p_id) += sam_min;

                /* 3. find the best N element */
                #if 0
                    auto result = std::min_element(std::begin(err), std::end(err));
                    BEST1(p_id) = sam_min + std::distance(std::begin(err), result);
                    COST1(p_id) = *result;
                #endif
            }
            '''% {'win_width': win_width }

            weave.inline(code, ['ima','imb','width','node_cnt','pb','vmax','vmin','dxy','dxy_local','px','py','best','cost','vlist'],#
                support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<map>','<csignal>','<set>'],
                compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])

        res = ec.DfromV(best)
        return res, var**2

    def test_EPLMatch():
        f = plt.figure(num='epl match'); f.clear()
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')

        pref = plt.ginput(1, timeout=-1)[0]
        best0,var = searchEPL(pref[0], pref[1], f0.im, f1.im, getG(f0,f1), iD(10),iD(0.1), win_width=4, debug=True)
#        ar.plot(f1.px,f1.py,'b.',ms=2)
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




#%% main
    ''' set up matching Frame'''
    refid = 0

    fs = []
    seq = [refid, 9, 1, 2, 3, 4,5,6,7,8]#
    for fid in seq:
        try:
            f = Frame(frames[fid], wGc[fid],Z=Zs[fid])
        except:
            f = Frame(frames[fid], wGc[fid])
        fs.append(f)
    f0,f1 = fs[0],fs[1]

    if 0:
        test_calcEpl()
        test_EPLMatch()

    if 1:
        ds,vs,data,dr = [],[],[[] for _ in range(len(fs))],[[] for _ in range(len(fs))]
        for i in range(1,len(fs)):
            d,var = f0.searchEPL(fs[i], dmin=iD(5), dmax=iD(2), win_width=3) #
            data[i].extend(searchEPL.vlist)
            dr[i].extend(searchEPL.rlist)
            ds.append(d)
            vs.append(var)

        def mergeD(d0, var0, d1, var1):
            var_sum = var0+var1
            d = var1/var_sum*d0 + var0/var_sum*d1
            var = var1*var0/var_sum
            return d,var

        p0 = 37838
        d,var = ds[0],vs[0]
        print vec([v[p0] for v in vs ])     # variance of each match of p0
        for d_,var_ in reversed(zip(ds[1:],vs[1:])):
            d,var = mergeD(d, var, d_, var_)
            print var[p0]
            plotxyzrgb(f0.makePC(1.0/d))
            plt.waitforbuttonpress()
        [plt.plot(np.linspace(0,1,len(obj[p0])),obj[p0],'o-') for obj in data[1:] ]


        if not f0.Z is None and 0:
            dt = f0.Z[f0.py,f0.px]
            plotxyzrgb(f0.makePC(dt, 0, 10),hold=True)
            err = np.abs(dt-1.0/d)
            emask = (1.0/d)<2
            pf('emask');
            pis(f0.im)
            plt.plot(f0.px,f0.py,'b.',ms=2)
            plt.plot(f0.px[emask],f0.py[emask],'r.',ms=2)
            plt.colorbar()

        '''filtering'''
        if 0:
            d[conditions(d>10,d<1)] = np.nan
            d_result = np.full_like(f0.im, np.nan,'f')
            d_result[f0.py, f0.px] = d
            df = scipy.ndimage.filters.generic_filter(d_result, np.nanmedian, size=15)
            plotxyzrgb(f0.makePC(1.0/df[f0.py, f0.px], 0, 5))

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
        d,var = f0.searchEPL(f1, dmin=iD(5), dmax=iD(0.1), win_width=3) #
        I, nbr_list, nbrs_cnt, enode_out = f0.getIncidenceMat(True)
        L = I.transpose().dot(I)
        D = L.diagonal()
        A = scipy.sparse.diags(D) - L

        plotxyzrgb(f0.makePC(1.0/d,-np.inf,np.inf))
        '''setup neighbor LUT'''
        import pandas as pd
        nbrs = pd.DataFrame(nbr_list, dtype=int).fillna(-1).values

        node_edge = np.abs(I).T

#%%
        rec =[]

        def evalE_l1(x,y,Lambda):
            return np.sqrt(node_edge.dot(I.dot(x)**2)).sum() + Lambda*np.sum((x-y)**2)

        @timing
        def solveFixIterl1c(x, y, Lambda, I=I, nbrs=nbrs, nbrs_cnt=nbrs_cnt, it_num=20):
            code = r'''
                size_t N = Nx[0];
                // 1. Get the gradient
                for(size_t pid=0; pid<N; pid++){
                    float coef_sum = 2.0*Lambda;
                    float value_sum = 2.0*Lambda*Y1(pid);

                    size_t M = NBRS_CNT1(pid);
                    for(size_t m=0; m<M; m++){
                        size_t mid = NBRS2(pid, m);
                        float Ruv = NGRAD1(pid) + NGRAD1(mid);
                        coef_sum += Ruv;
                        value_sum += Ruv * X_1(mid);
                    }
                    X1(pid) = value_sum/coef_sum;
                }'''

            for it in range(it_num):
                eFlow = I.dot(x)        # Flow on edges = ▽x
                nGrad = np.sqrt(node_edge.dot(eFlow**2))     # norm(▽x) of each nodes
                nGrad = 1.0/(nGrad + 1e-4)
                x_ = x.copy()
                weave.inline(code, ['nbrs','nbrs_cnt','nGrad','y','x','x_','Lambda'],
                             extra_compile_args=['-std=gnu++11 -msse2 -O3'])
                print evalE_l1(x, y, Lambda)
            return x

        @timing
        def solvePDl1(x0, f, Lambda, it_num=10): #A=A, D=D, I=I ,
            tau,sigma = 1.0/D, 0.5

            y = np.zeros(I.shape[0])
            x = x0.copy()
            x_ = x0.copy()

            for it in range(it_num):
                y += sigma*I.dot(x_)
                nGrad = np.maximum(1, np.sqrt(node_edge.dot(y**2)))
                y /= nGrad[enode_out]   # prox_f

                x_ = x.copy()
                x -= tau*I.T.dot(y)
                x = (x + Lambda*tau*f)/(1.0+Lambda*tau)# prox_g
                x_ = 2*x - x_
                print evalE_l1(x, f, Lambda)
            return x

        debug = False
        p0, = np.where(conditions(f0.px==161,f0.py==281)) #391 220
        p1 = p0+1

        @timing
        def searchC(f0, Lambda, theta, d, cGr=getG(f1, f0)):
            xr,yr = f0.px, f0.py
            Rcr,Tcr = cGr[:3,:3],cGr[:3,3]

            H = K.dot(Rcr.dot(inv(K)))
            Pinf = H.dot(projective(xr, yr))  # Pinf=K.dot(Rcr.dot(backproject(xr, yr))
            Pe = vec(K.dot(Tcr))

            dxy_raw = -Pe[2]/Pinf[2]*Pinf[:2]+Pe[:2]   # 2xN
            dxy_norm = np.linalg.norm(dxy_raw, axis=0)   # N

            x = d/(Pinf[2] + d*Pe[2])*dxy_norm
            y = np.empty_like(x)
            if debug:
                fig,a = plt.subplots(1,1,num='searchC')
                cset = [a.plot(f0.v[p0])[0] for i in range(2)]
                lset = [a.axvline(obj,0,2, c=color) for obj,color in zip([x[0],x[1]],'bg')]

            Ed = 0
            Cost = f0.v
            for pid in xrange(f0.p_cnt):
                m = Cost[pid].shape[0]
                rm = np.arange(m)
                cost = Lambda*Cost[pid] + ((rm-x[pid])/m)**2/theta

                y[pid] = np.nanargmin(cost)
                Ed += Cost[pid][y[pid]]
                if debug and pid==p0:
#                    print x[pid], y[pid]
                    cset[0].set_data(rm, cost)
                    cset[1].set_data(rm, Cost[pid])
                    lset[0].set_xdata(y[pid])
                    lset[1].set_xdata(np.nanargmin(Cost[pid]))
                    plt.pause(0.01)
                    plt.waitforbuttonpress()

            a = y/dxy_norm
            dinv = a*Pinf[2]/(1-a*Pe[2])   # λ/(Pinf[2] + λPe1[2])*dxy_norm = v
            print Lambda*Ed, np.abs(I.dot(x)).sum()
            return dinv
#%%

        beta = 1e-2
        theta_init,theta_end = 1e4,1e-2
        theta = theta_init
        n = 0
        x = d.copy()
        y = x.copy()
        rec =[]

        Lambda=100.0
        solver = [solvePDl1, solveFixIterl1c][0]
        while theta > theta_end:
            x = solver(x, y, Lambda=1,it_num=10);plotxyzrgb(f0.makePC(1.0/x,-np.inf,np.inf))
            y = searchC(f0, Lambda, theta, x)

            theta = theta*(1.0-beta*n)
            n += 1
            print n


#            plt.waitforbuttonpress()
#        plot(rec)
#        plotxyzrgb(f0.makePC(1.0/d,-np.inf,np.inf))




















#%% distance transform
    @timing
    def dt(f, q=None, p=None, Lambda=1.0):
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
                s = ( (f[q_id]/Lambda+square(q[q_id]))-(f[v_id[k]]/Lambda+square(q[v_id[k]])) )/(2*q[q_id]-2*q[v_id[k]])
                while (s <= z[k]):
                    k -= 1
                    s = ( (f[q_id]/Lambda+square(q[q_id]))-(f[v_id[k]]/Lambda+square(q[v_id[k]])) )/(2*q[q_id]-2*q[v_id[k]])
                k += 1
                v_id[k] = q_id
                z[k] = s

            k = 0
            for p_id in nditer(np.argsort(p)):
                while z[k+1] < p[p_id]:
                    k += 1
                d[p_id] = Lambda*square(p[p_id]-q[v_id[k]]) + f[v_id[k]]
                d_arg[p_id] = v_id[k]
        else:
            scode = r'''
                template <class T>
                    inline T square(const T &x) { return x*x; };

                #define INF std::numeric_limits<float>::infinity()
                void dt(float *f, float Lambda, float *p, int n,
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
                        float s = 0.5*((f[q_id]/Lambda+square(q[q_id]))-(f[v_id[k]]/Lambda+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
                        while (s <= z[k]) {
                            k--;
                            s = 0.5*((f[q_id]/Lambda+square(q[q_id]))-(f[v_id[k]]/Lambda+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
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
                        d[p_id] = Lambda*square(p[p_id]-q[v_id[k]]) + f[v_id[k]];
                        d_arg[p_id] = v_id[k];
                    }

                    delete [] v_id;
                    delete [] z;
                }'''
            code = r'''
              //std::raise(SIGINT);
              /*dt(&F1(0), Lambda, &P1(0), n,
                 &Q1(0), m,
                 &D1(0), &D_ARG1(0));*/
             dt(f,Lambda,p,n,q,m,d,d_arg);
            '''
            weave.inline(code,['d','d_arg','f','n','m','p','q','Lambda'],
                         support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<stdio.h>','<csignal>'],
                         compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])
        return d,d_arg
    def test_dt():
        a = np.array(range(5,0,-1)+range(6))*5
        cost,best_id = dt(a)
        plt.plot(a)
        plt.plot(cost)
        [plt.plot( a+(np.arange(11)-i)**2, ':') for i in range(11)]

        cost,best_id = dt(np.arange(10,0,-1),np.arange(10), np.arange(3)*3)

    if 0:

        N = 30
        C = (np.sin(2*np.pi*np.arange(N, dtype='f')/N + vec(2,-3,4))+1)*10
        cost,best = map(np.asarray, zip(*[dt(C[i]) for i in range(3)]))
        f,a = plt.subplots(1,1,num='cost');a.clear()
        a.plot(C[0])
        a.plot(cost[0])
        [a.plot( C[0]+(np.arange(N)-i)**2, ':') for i in range(N)]

        L = np.array([[1,-1,0],[0,1,-1],[-1,0,1]],'f') #
        LtL = L.T.dot(L)
        def prox_f2(x, Lambda=1.0):
            return Lambda/(Lambda+1.0)*x
        def prox_fl2(x, Lambda=1.0):
            v = np.sqrt(np.sum(x**2))
            return np.where(v<Lambda, np.zeros_like(x), (1-Lambda/v)*x)
        def prox_g(x, Lambda=1):
            return np.argmin((vec(x)-np.arange(N))**2*(0.5/Lambda) + C,axis=1)
#            return best[range(3), np.clip(x,0,N-1).astype('i')]

        x = np.array([25,5,0],'f')#np.argmin(C, axis=1)#
        z = x.copy()
        u = x - z

        f,a = plt.subplots(1,1,num='ADMM');a.clear()
        a.plot(C.T)
        a.set_xlim(-1,30)
        l1,l2,l3 = (a.axvline(foo,0,2,c=color) for foo,color in zip(vec(x),['b','g','r']))

        for it in range(30):
            [foo.set_xdata(bar) for foo,bar in zip([l1,l2,l3],vec(x))]
            print np.sqrt(np.sum(L.dot(x)**2))+C[range(3),x.astype('i')].sum()
            print x,z,u

            plt.pause(0.01)
            plt.waitforbuttonpress()

            x = prox_g(x-L.T.dot(L.dot(x)-z+u))
            z = prox_fl2(L.dot(x) + u)
            u = u+L.dot(x)-z

