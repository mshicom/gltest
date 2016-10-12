#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:42:37 2016

@author: kaihong
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.ndimage
import scipy.io
from vtk_visualizer import plotxyzrgb,plotxyz

from scipy import weave,sparse
from tools import *
from EpilineCalculator import EpilineCalculator,EpilineDrawer



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


import scipy.signal
import cv2
class Frame(object):
    __slots__ = ['im', '_pyr', 'wGc', 'dx','dy',     \
                 'px','py','P',     \
                 'nbrs','v', 'Z']
    def __init__(self, img, wGc=np.eye(4), Z=None):
        self.im = np.ascontiguousarray(img.astype('f')/255.0)
        self.wGc = wGc.copy()
        self.Z = Z
        self.py, self.px = None,None
        self._pyr = {0:self.im}

    def pyr_im(self, layer):
        layer = np.maximum(layer,0)
        if layer not in self._pyr:
            self._pyr[layer] = cv2.pyrDown(self.pyr_im(layer-1))
        return self._pyr[layer]

    def extractPts(self, K, gthreshold=None, pyr_level=0):
        ''' 1.extract pixels with significant gradients'''
        im = self.pyr_im(pyr_level)
        h,w = im.shape

        grad,orin = scharr(im)
#            self.grad, self.orin = grad,orin
#        dx,dy = np.gradient(self.im)
#        grad = np.sqrt(dx**2 + dy**2)
#        self.grad, self.orin = grad, np.arctan2(dy,dx)

        if gthreshold is None:
            gthreshold = np.percentile(grad, 80)
        u, v = np.meshgrid(range(w),range(h))
        border_width = 20
        mask = reduce(np.logical_and, [w-border_width>u, u>=border_width,
                                       h-border_width>v, v>=border_width, grad>gthreshold]) # exclude border pixels
#            mask = conditions(189>u, u>107, 293>v, v>225) #, grad>gthreshold exclude border pixels

        y,x = np.where(mask)
        self.py, self.px = y,x
        self.v = []
#        self.dx, self.dy = dx[y,x], dy[y,x]
        ''' 2. corresponding back-projected 3D point'''
        self.P = backproject(x, y, K)

        ''' 3. patch pixels'''
#            patt = [(y,x),(y-2,x),(y-1,x+1),(y,x+2),(y+1,x+1),(y+2,x),(y+1,x-1),(y,x-2),(y-1,x-1)]
#            self.v = np.vstack([self.im[ind].astype('i2') for ind in patt]).T
        ''' 4. Neighbors Info'''
#            self.nbrs = self.calcNeighborsInfo(mask)
        if 0:
            I = self.getIncidenceMat(False)
            D = (I.transpose().dot(I)).diagonal()
            self.trimPts(D!=0)
        return gthreshold


    def trimPts(self, mask, d=None):
        var_list = (self.px, self.py, self.P)
        self.px, self.py, self.P = (np.compress(mask,dump,axis=-1) for dump in var_list)

        if not d is None:
            return d[mask]

    def calcPtsAngle(self, M):
        p0 = self.wGc[0:3,0:3].dot(self.P)
        p = M.dot(p0)
        theta = np.arctan2(p[1], p[0])
        phi = np.arctan2(np.sqrt(p[0]**2 + p[1]**2), p[2])
        return theta, phi

    def projTo(self, f, K, d=None, mask=None):
        assert(isinstance(f, Frame))
        G = getG(f, self)
        if d is None:
            assert(self.Z is not None)
            d = self.Z[self.py, self.px]
        P0 = self.P*d
        P1 = K.dot(G[0:3,0:3].dot(P0)+G[0:3,3][:,np.newaxis])
        if mask is None:
            return P1[0]/P1[2], P1[1]/P1[2]
        else:
            return (P1[0]/P1[2])[mask], (P1[1]/P1[2])[mask]

    def makePC(self, depth, valid_mask):#
        vm_ind, = np.where(valid_mask)
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
        node_cnt = len(px)

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

        node_cnt = len(self.px)
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

    def getId(self, x=None,y=None):
        global p0
        while x is None:
            plt.figure(num='pick a point');
            plt.imshow(self.im)
            plt.plot(self.px,self.py,'b.',ms=2)
            (x,y), = np.round(plt.ginput(1, timeout=0))
            p = np.where(conditions(self.px==x,self.py==y))[0]
            if not p:
                plt.title("point not valid, pick another")
                x = None
        p0 = int(p)
        print x,y,p
        return p

    @timing
    def searchEPL(self, f1, K, dmin=0.0, dmax=1e6, win_width=4, levels=5, keepErr=False):
        if self.px is None:
            self.extractPts(K)
        cGr = getG(f1, self)
        p_cnt = len(self.px)

        dmin = np.full(p_cnt, dmin,'f') if np.isscalar(dmin) else dmin
        dmax = np.full(p_cnt, dmax,'f') if np.isscalar(dmax) else dmax

        d_min, d_max = dmin.copy(), dmax.copy()
        for level in reversed(range(levels)):
            factor = 0.5**level
            px = self.px*factor
            py = self.py*factor
            K_ = np.diag([factor,factor,1.0]).dot(K)

            imr = self.pyr_im(level)
            imc = f1.pyr_im(level)
            d,vm,var = searchEPL(px, py, imr, imc, cGr, K_, d_min, d_max, win_width, keepErr)

            if level>0:
                r_list, ec = searchEPL.rlist, searchEPL.ec
                p_id, = np.where(vm)
                d_min[p_id] = np.maximum(ec.DfromV(r_list[0][p_id], p_id), dmin[p_id])
                d_max[p_id] = np.minimum(ec.DfromV(r_list[1][p_id], p_id), dmax[p_id])

#                print d_min[p_id], d[p_id] ,d_max[p_id]
#                print searchEPL.vlist
        return d,vm,var

    @timing
    def searchEPLs(self, fs, K, dmin=0.0, dmax=1e6, win_width=4):
        if self.px is None:
            self.extractPts(K)
        p_cnt = len(self.px)

        dmin = np.full(p_cnt, dmin,'f') if np.isscalar(dmin) else dmin
        dmax = np.full(p_cnt, dmax,'f') if np.isscalar(dmax) else dmax
        d_min, d_max = dmin.copy(), dmax.copy()

        cGrs = [getG(f, self) for f in fs]
        bl = np.array([np.linalg.norm(G[:3,3]) for G in cGrs])
        odr = np.argsort(bl)
        res = []
        for i in odr:
            d,vm,err = searchEPL(self.px, self.py,
                                 self.pyr_im(0), fs[i].pyr_im(0),
                                 cGrs[i], K,
                                 d_min, d_max, win_width)

            r_list, ec = searchEPL.rlist, searchEPL.ec
            p_id, = np.where(vm)
            d_min[p_id] = np.maximum(ec.DfromV(r_list[0][p_id], p_id), dmin[p_id])
            d_max[p_id] = np.minimum(ec.DfromV(r_list[1][p_id], p_id), dmax[p_id])
            res.append((d.copy(),vm.copy(),d_max-d_min))

        return res

def getG(f0,f1):
    '''return 1G0, which p1 = 1G0 * p0  '''
    return np.dot(inv(f0.wGc), f1.wGc)

#@timing
def searchEPL(px, py, imr, imc, cGr, K, dmin=0, dmax=1e6, win_width=4, keepErr=False):
    # px, py, imr, imc, cGr, win_width, debug, dmin, dmax= pref[0], pref[1], f0.im, f1.im, getG(f1,f0), 4, True, None,None
    px,py = np.atleast_1d(px, py)

    ec = EpilineCalculator(px, py, cGr, K)

    vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(imr.shape, dmin, dmax)

    pb,dxy,dxy_local = ec.nPinf, ec.dxy, ec.dxy_local
    var = (d_max-d_min)/(vmax-vmin)

    '''border check'''
    h,w = imr.shape
    ppx = px + dxy_local[0]*vec(-win_width, win_width)
    ppy = py + dxy_local[1]*vec(-win_width, win_width)
    valid_mask = conditions(valid_mask,
                            ppx[0]>=0, ppx[0]<=w, ppx[1]>=0, ppx[1]<=w,
                            ppy[0]>=0, ppy[0]<=h, ppy[1]>=0, ppy[1]<=h)

    node_cnt = len(px)
    errs = []
    best = np.empty(node_cnt,'i')
    best_err = np.empty(node_cnt,'f')
    best_left = np.empty(node_cnt,'i')
    best_right = np.empty(node_cnt,'i')

    width = imr.shape[1]

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
                            float *out, size_t &argmin, float &min_diff)
        {
            size_t out_size = cur.size()-ref.size();
            assert(out_size>0);
            size_t in_size = ref.size();

            min_diff = std::numeric_limits<float>::infinity();
            for(size_t i=0; i <= out_size; i++ ){
                float diff = 0;
                for(size_t j=0; j < in_size;j++ ){
                    float err = ref[j] - cur[i+j];
                    err = (err>0.5)? 0.5 : err;
                    diff += err*err;
                }
                out[i] = diff;

                if (diff < min_diff){
                    min_diff = diff;
                    argmin = i;
        } } }

        inline void searchBoundary(float *v, const size_t v_cnt, size_t &origin, size_t &left, size_t &right )
        {
            left = origin;
            while( left>0 ){
                if(v[left-1] < v[left])
                    break;
                else
                    left--;
            }

            right = origin;
            while( right<v_cnt-1 ){
                if(v[right+1] < v[right])
                    break;
                else
                    right++;
        } }
        '''

    code = r'''
        const size_t M = node_cnt;
        const int win_width = %(win_width)d;
        const int win_size = 2*win_width+1;
        std::vector<float> ref_samples(win_size);
        std::vector<float> cur_samples(1000);

        #if %(KEEP_ERR)d
            if (!PyList_Check(py_errs))
                py::fail(PyExc_TypeError, "err must be a list");
            const long int zero = 0;
        #else
            std::vector<float> err(1000);
        #endif

        // std::raise(SIGINT);

        // foreach pixel in ref image
        for(size_t p_id=0; p_id<M; p_id++){
            if (!VALID_MASK1(p_id))
            {
                #if %(KEEP_ERR)d
                    PyObject* new_array = PyArray_SimpleNew(1, (long int*)&zero,  NPY_FLOAT32);
                    PyList_Append(py_errs, new_array);
                #endif
                continue;
            }

            /* 1. Sampling the Intensity */
            const int sam_min = std::floor(VMIN1(p_id));
            const int sam_max = std::ceil(VMAX1(p_id));
            const long int sample_size = sam_max-sam_min+1;
            assert(sample_size>=0);

            cur_samples.resize(sample_size+2*win_width);

            for(int i=0; i<sample_size+2*win_width; i++)
                cur_samples[i] = sample(&IMC1(0), width,
                                        PB2(0,p_id)+(sam_min+i-win_width)*DXY2(0,p_id),
                                        PB2(1,p_id)+(sam_min+i-win_width)*DXY2(1,p_id));

            for(int pos=0; pos<win_size; pos++)
                ref_samples[pos] = sample(&IMR1(0),width,
                                        PX1(p_id)-(pos-win_width)*DXY_LOCAL2(0,p_id),
                                        PY1(p_id)-(pos-win_width)*DXY_LOCAL2(1,p_id));
            #if %(KEEP_ERR)d
                PyObject* new_array = PyArray_SimpleNew(1, (long int*)&sample_size, NPY_FLOAT32);
                PyList_Append(py_errs, new_array);
                float *err = (float *)PyArray_DATA(new_array); // pointer to data.
            #else
                err.resize(sample_size);
            #endif

            /* 2. go through all the points in cur */
            size_t argmin;
            convolution(ref_samples, cur_samples, &err[0], argmin, BEST_ERR1(p_id));

            /* 3. find the boundary of the basin */
            size_t left, right;
            searchBoundary(&err[0], sample_size, argmin, left, right);

            BEST1(p_id) = sam_min + argmin;
            BEST_LEFT1(p_id) = sam_min + left;
            BEST_RIGHT1(p_id) = sam_min + right;
        }
        '''% {'win_width': win_width, 'KEEP_ERR': keepErr}
    weave.inline(code, ['imr','imc','width','node_cnt','pb',
                        'vmax','vmin','dxy','dxy_local','px','py',
                        'best','valid_mask','best_err','best_left','best_right','errs'],#
                support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<map>','<csignal>','<set>'],
                compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])

    searchEPL.ec = ec
    searchEPL.vlist = errs
    searchEPL.rlist=(best_left, best_right)
    searchEPL.dlist=(d_min, d_max)
    searchEPL.err = best_err
#    valid_mask = conditions(valid_mask, best_err<0.0138) # 9*(10/255.0)**2
    res = ec.DfromV(best).ravel()
    return res, valid_mask, best_err


if __name__ == "__main__":
    frames, wGc, K, Zs = loaddata1()
#    frames, wGc, K, Zs = loaddata2()
#    from orb_kfs import loaddata4
#    frames, wGc, K = loaddata4(10)
#    EpilineDrawer(frames[0:], wGc[0:], K)
    h,w = frames[0].shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    ''' set up matching Frame'''

    fs = []
    for fid in range(len(frames)):
        try:
            f = Frame(frames[fid], wGc[fid], Z=Zs[fid])
        except:
            f = Frame(frames[fid], wGc[fid])
        fs.append(f)
    f0,f1 = fs[0],fs[3]
    f0.extractPts(K)

    if 1:
        baseline = lambda f0,f1: np.linalg.norm(getG(f0, f1)[:3,3])
        fs.sort(key=lambda f: baseline(f0,f))
        [baseline(f0,f) for f in fs]
#%%
    from ceres import ba
    cGr = [getG(f,f0) for f in fs]

    #f0.px,f0.py = np.atleast_1d(170,267)
    d0,vm0,err = f0.searchEPL(fs[-1], K, dmin=iD(5), dmax=iD(0.1), win_width=3, levels=1, keepErr=0) #
    plotxyzrgb(f0.makePC(1.0/d0, conditions(vm0, d0>iD(5), d0<iD(0.1))))

    dba = d.copy()
    ba(f0.px, f0.py, dba, vm, [frames[0],frames[6]], [cGr[0], cGr[6]], K)
    plotxyzrgb(f0.makePC(1.0/dba, conditions(vm, dba>iD(5), dba<iD(0.1))))

    res = f0.searchEPLs(fs[1:], K, dmin=iD(5), dmax=iD(0.1), win_width=3) #
    for d,vm,err in res:
        plotxyzrgb(f0.makePC(1.0/d, conditions(vm, d>iD(5), d<iD(0.1))))
        plt.waitforbuttonpress()

    davg = np.add.reduce([e[0] for e in res])/len(res)
    vmavg = reduce(conditions,[e[1] for e in res])
    plotxyzrgb(f0.makePC(1.0/davg, conditions(vmavg, davg>iD(5), davg<iD(0.1))))

    ims = {0:frames[:5]}
    Ks  = {0:K}
    ps  = {0: [f0.px, f0.py]}
    scale_mat = np.diag([0.5, 0.5, 1])
    for level in range(1,5):
        ims[level] = [cv2.pyrDown(im) for im in ims[level-1]]
        Ks[level] = scale_mat.dot(Ks[level-1])
        ps[level] = [0.5*p for p in ps[level-1]]
    d_ = d.copy()
    d_[:] = 0.001
    ds = {5:d_.copy()}
    level_ba = 0
    for level_ba in reversed(range(5)):
        if level_ba not in ds and 1:
            ba(ps[level_ba][0], ps[level_ba][1], d_, vm, ims[level_ba], cGr, Ks[level_ba])
            ds[level_ba] = d_.copy()
        plotxyzrgb(f0.makePC(1.0/ds[level_ba], conditions(vm, ds[level_ba]>iD(5), ds[level_ba]<iD(0.1))))
        plt.waitforbuttonpress()
#%%
    p = f0.getId()
    f0.px,f0.py = np.atleast_1d(52,256)



#%%

    ec0 = searchEPL.ec
    ec1 = EpilineCalculator(f0.px, f0.py, getG(fs[2],f0), K)
    p_valid = ec1.XYfromD(d)

    i0 = f0.im[f0.py, f0.px]
    i_valid = sample(fs[2].im, p_valid[0],p_valid[1])
    mask = conditions(vm, ~np.isnan(i_valid), np.abs(i_valid-i0)<5.0/255)

    pis(sim(f0.im,fs[2].im))
    plt.plot(f0.px, f0.py,'r.',ms=1)
    plt.plot(p_valid[0]+w, p_valid[1],'b.',ms=1)
    plotxyzrgb(f0.makePC(1.0/d, mask))

    d2,vm2,var2 = f0.searchEPL(fs[2], K, dmin=iD(5), dmax=iD(0.1), win_width=4) #
    def mergeD(d0, var0, d1, var1):
        var_sum = var0+var1
        d = var1/var_sum*d0 + var0/var_sum*d1
        var = var1*var0/var_sum
        return d,var
    d3,var3 =mergeD(d,var,d2,var2)
    plotxyzrgb(f0.makePC(1.0/d3, mask))


    def dumpToFs():
        f = plt.figure(num='epl match'); #f.clear()
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')

        pref = (176,284)

        win_width=0
        fs_array={}
        for i in range(1,len(fs)):
            i_str = str(i)
            ec = EpilineCalculator(pref[0], pref[1], getG(fs[i],f0), K)
            vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(f0.im.shape)
            p0,dxy,dxy_local = ec.nPinf, ec.dxy, ec.dxy_local

            ref_pos = vec(pref) - vec(dxy_local)*np.arange(-win_width, win_width+1)
            cur_pos = ec.XYfromD(np.linspace(d_min, d_max, vmax+1))
#            vec(p0) + vec(dxy)*np.arange(np.floor(vmin)-win_width, np.ceil(vmax)+win_width+1)
            ref = sample(f0.im,ref_pos[0],ref_pos[1])
            cur = sample(fs[i].im,cur_pos[0],cur_pos[1])
            fs_array["ref"+i_str] = ref
            fs_array["cur"+i_str] = cur
            err = [np.sum((ref-cur[i:i+len(ref)])**2) for i in range(len(cur)-len(ref)+1) ]
            ab.plot(np.linspace(d_min, d_max, vmax+1),err)
        print pref

    plt.pause(0.01)
#%% main

    if 0:
        test_EPLMatch()
        test_EpilineCalculator()

    if 0:
        ds,vs,ecs,vms,data,dr = [],[],[],[],[[] for _ in range(len(fs)-1)],[[] for _ in range(len(fs)-1)]
        for i in range(1,len(fs)):
            d,vm,var = f0.searchEPL(fs[i], K, dmin=iD(5), dmax=iD(0.1), win_width=3) #
            data[i-1].extend(searchEPL.vlist)
            dr[i-1].extend(searchEPL.rlist)
            ecs.append(searchEPL.ec)
            ds.append(d)
            vs.append(var)
            vms.append(vm)

        def getData(frame, p0):
                minmax = dr[frame][p0]
                minmax = ecs[frame].DfromV(vec(minmax), p0).ravel()
                y = data[frame][p0]
                return minmax,y

        def mergeCurve(minmax1,y1,minmax2,y2):
                len1,len2 = len(y1),len(y2)
                if len1<len2:
                    len1,minmax1,y1,len2,minmax2,y2 = len2,minmax2,y2,len1,minmax1,y1

                dom1 = np.linspace(minmax1[0],minmax1[1],len(y1))
                dom2 = np.linspace(minmax2[0],minmax2[1],len(y2))
                y2_ = np.interp(dom1, dom2, y2)
                return minmax1, y1+y2_

        d_total = np.full_like(f0.px, np.nan,'f')
        for pid in range(len(f0)):
            print pid
            minmax,y = getData(0, pid)
            for frame in range(len(dr)):
                if not vms[frame][pid]:
                    continue

                minmax_,y_ = getData(frame, pid)
                minmax, y = mergeCurve(minmax,y,minmax_,y_)
            d_total[pid] = (float(np.argmin(y))/len(y))*(minmax[1]-minmax[0])+minmax[0]

        plotxyzrgb(f0.makePC(1.0/d_total, conditions(*vms)))

        def test_mergeCurve():
            def plotline(p0):
                pf()
                for frame in range(len(fs)-1):
                    vmin,vmax = dr[frame][p0]
                    dlist = ecs[frame].DfromV(vec(np.arange(vmin,vmax+1)), p0).ravel()
                    plt.plot(dlist, data[frame][p0],'.-')
                    plt.pause(0.01)
                    plt.waitforbuttonpress()

            p0 = f0.getId()
            plotline(p0)
            minmax,y = getData(0, p0)
            for frame in range(len(dr)):
                minmax_,y_ = getData(frame, p0)
                minmax, y = mergeCurve(minmax,y,minmax_,y_)
            y /= len(dr)
            plt.plot(np.linspace(minmax[0],minmax[1],len(y)), y, 'r-',linewidth=5)
            plt.pause(0.01)
            plt.waitforbuttonpress()


        def mergeD(d0, var0, d1, var1):
            var_sum = var0+var1
            d = var1/var_sum*d0 + var0/var_sum*d1
            var = var1*var0/var_sum
            return d,var

#        p0 = 37838
        d,var,vm = ds[0],vs[0],vms[0]
#        print vec([v[p0] for v in vs ])     # variance of each match of p0
        for d_,var_,vm_ in reversed(zip(ds[1:],vs[1:],vms[1:])):
            d,var = mergeD(d, var, d_, var_)
            vm = conditions(vm,vm_)
#            print var[p0]
            plotxyzrgb(f0.makePC(1.0/d, vm))
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
        d,vm,var = f0.searchEPL(fs[3], K, dmin=iD(5), dmax=iD(0.1), win_width=4) #
        I, nbr_list, nbrs_cnt, enode_out = f0.getIncidenceMat(True)
        L = I.transpose().dot(I)
        D = L.diagonal()
        A = scipy.sparse.diags(D) - L

        plotxyzrgb(f0.makePC(1.0/d, vm))
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

            x = d/(Pinf[2] + d*Pe[2])*dxy_norm
            y = np.empty_like(x)
            if debug:
                fig,a = plt.subplots(1,1,num='searchC')
                cset = [a.plot(f0.v[p0])[0] for i in range(2)]
                lset = [a.axvline(obj,0,2, c=color) for obj,color in zip([x[0],x[1]],'bg')]

            Ed = 0
            Cost = f0.v
            for pid in xrange(len(f0.px)):
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






















