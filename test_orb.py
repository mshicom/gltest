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

import scipy
import scipy.ndimage
import scipy.io
from vtk_visualizer import *

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

class Frame(object):
    __slots__ = ['im', 'wGc',               \
                 'px','py','p_cnt','P',     \
                 'nbrs','v','grad','orin',
                 'Z']
    def __init__(self, img, wGc=np.eye(4), extractPts=False, Z=None):
        self.im = np.ascontiguousarray(img.astype('f')/255.0)
        self.wGc = wGc.copy()
        self.Z = Z
        self.py, self.px = None,None
        if extractPts:
            self.extractPts()

    def extractPts(self, K, gthreshold=0.025):
        ''' 1.extract pixels with significant gradients'''
        h,w = self.im.shape

#            grad,orin = scharr(self.im)
#            self.grad, self.orin = grad,orin
        dx,dy = np.gradient(self.im)
        grad = np.sqrt(dx**2 + dy**2)
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
        self.p_cnt = len(x)
        self.v = []
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

    def searchEPL(self, f1, K, dmin=None,dmax=None,win_width=4):
        if self.px is None:
            self.extractPts(K)
        d,vm,var = searchEPL(self.px, self.py, self.im, f1.im, getG(self, f1), K, dmin, dmax, win_width, False)

        self.v = searchEPL.vlist
        return d,vm,var

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

    def getId(self, x=None,y=None):
        global p0
        if x is None:
            plt.figure(num='pick a point');
            plt.imshow(self.im)
            plt.plot(self.px,self.py,'b.')
            (x,y), = np.round(plt.ginput(1, timeout=0))
        p0 = int(np.where(conditions(self.px==x,self.py==y))[0])
        return p0

def getG(f0,f1):
    '''return 1G0, which p1 = 1G0 * p0  '''
    return np.dot(inv(f0.wGc), f1.wGc)

@timing
def searchEPL(px, py, imr, imc, rGc, K, dmin=0, dmax=np.inf, win_width=4, debug=False):
    # px, py, imr, imc, rGc, win_width, debug, dmin, dmax= pref[0], pref[1], f0.im, f1.im, getG(f0,f1), 4, True, None,None
    px,py = np.atleast_1d(px, py)

    ec = EpilineCalculator(px, py, rGc, K)

    vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(imr.shape, dmin, dmax)

    pb,dxy,dxy_local = ec.nPinf, ec.dxy, ec.dxy_local
    var = (d_max-d_min)/(vmax-vmin)

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
            if not valid_mask[p_id]:
                continue
            sam_min,sam_max = int(vmin[p_id]), int(vmax[p_id])
            sam_cnt = int(sam_max-sam_min+1)
            err = np.empty(sam_cnt, 'f')

#            if 0:
#                p0 = vec(px[p_id],py[p_id])
#                ref_pos = p0 - vec(dxy_local[:,n])*np.arange(-win_width, win_width+1)
#                ref_samples = sample(imr, ref_pos[0], ref_pos[1])
#                cur_pos = vec(pb[:,n]) + vec(dxy[:,n])*np.arange(sam_min-win_width, sam_max+win_width+1)
#                cur_samples = sample(imc, cur_pos[0], cur_pos[1])
#
#                for i in xrange(sam_cnt):
#                    diff = ref_samples - cur_samples[i:i+2*win_width+1]
#                    err[i] = (diff*diff).sum()
#                mininum = np.nanargmin(err)
#                best[n] = sam_min+mininum  #np.argpartition(err, 5)
#                cost[n] = err[mininum]
#            else:
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
                ref_pos = p0 - vec(dxy_local[:,p_id])*np.arange(-win_width, win_width+1)
                cur_pos = vec(pb[:,p_id]) + vec(dxy[:,p_id])*np.arange(sam_min-win_width, sam_max+win_width+1)

                al.plot(ref_pos[0], ref_pos[1],'g.'); al.plot(px, py,'r.')
                pm = pb+best*dxy
                ar.plot(cur_pos[0], cur_pos[1],'g.');ar.plot(pm[0],pm[1],'ro');
                ab.plot(err)
                ab.vlines(best[p_id]-sam_min,0,1)

                if not f0.Z is None:
                    pt = trueProj(ref_pos[0],ref_pos[1], cGr=inv(rGc), Zr=f0.Z)
                    ar.plot(pt[0], pt[1],'b.')
                    ab.vlines((pt[0][win_width]-pb[0])/dxy[0]-sam_min,0,1,'g')
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
            if (!VALID_MASK1(p_id))
                continue;

            /* 1. Sampling the Intensity */
            int sam_min = std::floor(VMIN1(p_id));
            int sam_max = std::ceil(VMAX1(p_id));
            int sample_size = sam_max-sam_min+1;
            if(sample_size<1){
                BEST1(p_id) = -1;
                continue;   // discard pixel whose epi-line length is 0

            }
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
            convolution(ref_samples, cur_samples, &err[0], BEST1(p_id));
            BEST1(p_id) += sam_min;

            /* 3. find the best N element */
            #if 0
                auto result = std::min_element(std::begin(err), std::end(err));
                BEST1(p_id) = sam_min + std::distance(std::begin(err), result);
                COST1(p_id) = *result;
            #endif
        }
        '''% {'win_width': win_width }

        weave.inline(code, ['ima','imb','width','node_cnt','pb','vmax','vmin','dxy','dxy_local','px','py','best','valid_mask'],#
            support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<map>','<csignal>','<set>'],
            compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])

    res = ec.DfromV(best[valid_mask],np.where(valid_mask)).ravel()
    return res, valid_mask, var[valid_mask]**2

if __name__ == "__main__":
#    frames, wGc, K, Zs = loaddata1()
    frames, wGc, K = loaddata2()
    EpilineDrawer(frames[0:], wGc[0:], K)
    h,w = frames[0].shape[:2]
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    def test_EpilineCalculator():
        f,a = plt.subplots(1, 1, num='test_EpilineCalculator')
        a.imshow(sim(f0.im, f1.im))

        pref = np.round(plt.ginput(1, timeout=-1)[0])
        a.plot(pref[0], pref[1],'r.')

        ec = EpilineCalculator(pref[0], pref[1], getG(f0,f1), K) #
        vmin,vmax, d_min, d_max, valid_mask = ec.getLimits(f0.im.shape)
        if valid_mask:
            pmin = ec.XYfromV(vmin)
            pmax = ec.XYfromV(vmax)
            a.plot([pmin[0]+640,pmax[0]+640], [pmin[1],pmax[1]],'g-')

            cGr = getG(f1,f0)
            Z = np.linspace(0.5, 10.0, 40)
            pcur = K.dot(transform(cGr, backproject(pref[0], pref[1], K)*Z))
            pcur /= pcur[2]
            a.plot(pcur[0]+640, pcur[1],'b.')

            plt.pause(0.01)
        else:
            print 'epiline not valid'

        try:
            ec2 = EpilineCalculator(f0.px, f0.py, getG(f0,f1), K) #

            tx,ty = trueProj(f0.px, f0.py, getG(f1,f0), Zr=f0.Z)
            td = 1.0/sample(f0.Z, f0.px, f0.py)
            d = ec2.DfromX(tx); assert( np.allclose(td, d) )
            v = ec2.VfromD(td); assert( np.allclose(v, ec2.VfromX(tx)) )
            xy = ec2.XYfromD(td); assert( np.allclose(xy[0], tx) and np.allclose(xy[1], ty))
            z = ec2.ZfromXY(tx,ty); assert( np.allclose(1.0/td, z) )
        except:
            pass

    def test_EPLMatch():
        f = plt.figure(num='epl match'); #f.clear()
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        plt.tight_layout()
        al.imshow(f0.im, interpolation='none'); ar.imshow(f1.im, interpolation='none')

        pref = plt.ginput(1, timeout=-1)[0]
        best0,vm,var = searchEPL(pref[0], pref[1], f0.im, f1.im, getG(f0,f1), K, iD(10),iD(0.1), win_width=4, debug=True)
#        ar.plot(f1.px,f1.py,'b.',ms=2)
        plt.pause(0.01)


    ''' set up matching Frame'''
    refid = 2

    fs = []
    seq = [refid,  3, 4, 9]#,3, 4,5,6,7,8s
    for fid in seq:
        try:
            f = Frame(frames[fid], wGc[fid], Z=Zs[fid])
        except:
            f = Frame(frames[fid], wGc[fid])
        fs.append(f)
    f0,f1 = fs[0],fs[1]
    f0.extractPts(K)
#%% main

    if 0:
        test_EPLMatch()
        test_EpilineCalculator()

    if 1:
        ds,vs,ecs,data,dr = [],[],[],[[] for _ in range(len(fs)-1)],[[] for _ in range(len(fs)-1)]
        for i in range(1,len(fs)):
            d,vm,var = f0.searchEPL(fs[i], K, dmin=iD(5), dmax=iD(1), win_width=3) #
            data[i-1].extend(searchEPL.vlist)
            dr[i-1].extend(searchEPL.rlist)
            ecs.append(searchEPL.ec)
            ds.append(d)
            vs.append(var)

        def test_mergeCurve():
            def plotline(p0):
                pf()
                for frame in range(len(fs)-1):
                    vmin,vmax = dr[frame][p0]
                    dlist = ecs[frame].DfromV(vec(np.arange(vmin,vmax+1)), p0).ravel()
                    plt.plot(dlist, data[frame][p0],'*-')
            @timing
            def mergeCurve(minmax1,y1,minmax2,y2):
                len1,len2 = len(y1),len(y2)
                if len1<len2:
                    len1,minmax1,y1,len2,minmax2,y2 = len2,minmax2,y2,len1,minmax1,y1

                dom1 = np.linspace(minmax1[0],minmax1[1],len(y1))
                dom2 = np.linspace(minmax2[0],minmax2[1],len(y2))
                y2_ = np.interp(dom1, dom2, y2)
                return minmax1, 0.5*(y1+y2_)

            def getData(frame):
                minmax = dr[frame][p0]
                minmax = ecs[frame].DfromV(vec(minmax), p0).ravel()
                y = data[frame][p0]
                return minmax,y

            plotline(f0.getId())
            minmax,y = getData(0)
            for frame in range(len(dr)):
                minmax_,y_ = getData(frame)
                minmax, y = mergeCurve(minmax,y,minmax_,y_)
            plt.plot(np.linspace(minmax[0],minmax[1],len(y)), y, '*-',linewidth=4)



        def mergeD(d0, var0, d1, var1):
            var_sum = var0+var1
            d = var1/var_sum*d0 + var0/var_sum*d1
            var = var1*var0/var_sum
            return d,var

#        p0 = 37838
        d,var = ds[0],vs[0]
        print vec([v[p0] for v in vs ])     # variance of each match of p0
        for d_,var_ in reversed(zip(ds[1:],vs[1:])):
            d,var = mergeD(d, var, d_, var_)
#            print var[p0]
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
        d,vm,var = f0.searchEPL(f1, K, dmin=iD(5), dmax=iD(0.1), win_width=3) #
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

