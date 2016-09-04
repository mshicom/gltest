#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 22:55:41 2016

@author: nubot
"""
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/kaihong/workspace/gltes")
sys.path.append("/home/nubot/data/workspace/gltes")
from tools import *
from EpilineCalculator import EpilineDrawer,EpilineCalculator
from test_orb import Frame,getG
from vtk_visualizer import plotxyzrgb,plotxyz
from scipy import weave

import cv2

def sample(dIc,x,y):
    x,y = np.atleast_1d(x, y)
    return scipy.ndimage.map_coordinates(dIc, (y,x), order=1, mode =  'nearest')

#frames, wGc, K, Zs = loaddata1()
frames, wGc, K = loaddata2()
#    from orb_kfs import loaddata4
#    frames, wGc, K = loaddata4(20)
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
f0 = fs[0]
f0.extractPts(K)

if 0:
    baseline = lambda f0,f1: np.linalg.norm(getG(f0, f1)[:3,3])
    fs.sort(key=lambda f: baseline(f0,f))
    [baseline(f0,f) for f in fs]

#%%
pf()
pis(f0.im)
x,y = plt.ginput(0,-1)[0]

def interceptLine(xmin,xmax,ymin,ymax, x0, y0,dxy):
    tx = (vec(xmin,xmax) - x0) / dxy[0]                # x0+tx*dx = xmin/xmax, (2x1 - 1xN)/1xN
    tx = np.where(dxy[0]>0, tx, np.roll(tx,1,axis=0))   # tx[0,1] := [v_xmin,v_xmax]
    ty = (vec(ymin,ymax) - y0) / dxy[1]
    ty = np.where(dxy[1]>0, ty, np.roll(ty,1,axis=0))   # ty[0,1] := [v_ymin,v_ymax]\
    v_xmin,v_xmax,v_ymin,v_ymax = tx[0],tx[1],ty[0],ty[1]
    vmax = np.minimum(v_xmax, v_ymax)
    vmin = np.maximum(v_xmin, v_ymin)
    valid_mask = conditions(v_xmin<v_ymax, v_xmax>v_ymin, vmax>0)
    return vmin, vmax, valid_mask

f1 = fs[2]
ec = EpilineCalculator(x, y, getG(f0,f1), K)
dxy_l, dxy,pinf = ec.dxy_local, ec.dxy, ec.nPinf

vri,vra, valid = interceptLine(1,w-1,1,h-1, x, y, dxy_l)
assert(valid)
ref = np.fliplr([np.arange(vri,vra+1,dtype='i' )])[0]
ref_sample = sample(f0.im, x+ref*dxy_l[0], y+ref*dxy_l[1])

vci,vca, valid = interceptLine(1,w-1,1,h-1, pinf[0], pinf[1], dxy)
assert(valid)
cur = np.arange(vci, vca+1, dtype='i' )
cur_sample = sample(f1.im, pinf[0]+cur*dxy[0], pinf[1]+cur*dxy[1])

@timing
def fast_dp2(ref_sample, cur_sample, occ_cost):
    result = np.full_like(ref,-1,'i2')

    code = r"""
        const size_t M = Nref_sample[0];
        const size_t N = Ncur_sample[0];
        const size_t N1 = N+1;

        //std::raise(SIGINT);
        const float occ_cost = %(occ_cost)f;

        auto Costs = new float[(M+1)*(N+1)];
        auto Bests = new unsigned char[(M+1)*(N+1)];

        #define C(y,x)  Costs[(y)*N1+(x)]
        #define B(y,x)  Bests[(y)*N1+(x)]

        for (size_t m=0; m<=M; m++)
            C(m, 0) = m*occ_cost;
        for (size_t n=1; n<=N; n++)
            C(0, n) = n*occ_cost;

        for (size_t m=1,md=0; m<=M; m++,md++)
            for(size_t n=1,nd=0; n<=N; n++,nd++ )
            {
                float err = REF_SAMPLE1(md) - CUR_SAMPLE1(nd);

                float Edata = err*err;
                float c1 = C(m-1, n-1) + Edata;
                float c2 = C(m-1, n) + occ_cost;
                float c3 = C(m, n-1) + occ_cost;

                float c_min = c1;
                unsigned char  c_min_id = 0;

                if(c2<c_min) { c_min=c2; c_min_id=1; }
                if(c3<c_min) { c_min=c3; c_min_id=2; }

                C(m, n) = c_min;
                B(m, n) = c_min_id;
            }

        int l=M, r=N;
        while (l!=0 && r!=0)
            switch(B(l,r)) {
                case 0:
                    RESULT1(l-1) = r-1;
                    l -= 1; r -= 1;
                    break;
                case 1:
                    l -= 1; break;
                case 2:
                    r -= 1; break;
                default:
                    std::cerr << "unknown value of x";
                    goto exit_loop;
            }
        exit_loop: ;

        delete[] Costs;
        delete[] Bests;
        #undef C(y,x)
        #undef B(y,x)
    """% {'occ_cost': occ_cost}
    weave.inline(code,
               ['ref_sample', 'cur_sample','result'],
                compiler='gcc',headers=['<csignal>','<cmath>'],
                extra_compile_args=['-std=gnu++11 -msse2 -O3'],
                verbose=2  )
    return result
match= fast_dp2(ref_sample, cur_sample, 0.0016)

f = plt.figure()
gs = plt.GridSpec(2,2)
a1,a2 = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
a3 = f.add_subplot(gs[1,:])
#plt.tight_layout()

a1.imshow(f0.im, interpolation='none', aspect=1)
a1.set_title('pick a point in this image')
a1.autoscale(False)
a1.plot(x,y,'r.')
a1.plot([x+vri*dxy_l[0],x+vra*dxy_l[0]],
        [y+vri*dxy_l[1],y+vra*dxy_l[1]])

i2 = a2.imshow(f1.im, interpolation='none', aspect=1)
a2.autoscale(False)
a2.plot(pinf[0],pinf[1],'r.')
a2.plot([pinf[0]+vci*dxy[0], pinf[0]+vca*dxy[0]],
        [pinf[1]+vci*dxy[1], pinf[1]+vca*dxy[1]])

mask = match!=-1
a3.plot(cur[match[mask]], ref_sample[mask],'r')
a3.plot(cur, cur_sample, 'b')
d = ref[mask]+cur[match[mask]]

a1.plot(f0.px,f0.py,'b.',ms=1)