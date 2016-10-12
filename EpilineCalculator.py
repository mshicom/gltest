#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:50:58 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt

import scipy

from tools import *
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

def relG(wG0, wG1):
    return np.dot(np.linalg.inv(wG0), wG1)

class EpilineCalculator(object):
    """
    Examples
    --------
    >>> ec = EpilineCalculator(xr, yr, cGr, K)
    >>> vmin, vmax, d_min, d_max, valid_mask = ec.getLimits(frames[index].shape)

    Notes
    -----
    1. suppose (X, X') are the Ref and Cur image pixel pairs, given the relative
    camera pos (R,T), X'=(R,T).dot(X), we have a ray X'∈R3:
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
    """
    def __init__(self, xr, yr, cGr, K):
        # xr, yr, rGc, K = f0.px, f0.py, getG(f0,f1), K
        xr,yr = np.atleast_1d(xr,yr)

        Rcr,Tcr = cGr[:3,:3], cGr[:3,3]
        Rrc,Trc = Rcr.T, -Rcr.T.dot(Tcr)

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

        self.rFc = lambda : inv(K.T).dot(skew(Trc)).dot(Rrc).dot(inv(K))  # xr'*rFc*xc = 0 '''

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

        self.v_limits = []
        def getLimits(shape, dmin=0.0, dmax=1e6):
            """
            There are in total 5 constraints in the epiline parameters λ and v:
            a. Valid image region: 0<x<w, 0<y<h;
            b. Expected search range indicated by [dmin, dmax];
            c. if Pe[3]>0, v_max=dxy_norm/Pe[3] for λ=np.inf
            d. if Pe[3]<0, Ref is behind Cur, λ<-Pinf[2]/Pe[2] to ensure the resulting 3D point
                will be in front of the Cur camera,
            e. total epiline length is no less than 1
            """
            # a. valid border is trimmed a little bit, i.e. 4 pixels, for
            h,w = shape
            tx = self.VfromX(vec(4, w-4))
            tx = np.where(dxy[0]>0, tx, np.roll(tx,1,axis=0))   # tx[0,1] := [v_xmin,v_xmax]
            ty = self.VfromY(vec(4, h-4))
            ty = np.where(dxy[1]>0, ty, np.roll(ty,1,axis=0))   # ty[0,1] := [v_ymin,v_ymax]\
            v_xmin,v_xmax,v_ymin,v_ymax = tx[0],tx[1],ty[0],ty[1]
            vmax = np.minimum(v_xmax, v_ymax)
            vmin = np.maximum(v_xmin, v_ymin)
            valid_mask = conditions(v_xmin<v_ymax, v_xmax>v_ymin, vmax>0)

            # d
            if Pe[2]<0:
                dmax = np.clip(dmax, dmin, (1e-3-Pinf[2])/Pe[2])
            # b
            vmin = np.maximum(vmin, self.VfromD(dmin))
            vmax = np.minimum(vmax, self.VfromD(dmax))
            # c
            if Pe[2]>0:
                vmax = np.minimum(vmax, dxy_norm/Pe[2])
            vmin, vmax = np.ceil(vmin), np.floor(vmax)
            d_min, d_max = self.DfromV(vmin), self.DfromV(vmax)
            # e
            valid_mask = conditions(valid_mask, (vmax-vmin)>1)
            self.v_limits = (vmin.astype('i4'), vmax.astype('i4'))
            return vmin.astype('i4'), vmax.astype('i4'), d_min, d_max, valid_mask
        self.getLimits = getLimits

        def getDRange(pid):
            if self.v_limits:
                vmin,vmax = self.v_limits[0][pid], self.v_limits[1][pid]
                return self.DfromV(vec(np.arange(vmin,vmax+1)), pid).ravel()
        self.getDRange = getDRange

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
                sam_min,sam_max = vmin[p_id], vmax[p_id]
                sam_cnt = int(sam_max-sam_min+1)
                dom.append(self.DfromV(np.arange(sam_min, sam_max+1), p_id))
                err = np.empty(sam_cnt, 'f')

                ref_pos = vec(xr[p_id], yr[p_id]) - vec(dxy_local[:,p_id])*offset
                cur_pos = vec(nPinf[:,p_id]) + vec(dxy[:,p_id])*np.arange(np.floor(vmin[p_id])-win_width, np.ceil(vmax[p_id])+win_width+1)
                ref = sample(imr.astype('f'), ref_pos[0], ref_pos[1])
                cur = sample(imc.astype('f'), cur_pos[0], cur_pos[1])
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

        self.ind = 1
        self.ecs = {}
        slices= len(frames)-1

        f = plt.figure()
        gs = plt.GridSpec(2,2)
        a1,a2 = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        a3 = f.add_subplot(gs[1,:])
        plt.tight_layout()

        a1.imshow(frames[0], interpolation='none', aspect=1)
        a1.set_title('pick a point in this nPinfimage')
        a1.autoscale(False)

        i2 = a2.imshow(frames[1], interpolation='none', aspect=1)
        a2.autoscale(False)

        if p0 is None:
            xr, yr = np.round(plt.ginput(1, timeout=-1)[0])
        else:
            xr, yr = p0
        a1.plot(xr, yr,'r.')
        a1.set_title(' ')

        line1, = a2.plot([],'r-')        # epiline
        ticks, = a2.plot([],'b.',ms=5)   # fixed depth ticks
        Ddot, = a2.plot([],'ro',ms=5)    # moving depth tick
        lineD = a3.axvline()             # vline for moving depth tick
        Z = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        res,dom = {},{}


        for index in range(1, len(frames)):
            cGr = relG(wGc[index],wGc[0])
            ec = EpilineCalculator(xr, yr, cGr, K)
            res[index], dom[index] = ec.searchEPL(frames[0].astype('f'), frames[index].astype('f'),3,0)
            self.ecs[index] = ec
        curv1, = a3.plot(res[1])    # cost function curve

        def updateFrame():
            index = self.ind
            i2.set_data(frames[index])
            a2.set_title('frame %d' % index)

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
                ticks.set_data(pcur[0],pcur[1])
                curv1.set_data(dom_, res_)
                a3.set_xlim(dom_[0], dom_[-1])
            else:
                print 'frame %d epiline not valid' % index
                line1.set_data([],[])
                ticks.set_data([],[])
                Ddot.set_data([],[])
                curv1.set_data([],[])
            f.canvas.draw()

        self.d = np.inf
        self.v = 0
        def updateFloatingTick():
            ec = self.ecs[self.ind]
            vec0, pbase = ec.dxy.ravel(), ec.nPinf.ravel()
            self.v = ec.VfromD(self.d)

            pset =  pbase + self.v*vec0
            Ddot.set_data(pset[0], pset[1])
            lineD.set_xdata(self.d)
            a3.set_title('depth: %f' % (1/self.d))
            f.canvas.draw()

        def onscroll(event):
            if event.button == 'down':
                self.ind = np.clip(self.ind + 1, 1, slices)
            else:
                self.ind = np.clip(self.ind - 1, 1, slices)
            updateFrame()
            updateFloatingTick()

        def onmotion(event):
            if event.inaxes == a2 and 0:
                pmouse = np.array([event.xdata, event.ydata])
                ec = self.ecs[self.ind]
                vec0, pbase = ec.dxy.ravel(), ec.nPinf.ravel()

                vec1 = pmouse - pbase
                self.v = np.maximum(0, vec0.dot(vec1))
                self.d = ec.DfromV(self.v)
                updateFloatingTick()

            elif event.inaxes == a3:
                ec = self.ecs[self.ind]
                self.d = np.maximum(0, event.xdata)
                self.v = ec.VfromD(self.d)
                updateFloatingTick()

        updateFrame()
        f.canvas.mpl_connect('scroll_event', onscroll)
        f.canvas.mpl_connect('motion_notify_event', onmotion)

def InterceptBox(shape, p0,  dxy):
    h,w = shape[:2]
    dx,dy = dxy[:2]
    x0,y0 = p0[:2]

    tx = np.r_[0-x0, w-x0]/dx
    tx = np.where(dx>0, tx, np.roll(tx,1,axis=0))   # tx[0,1] := [v_xmin,v_xmax]
    ty = np.r_[0-y0, h-y0]/dy
    ty = np.where(dy>0, ty, np.roll(ty,1,axis=0))   # ty[0,1] := [v_ymin,v_ymax]\
    v_xmin,v_xmax,v_ymin,v_ymax = tx[0],tx[1],ty[0],ty[1]
    vmax = np.minimum(v_xmax, v_ymax)
    vmin = np.maximum(v_xmin, v_ymin)
    valid_mask = conditions(v_xmin<v_ymax, v_xmax>v_ymin, vmax>0)
    return vmin, vmax, valid_mask

def sampleEpl(xr,yr, imr, imc, cGr, K):
    e = EpilineCalculator(xr,yr, cGr, K)

    vmin, vmax, valid_mask = InterceptBox(imr.shape, vec(xr,yr), -e.dxy_local)
    pref = -np.arange(np.ceil(vmin+1),np.floor(vmax))*e.dxy_local + vec(xr,yr)
    ref = sample(imr, pref[0], pref[1])

    vmin, vmax, valid_mask = InterceptBox(imr.shape, e.nPinf, e.dxy )
    pcur = np.arange(np.ceil(vmin+1),np.floor(vmax))*e.dxy + e.nPinf
    cur = sample(imc, pcur[0], pcur[1])
    return ref, cur

if __name__ == "__main__":
#    frames, wGc, K, _ = loaddata1()
    from orb_kfs import loaddata4
    frames, wGc, K = loaddata4(10)

    xr,yr = 718.0, 451.0
    e=EpilineDrawer(frames, wGc, K, (xr,yr))

    #%%
    ref, cur = sampleEpl(xr, yr, frames[0], frames[-1], relG(wGc[-1], wGc[0]), K)
    f = pf()
    l1,l2 = plt.plot(ref,'r', cur,'b')
    Slider(l2)


    from scipy import weave
    def dp(ref, cur, occ_cost):
        result = np.full_like(ref, -1,'i2')
        code = r"""
            size_t M = Nref[0];
            size_t N = Ncur[0];
            size_t N1 = N+1;
            auto start = std::chrono::system_clock::now();

            auto Costs = new float[(M+1)*(N+1)];
            auto Bests = new unsigned char[(M+1)*(N+1)];

            #define C(y,x)  Costs[(y)*N1+(x)]
            #define B(y,x)  Bests[(y)*N1+(x)]

            for (size_t m=0; m<=M; m++)
                C(m, 0) = m*occ_cost;
            for (size_t n=1; n<=N; n++)
                C(0, n) = n*occ_cost;

            const size_t win_hsize = 2;
            for (size_t m=1,md=0; m<=M; m++,md++)
                for(size_t n=1,nd=0; n<=N; n++,nd++ ) {

                    float Edata = 0;
                    if(md >= win_hsize && md <= M-win_hsize
                    && nd >= win_hsize && nd <= N-win_hsize) {
                        for(size_t i=0; i<2*win_hsize+1; i++)
                            Edata += std::fabs((float)REF1(md-win_hsize+i) - (float)CUR1(nd-win_hsize+i));
                    }
                    else
                        Edata = std::fabs((float)REF1(md) - (float)CUR1(nd));
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

            //auto duration = std::chrono::duration<double>
            //    (std::chrono::system_clock::now() - start);
            //std::cout <<"runtime:" <<duration.count() << "s" <<std::endl;
        """
        weave.inline(code,
                   ['ref', 'cur', 'occ_cost','result'],
                    compiler='gcc',headers=['<chrono>','<cmath>'],
                    extra_compile_args=['-std=gnu++11 -msse2 -O3'],
                    verbose=2  )
        return result
    MtoN = dp(ref, cur, 10)
    [plt.plot([i, MtoN[i]], [ref[i], cur[MtoN[i]]],'g--') for i in range(len(MtoN)) if MtoN[i]!=-1]


    #%%
    e0 = EpilineCalculator(xr,yr, relG(wGc[-1], wGc[0]), K)
    err,dom = e0.searchEPL(frames[0], frames[-1])

    pf()
    plt.plot(err)

    best = 156 # np.argmin(err)
    v0 = e0.VfromD(dom[best])

    xc,yc = e0.XYfromD(dom[best])
    e1 = EpilineCalculator(xc,yc, relG(wGc[0], wGc[-1]), K)
    err1,dom1 = e1.searchEPL(frames[-1], frames[0])
    plt.plot(err1)
    best1 = np.argmin(err1)
    v = e1.VfromD(dom1[best1])
    print v0,v


    def test_EpilineCalculator():
        try:
            ec2 = EpilineCalculator(f0.px, f0.py, getG(f1,f0), K) #

            tx,ty = trueProj(f0.px, f0.py, getG(f1,f0), Zr=f0.Z)
            td = 1.0/sample(f0.Z, f0.px, f0.py)
            d = ec2.DfromX(tx); assert( np.allclose(td, d) )
            v = ec2.VfromD(td); assert( np.allclose(v, ec2.VfromX(tx)) )
            xy = ec2.XYfromD(td); assert( np.allclose(xy[0], tx) and np.allclose(xy[1], ty))
            z = ec2.ZfromXY(tx,ty); assert( np.allclose(1.0/td, z) )
        except:
            pass