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

def homogeneous(P):
    return np.lib.pad(P, ((0,1),(0,0)), mode='constant', constant_values=1)

normalize = lambda x:x/np.linalg.norm(x)
snormalize = lambda x:x/np.linalg.norm(x, axis=0)
vec = lambda x:np.reshape(x,(-1,1))
class Scaler:
    def __init__(self, vmin, vmax, levels):
        self.min, self.max, self.levels = (vmin, vmax, levels)
        self.a = levels/(vmax-vmin)
        self.b = -self.a*vmin

    def __call__(self, value, isInvert = False):
        if isInvert:
            return (value-self.b)/self.a
        else:
            return self.a*value+self.b

if __name__ == "__main__":
    if 'frames' not in globals() or 1:
        frames, wGc, K, Zs = loaddata1()
        h,w = frames[0].shape[:2]

    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    refid, curid = 0,9
    Iref, G0, Z = frames[refid].astype('f')/255.0, wGc[refid].astype('f'), Zs[refid].astype('f')
    Icur, G1  = frames[curid].astype('f')/255.0, wGc[curid].astype('f')
    Iref3 = np.tile(Iref.ravel(), (3,1))
    Icur3 = np.tile(Icur.ravel(), (3,1))
#%%
    cGr = np.dot(np.linalg.inv(wGc[curid]), wGc[refid])
    Rcr, Tcr = cGr[0:3,0:3], cGr[0:3,3]
    rGc = np.dot(np.linalg.inv(wGc[refid]), wGc[curid])
    Rrc, Trc = rGc[0:3,0:3], rGc[0:3,3]

    u,v = np.meshgrid(range(w), range(h))
    u,v = (u.ravel()-cx)/fx, (v.ravel()-cy)/fy
    pref = np.vstack([u, v, np.ones(w*h) ]).astype('f')
    pt = pref*Z.ravel()
    p0 = np.vstack([pt,Iref3])

    pref /= np.linalg.norm(pref, axis=0)*6
    p1 = np.vstack([pref,Iref3])

#    plotxyz(np.vstack([pref,Iref3]).T)

    pcur = rGc.dot(homogeneous(pref))[:3]
    p2 = np.vstack([pcur,Icur3])
#    plotxyz(np.vstack([pcur, Icur3]).T, hold=True)

    vis = get_vtk_control()
    vis.RemoveAllActors()
    vis.AddPointCloudActor(np.hstack([p0,p1,p2]).T)
    vis.AddLine([0,0,0], Trc)

    p = (182,286)#(202,299)
    ps = np.array([(p[0]-cx)/fx,(p[1]-cy)/fy,1])*Z[p[1],p[0]]
    vis.AddLine([0,0,0], ps)
    vis.AddLine(Trc, ps)
#%% calculate the
    '''define vectors correspond to 4 image corners '''
    corners = [[0,0],[0,h],[w,h],[w,0]]
    corners = [normalize(np.array([(cn[0]-cx)/fx,
                                   (cn[1]-cy)/fy,
                                   1])) for cn in corners]

    '''generate new coordinate system'''
    ax_z = normalize(Trc)                          # vector pointed to camera Cur serve as z axis
    ax_y = normalize(np.cross(ax_z, corners[0]))   # top-left corner serves as temperary x axis
    ax_x = normalize(np.cross(ax_y, ax_z))
    M = np.vstack([ax_x,ax_y,ax_z])

    '''transform the vector to new coordinate and then calculate the vector
       angle wrt. to x axis'''
    new_ps = [M.dot(cn) for cn in corners]
    angles = [np.rad2deg(np.arctan2(p[1], p[0])) for p in new_ps]

    '''re-adjust the x,y axis so that all pixel lies on the same half-plane'''
    ax_min = np.argmin(angles)
    ax_y = normalize(np.cross(ax_z, corners[ax_min]))   # top-left corner serves as temperary x axis
    ax_x = normalize(np.cross(ax_y, ax_z))
    M = np.vstack([ax_x,ax_y,ax_z])
    new_ps = [M.dot(cn) for cn in corners]
    angles = [np.rad2deg(np.arctan2(p[1], p[0])) for p in new_ps]
    print angles

    if 1:
        phi,theta = np.meshgrid(range(78), range(10,170))
        phi = np.deg2rad(phi.ravel())
        theta = np.deg2rad(theta.ravel())
        pxyz = np.vstack([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)])
        pxyz = M.T.dot(pxyz)
        vis.AddPointCloudActor(pxyz.T)


#%% generate target point
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)

    grad = calcGradient(Iref)
    grad_threshold = np.percentile(grad,80)

    u, v = np.meshgrid(range(w),range(h))
    ub, vb = (u-cx)/fx, (v-cy)/fy


    mask_ref = reduce(np.logical_and,[grad>grad_threshold, u>1, v>1, u<w-2, v<h-2])
    puv_ref = np.array(np.where(mask_ref)).T

    pts = np.vstack([ub[mask_ref], vb[mask_ref], np.ones(mask_ref.sum())])

    grad_ref = grad[mask_ref]
    grad_scaler = Scaler(grad_ref.min(), grad_ref.max(), 255)
    grad_ref = vec(grad_scaler(grad_ref))

    '''calc angle'''
    pvp = M.dot(pts)
    ang_ref =  np.rad2deg(np.arctan2(pvp[1,:], pvp[0,:]))
    ang_ref[ang_ref<0] += 360
    ang_scaler = Scaler(ang_ref.min(), ang_ref.max(), 360)
    ang_ref = vec(ang_scaler(ang_ref))

    ang_ref_z =  vec(np.rad2deg(np.arctan2(np.linalg.norm(pvp[:2,:],axis=0), pvp[2,:])))
    ang_ref_z[ang_ref_z<0] += 360

    '''fill the data structure'''
    data = [[] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_ref, ang_ref,ang_ref_z, grad_ref):
        """put pixels into bins base on their color"""
        v_int = np.round(a)
        data[int(v_int)].append((np.double(az), np.double(a - v_int),tuple(p)))

    if 0:
        rec_im = np.zeros((361,361))
        for p,a,az,g in zip(puv_ref, ang_ref,ang_ref_z, grad_ref):
           rec_im[int(np.round(az)),int(np.round(a))]=1
        pis(rec_im[57:140,:])


#%%
    grad = calcGradient(Icur)
    mask_cur = reduce(np.logical_and,[grad>grad_threshold, u>1, v>1, u<w-2, v<h-2])
    puv_cur = np.array(np.where(mask_cur)).T
    pts_cur = np.vstack([ub[mask_cur], vb[mask_cur], np.ones(mask_cur.sum())])
    pts_cur = M.dot(Rrc.dot(pts_cur))
    ang_cur = np.rad2deg(np.arctan2(pts_cur[1,:], pts_cur[0,:]))
    ang_cur[ang_cur<0] += 360
    ang_cur = vec(ang_scaler(ang_cur))
    ang_cur_z =  vec(np.rad2deg(np.arctan2(np.linalg.norm(pts_cur[:2,:],axis=0), pts_cur[2,:])))
    ang_cur_z[ang_cur_z<0] += 360

    grad_cur = grad[mask_cur]
    grad_cur = vec(grad_scaler(grad_cur))


    data_cur = [[] for _ in range(ang_scaler.levels+1)]
    for p,a,az,g in zip(puv_cur, ang_cur,ang_cur_z, grad_cur):
        """put pixels into bins base on their color"""
        if a > 360 or g > 255:
            continue
        v_int = np.round(a)
        data_cur[int(v_int)].append((np.double(az), np.double(a - v_int) ,tuple(p)))

#%% exam the depth calculation

    positive_range = lambda x: np.where(x>0, x, x+2*np.pi) #x if x>0 else x+2*np.pi
    def calcAngle(x, y, G=None):
        p0 = np.array([(x-cx)/fx, (y-cy)/fy, np.ones(len(x))])
        if not G is None:
            p0 = G[0:3,0:3].dot(p0)
        p = M.dot(p0)
        theta = positive_range(np.arctan2(p[1], p[0]))
        phi = positive_range(np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2]))
        return theta, phi

    def calcRange(ar, ac):
        B = np.linalg.norm(Trc)
        c = np.pi-ac
        b = ac-ar
        return B*np.sin(c)/np.sin(b)

    def rangetoAngle(ac, r):
        B = np.linalg.norm(Trc)
        c = np.pi-ac
        return ac-np.arcsin(B/r*np.sin(c))

    def test_calcRange():
        f,a = plt.subplots(1, 1, num='test_depth')
        a.imshow(sim(Iref, Icur))
        while 1:
            plt.pause(0.01)
            pref = np.round(plt.ginput(1, timeout=-1)[0])
            a.plot(pref[0], pref[1],'r.',ms=2)
            pcur = trueProj(pref[0], pref[1])
            a.plot(pcur[0]+640, pcur[1],'b.',ms=2)
            a_ref = calcAngle(pref[0], pref[1])
            a_cur = calcAngle(pcur[0], pcur[1], rGc)
            prange = calcRange(a_ref[1], a_cur[1])
            P = snormalize(np.array([(pref[0]-cx)/fx, (pref[1]-cy)/fy, 1.0]))*prange
            print 'Ground truth:%f, estimated:%f' % (P[2], Z[pref[1],pref[0]])
#    test_calcRange()



#%% demo: points on the scanline

    def trueProj(x, y, G=cGr, Z=Z):
        p0 = np.array([(x-cx)/fx, (y-cy)/fy, np.ones(len(x))])*Z[y,x]
        p =  K.dot(G[0:3,0:3].dot(p0)+G[0:3,3][:,np.newaxis])
        p /= p[2]
        return p[0],p[1]

    from sklearn.neighbors import NearestNeighbors
    def trueAssignmentForCur(curx, cury, rx, ry):
        tx, ty = trueProj(rx, ry)
        tp = np.round([tx,ty])
        tree = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=4).fit(tp.T)
        dis, ind = tree.kneighbors(np.array([curx, cury]).T)
        ind[dis>0] = -1
        return ind

    f,a = plt.subplots(1,2,num='all points')
    a[0].imshow(Icur, interpolation='none')

    a[0].plot(puv_cur[:,1], puv_cur[:,0],'b.',ms=5)
    tx,ty = trueProj(puv_ref[:,1], puv_ref[:,0])
    a[0].plot(tx,ty,'g.',ms=5)

    a[1].imshow(Iref, interpolation='none')
    a[1].plot(puv_ref[:,1], puv_ref[:,0],'r.',ms=5)

    if 0:
        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        ar,ac = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        for ang in range(65,ang_scaler.levels+1):
#            ac.clear(); ar.clear();
            ab.clear()
            ar.imshow(Iref, interpolation='none'); ac.imshow(Icur, interpolation='none')
            pr,pc = data[ang],data_cur[ang]

            if pc:
                pc.sort()
                pc = zip(*pc)
                y, x = zip(*pc[2])
                ab.plot(pc[0],Icur[y,x]*255+np.array(pc[1])*50,'r*-')
                ac.plot(x, y,'r.')

            if pr:
                pr.sort()
                pr = zip(*pr)

                y, x = zip(*pr[2])
                ab.plot(pr[0],Iref[y,x]*255+np.array(pr[1])*50,'b*-')
                ar.plot(x, y,'b.')

                tx,ty = trueProj(np.array(x), np.array(y))
                ac.plot(tx,ty,'g.')

            plt.pause(0.01)
            plt.waitforbuttonpress()

    if 1:
        from matplotlib.patches import ConnectionPatch
        d_result = np.full_like(Icur, np.nan,'f')
        for ang in range(ang_scaler.levels+1):
            pr,pc = data[ang],data_cur[ang]

            if pc and pr:
                pc.sort()
                pc = zip(*pc)
                la,laz = np.array(pc[0]), np.array(pc[1])

                pr.sort()
                pr = zip(*pr)
                ry, rx = map(np.array,zip(*pr[2]))
                ra, raz = np.array(pr[0]), np.array(pr[1])


                cury, curx = map(np.array, zip(*pc[2]))
                idInRef = trueAssignmentForCur(curx, cury, rx, ry).ravel()

                lyc,lxc,lac,match_idx = ( np.compress(idInRef!=-1, dump) for dump in [cury,curx,la,idInRef] )
                rxm,rym,ram = ( np.take(dump, match_idx) for dump in [rx,ry,ra] )
#                d_result[lyc, lxc] = calcRange(np.deg2rad(ang_scaler(ram, isInvert=True)),
#                                               np.deg2rad(ang_scaler(lac, isInvert=True)))

                ac = calcAngle(lxc,lyc,rGc)[1]
                ar = calcAngle(rxm,rym)[1]

                d_result[lyc, lxc] = calcRange(ar,ac)
#                ''' added line between matched pairs'''
#                tx,ty = trueProj(rx, ry)
#                for x0,y0,ind in zip(lxc,lyc,match_idx):
#                    a.add_artist(ConnectionPatch(xyA=(tx[ind],ty[ind]), xyB=(x0,y0),
#                                          coordsA='data', coordsB='data',
#                                          axesA=a, axesB=a))
#                plt.pause(0.01)
#                plt.waitforbuttonpress()
                print ang

        v,u = np.where(~np.isnan(d_result))
        p3d = snormalize(np.array([(u-cx)/fx, (v-cy)/fy, np.ones(len(u))]))*d_result[v,u]
        plotxyzrgb(np.vstack([p3d,np.tile(Icur[v,u]*255,(3,1))]).T)
#%% icp
    from matplotlib.patches import ConnectionPatch
    from scipy.optimize import linear_sum_assignment
    debug = True

    if debug:
        f = plt.figure(num='icp')

        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        plt.tight_layout()

    lim, rim = Icur, Iref #calcGradient(Icur), calcGradient(Iref)

    for a in [180]:#range(ang_scaler.levels+1):
        pr,pc = data[a],data_cur[a]

        if pc and pr:
            pc.sort()
            pc = zip(*pc)
            pr.sort()
            pr = zip(*pr)

            ry, rx = map(np.array,zip(*pr[2]))
            tx, ty = trueProj(rx, ry)
            cury, curx = map(np.array, zip(*pc[2]))

            ca,ra = np.array(pc[0]), np.array(pr[0])
            cb,rb = np.array(pc[1]), np.array(pr[1])

            idInRef = trueAssignmentForCur(curx, cury, rx, ry).ravel()

            cyc,cxc,cac,match_idx = ( np.compress(idInRef!=-1, dump) for dump in [cury,curx,ca,idInRef] )
            rxc,ryc,rac = ( np.take(dump, match_idx) for dump in [rx,ry,ra] )

            vl, vr = lim[cury, curx], rim[ry,rx]

            al.clear(); ar.clear();
            al.imshow(Icur, interpolation='none'); al.plot(curx,cury,'b.');al.plot(tx,ty,'g.')
            ar.imshow(Iref, interpolation='none'); ar.plot(rx,ry,'r.')

            ab.plot(ca,cb,'ro')
            ab.plot(ra,rb,'bs')

            ab.plot([cac,rac],[cb[idInRef!=-1],rb[match_idx]],'g-')

            plt.pause(0.01)
            plt.waitforbuttonpress()


#%%
#    lim, rim = (Icur*255).astype('u1').copy(), (Iref*255).astype('u1').copy()
    lim, rim = calcGradient(Icur*255), calcGradient(Iref*255)
    x_off, y_off = map(np.ravel, np.meshgrid(range(-1,2),range(-1,2)))
    min_disparity, max_disparity = 0, 4.0
    import itertools
    from matplotlib.patches import ConnectionPatch

    def fast_dp2(a, ly, lx, la,laz, ry, rx, ra,raz, occ_cost):
        M, N = la.size, ra.size
        l_pts, r_pts = la,ra

        dis = vec(l_pts)-r_pts        # corresponding dispairity value for array Edata
        dis_mask = np.logical_or(dis<0, dis>4)

        vl, vr = lim[ly, lx].astype('i'), rim[ry,rx].astype('i')
        Epos = np.abs(vec(laz)-raz)*50
        Edata = np.abs(vec(vl)-vr) + Epos
        Edata[dis_mask] = 65530   # negative disparity should not be considered


        vl = np.array([ lim[y+y_off, x+x_off] for y,x in zip(vec(ly),vec(lx)) ],'u1')
        vr = np.array([ rim[y+y_off, x+x_off] for y,x in zip(vec(ry),vec(rx)) ],'u1')
        result = np.full_like(l_pts,-1,'i2')
        scode = r"""
            inline float calcErr(uint8_t *a, uint8_t *b)
            {
                float sum = 0;
                for(size_t i=0; i<9; i++)
                    sum += std::fabs((float)*(a++) - (float)*(b++));
                return sum/9.0;
            }
            """
        code = r"""
            size_t M = Nl_pts[0];

            size_t N = Nr_pts[0];
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

            for (size_t m=1,md=0; m<=M; m++,md++)
                for(size_t n=1,nd=0; n<=N; n++,nd++ )
                {
                    #if 0
                        float disparity = L_PTS1(md) - R_PTS1(nd);
                        float Edata = (disparity<min_disparity or disparity> max_disparity)? 65530 : calcErr(&VL2(md,0), &VR2(nd,0));
                        float c1 = C(m-1, n-1) + Edata + EPOS2(md,nd);
                    #else
                        float c1 = C(m-1, n-1) + EDATA2(md,nd) + EPOS2(md,nd);
                    #endif

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
                   ['l_pts', 'r_pts', 'vl','vr','Edata', 'Epos','occ_cost','min_disparity','max_disparity','result'],
                    support_code = scode,
                    compiler='gcc',headers=['<chrono>','<cmath>'],
                    extra_compile_args=['-std=gnu++11 -msse2 -O3'],
                    verbose=1  )
        return result

    debug = True
    occ_cost = 10
    d_result = np.full_like(Icur, np.nan,'f')

    if debug:
        f = plt.figure(num='dpstereo')

        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        plt.tight_layout()

    for a in  range(ang_scaler.levels+1):#[65]:
        pr,pc = data[a],data_cur[a]

        if pc and pr:
            pc.sort()
            pc = zip(*pc)
            ly, lx = map(np.array, zip(*pc[2]))
            la,laz = np.array(pc[0]), np.array(pc[1])

            pr.sort()
            pr = zip(*pr)
            ry, rx = map(np.array,zip(*pr[2]))
            ra, raz = np.array(pr[0]), np.array(pr[1])

            if 0:
                '''1. get all matching error array and corresponding
                      dispairity value, use broacasting to get MxN array,
                      rows for sequential target points(in left image),
                      colums for candidate matching points (in right image)'''
                pl,pr = la,ra
                M, N = la.size, ra.size

                dis = vec(pl)-pr        # corresponding dispairity value for array Edata
                dis_mask = np.logical_or(dis<min_disparity, dis>max_disparity)

    #            Edata[dis_mask] = [ calcPatchScore(y,pl[x0],pr[x_can]) for x0,x_can in zip(*dis_mask.nonzero())]
                vl, vr = lim[ly, lx].astype('i'), rim[ry,rx].astype('i')
                Edata = np.abs(vec(vl)-vr)
                Edata[ dis_mask] = 65530   # negative disparity should not be considered


                '''2. Path generation. The state S[l,r] means there are l pixels in the
                      left image matched to r pixels in the right. There are 3 possible
                      ways to reach the state S[l,r]:
                        a. start from state S[l-1, r-1], match one more pair;
                        b. start from state S[l-1, r], skip a point in left image l+1->l;
                        c. start from state S[l, r-1], skip a point in right image r+1->r; '''
                C = np.zeros((M+1,N+1),'f')             # C[i,j] holds the cost to reach the point S[i,j]
                C[0,:] = np.arange(N+1)*occ_cost
                C[:,0] = np.arange(M+1)*occ_cost
                Best_rec = np.zeros_like(C,'u1')    # Best_rec[i,j] records the best way to reached it
                for m, n in itertools.product(range(1,M+1),range(1,N+1)):
                    c1 = C[m-1, n-1] + Edata[m-1, n-1]
                    c2 = C[m-1, n] + occ_cost
                    c3 = C[m, n-1] + occ_cost
                    Best_rec[m, n], C[m,n] = min(enumerate([c1,c2,c3]), key=lambda x:x[1])


                '''3. Backtrack. Our final goal is the point S[M,N], i.e. M left points
                      match to N right points '''
                m_idx, n_idx = M, N
                res = np.full_like(pl,-1,'i2')
                while m_idx!=0 and n_idx!=0:
                    choice = Best_rec[m_idx, n_idx]
                    if choice == 0:
                        ''' both points are matched'''
                        res[m_idx-1] = n_idx-1  # first 1-based
                        m_idx -= 1
                        n_idx -= 1
                    elif choice == 1:
                        ''' left points is skipped'''
                        m_idx -= 1
                    else:
                        n_idx -= 1

            else:
                res = fast_dp2(a, ly, lx, la,laz, ry, rx, ra,raz, occ_cost)
            if np.all(res==-1):
                continue

            lyc,lxc,lac,match_idx = ( np.compress(res!=-1, dump) for dump in [ly,lx,la,res] )
            rxm,rym,ram = ( np.take(dump, match_idx) for dump in [rx,ry,ra] )
#            d_result[lyc, lxc] = calcRange(ang_scaler(ram, isInvert=True),
#                                           ang_scaler(lac, isInvert=True))
#            assert(np.all(d_result[lyc, lxc]>0))
            if debug:
                al.clear(); ar.clear();ab.clear()
                al.imshow(Icur, interpolation='none');  #al.set_xlim([lx.min(),lx.max()]); al.set_ylim([ly.min(),ly.max()]);
                ar.imshow(Iref, interpolation='none');  #ar.set_xlim([rx.min(),rx.max()]); ar.set_ylim([ry.min(),ry.max()]);
                ar.plot(rx, ry,'b.')

                tx,ty = trueProj(rx, ry)
                al.plot(tx,ty,'b.')
                al.plot(lx, ly,'r.')

                ''' added line between matched pairs'''
                for x0,y0,x1,y1 in zip(lxc, lyc, tx, ty):
                    al.add_artist(ConnectionPatch(xyA=(x1,y1), xyB=(x0,y0),
                                          coordsA='data', coordsB='data',
                                          axesA=al, axesB=al))

                ab.plot(la, lim[ly, lx],'r*-')
                ab.plot(ra, rim[ry, rx],'b*-')
                ab.plot([lac, ram],
                        [lim[lyc, lxc],rim[rym, rxm]],'g--')

                plt.pause(0.01)
                plt.waitforbuttonpress()
        print a


#%%
    if 0:

        lim, rim = (Icur*255).astype('u1').copy(), (Iref*255).astype('u1').copy()
        def fast_dp(a, ly, lx, la, ry, rx, ra):
            res = np.full_like(la, -1, dtype='i8')
            scode = r"""
                #include <vector>
                #include <iostream>
                #include <fstream>
                #include <memory>
                #include <cmath>
                #include <limits>
                #include <cstddef>
                #include <cstdio>
                #include <chrono>
                #define CAP  512
                struct Cost
                {
                   float x;
                   int v;
                   std::vector<float> d_list;
                   std::vector<size_t> pre_idx;
                   std::vector<float> d_costs;

                   Cost(int x, int v)
                   :x(x), v(v) {}

                   Cost() {
                       d_list.reserve(CAP);
                       pre_idx.reserve(CAP);
                       d_costs.reserve(CAP);
                   }
                };
                typedef std::shared_ptr<Cost> ptrCost;
            """
            code = r"""

                auto start = std::chrono::system_clock::now();

                size_t M = Nla[0];
                size_t N = Nra[0];
                std::vector< ptrCost > states;
                states.reserve(M);

                // I. forward steps
                // special care for the first point, no regularization term
                size_t target_idx = 0;
                {
                    ptrCost p_current_state = std::make_shared<Cost>();
                    float x = LA1(target_idx);               // pixel angle
                    int v = LIM2(LY1(target_idx), LX1(target_idx));    // pixel intensity
                    p_current_state->x = x;
    std::cout<< "[" << x << "]:-----------------\n";
                    // 1. setup occlusion/no-match as the first candidate.
                    {
                        p_current_state->d_list.push_back(0);
                        p_current_state->d_costs.push_back(Penalty_Occ);    // constant cost for Occulsion assumption
                        p_current_state->pre_idx.push_back(0);
        std::cout<< '\t' << 0 <<'['<< Penalty_Occ <<"] \t" ;

                    }

                    // 2. calculate the N candiate paths with corresponding optimum cost
                    for (size_t candidate_idx=1; candidate_idx<N+1; candidate_idx++)
                    {
                        size_t candidate_data_idx = candidate_idx-1;
                        float x_can = RA1(candidate_data_idx);
                        float disparity = x-x_can;

                        // discard points with negative or too small disparity(i.e too far away).
                        // Terminate the loop once the condition breaks, assuming the point-list
                        // are already ordered in x coordinate,
                        if (disparity>max_disparity)
                            continue;
                        else if (disparity<min_disparity)
                            break;

                        // 2a. matching cost for this disparity value
                        int v_can = RIM2(RY1(candidate_data_idx), RX1(candidate_data_idx));
                        float Edata = std::fabs(v - v_can);
    std::cout<< candidate_idx <<'['<< Edata <<"] \t" ;

                        p_current_state->d_list.push_back(disparity);
                        p_current_state->d_costs.push_back(Edata);
                        p_current_state->pre_idx.push_back(candidate_idx);
                    }
    std::cout<< std::endl;
                    // 3. done with this target pixel and move on the next
                    states.push_back(p_current_state);
                }

                for (target_idx=1; target_idx<M; target_idx++) {
                    ptrCost p_last_state = states.back();
                    ptrCost p_current_state = std::make_shared<Cost>();

                    float x = LA1(target_idx);               // pixel angle
                    int v = LIM2(LY1(target_idx), LX1(target_idx));    // pixel intensity
                    p_current_state->x = x;
                    p_current_state->v = v;
    std::cout<< "[" << x << "]:-----------------\n";
                    /* 1. setup occlusion/no-match as the first candidate, if there are no
                       valid candiate matching points, then this will be the only option.*/
                    {
                        float min_Edata_last = p_last_state->d_costs[0];
                        size_t min_E_idx = 0;

                        for (size_t k=1; k<p_last_state->d_list.size(); k++)
                        {
                            float Edata_last = p_last_state->d_costs[k];  // cummulated path cost
                            if(Edata_last < min_Edata_last){
                                min_Edata_last = Edata_last;
                                min_E_idx = k;
                            }
                        }
                        p_current_state->d_list.push_back(0);
                        p_current_state->d_costs.push_back(min_Edata_last + Penalty_Occ); // constant cost for Occulsion assumption,no penalty for state changes to occulsion
                        p_current_state->pre_idx.push_back(min_E_idx);

                    }

                    /* 2. calculate the N-1 candiate paths with corresponding optimum cost.*/
                    for (size_t candidate_idx=1; candidate_idx<N+1; candidate_idx++) {
                        size_t candidate_data_idx = candidate_idx-1;
                        float x_can = RA1(candidate_data_idx);
                        float disparity = x-x_can;

                        /* discard points with negative or too small disparity(i.e too far away).
                           Assuming the point-list are ordered in x coordinate, terminate the
                           loop once the condition breaks */
                        if (disparity>max_disparity)
                            continue;
                        else if (disparity<min_disparity)
                            break;


                        /* 2a. matching cost for this disparity value*/
                        int v_can = RIM2(RY1(candidate_data_idx), RX1(candidate_data_idx));
                        float Edata = std::fabs(v - v_can);
        std::cout<< "\t disparity:" << disparity << '-' << "cost:" << Edata << std::endl;
        std::cout<< "\t\t";

                        /* 2b. choose a optimum path from state(every possible last disparity) to this disparity */
                        float min_Etotal_last = p_last_state->d_costs[0];   // again, the fisrt one for occulsion/no-match
                        size_t min_E_idx = 0;                  // occulsion = default
        std::cout<< 0<< '['<< min_Etotal_last<<"]\t";

                        for (size_t k=1; k<p_last_state->d_list.size(); k++) {
                            float disparity_last = p_last_state->d_list[k];
                            float xdiff = std::abs(x - p_last_state->x);
                            float diff = disparity_last - disparity;
                            float Ereg = (diff < 0)?   1e6 :        // ordering constraint
                                         (diff < 0.05)?  0 :        // no jump no penalty
                                         (diff < 2)?  Penalty1 : Penalty2;  // large jump
                            float Etotal = p_last_state->d_costs[k] + Ereg/xdiff;  // cummulated path cost
            std::cout<< disparity_last<< '['<< Etotal<<"]\t";
                            if(Etotal < min_Etotal_last) {
                                min_Etotal_last = Etotal;
                                min_E_idx = k;
                            }
                        }
            std::cout<< "\t min:No." << min_E_idx << '('<<p_last_state->d_list[min_E_idx] <<','<< min_Etotal_last << ')' << std::endl;

                        // 2c. only keep the optimal path from last state to this disparity value
                        p_current_state->d_list.push_back(disparity);
                        p_current_state->d_costs.push_back(Edata + min_Etotal_last);
                        p_current_state->pre_idx.push_back(min_E_idx);

                    }
    std::cout<< std::endl;
                    /* 3. done with this target pixel and move on the next*/
                    states.push_back(p_current_state);
                }

                /* II. backtrace step
                   1. best score in the final step. */
                ptrCost p_last_state = states.back();
                float E_min = std::numeric_limits<float>::max();
                size_t min_path_idx = 0;
                for (size_t candidate_idx = 0; candidate_idx < p_last_state->d_costs.size(); candidate_idx++)
                {
                    float cost = p_last_state->d_costs[candidate_idx];
                    if(E_min > cost)
                    {
                        E_min = cost;
                        min_path_idx = candidate_idx;
                    }
                }

                // 2. preceding state
                for (size_t state_idx=M-1; state_idx+1>0; state_idx--)
                {
                    RES1(state_idx) = min_path_idx; //states[state_idx]->d_list[min_path_idx];

                    // corresponding path from previous step
                    min_path_idx = states[state_idx]->pre_idx[min_path_idx];
                }

                auto duration = std::chrono::duration<double>
                                (std::chrono::system_clock::now() - start);
            """
    #        import timeit
    #        start = timeit.default_timer()
            Penalty1 = 10.0
            Penalty2 = 30.0
            Penalty_Occ = 150.
            min_disparity = 0.
            max_disparity = 140.

            weave.inline(code,
                       ['lim', 'rim', 'ly', 'lx', 'la', 'ry', 'rx', 'ra', 'res',
                       'Penalty1','Penalty2','Penalty_Occ','min_disparity','max_disparity'],
                        support_code = scode,
                        compiler='gcc',
                        extra_compile_args=['-std=gnu++11 -O3'],
                        verbose=2  )
    #        end = timeit.default_timer()
    #        print 'fps:',1.0/(end - start)
    #        print res
            return res

        debug = True
        d_result = np.full_like(Icur, -1)

        if debug:
            f = plt.figure(num='dpstereo')
            gs = plt.GridSpec(2,2)
            al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
            ab = f.add_subplot(gs[1,:])
            ab.autoscale()

        for a in [65]: # range(ang_scaler.levels+1):
            pr,pc = data[a],data_cur[a]

            if pc and pr:
                pc.sort()
                pc = zip(*pc)
                ly, lx = map(np.array, zip(*pc[2]))
                la = np.array(pc[0])

                pr.sort()
                pr = zip(*pr)
                ry, rx = map(np.array,zip(*pr[2]))
                ra = np.array(pr[0])

                res = fast_dp(a, ly, lx, la, ry, rx, ra)
                if np.all(res==0):
                    continue

                lyc,lxc,lac,match_idx = ( np.compress(res!=0, dump) for dump in [ly,lx,la,res] )
                rxm,rym,ram = ( np.take(dump, match_idx-1) for dump in [rx,ry,ra] )
                d_result[lyc, lxc] = calcRange(ang_scaler(ram, isInvert=True),
                                               ang_scaler(lac, isInvert=True))
                if debug:
                    al.clear(); ar.clear();ab.clear()
                    al.imshow(Icur); ar.imshow(Iref)
                    al.plot(lx, ly,'r.')
                    ar.plot(rx, ry,'b.')
                    tx,ty = trueProj(rx, ry)
                    al.plot(tx,ty,'g.')

                    ab.plot(la, lim[ly, lx],'r*-')
                    ab.plot(ra, rim[ry, rx],'b*-')
                    ab.plot([lac, ram],
                            [lim[lyc, lxc],rim[rym, rxm]],'g-')

                    plt.pause(0.01)
    #                plt.waitforbuttonpress()
            print a

#%%

    l = np.array([Tcr[2]*(u-cx) -fx*Tcr[0],
                  Tcr[2]*(v-cy) -fy*Tcr[1]])
