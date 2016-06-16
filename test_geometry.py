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
#%%


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
    refid, curid = 0,8
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

    if 0:
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

#%% demo: points on the scanline
    if 0:
        def trueProj(x, y, G=cGr):
            p0 = np.array([(x-cx)/fx, (y-cy)/fy, np.ones(len(x))])*Z[y,x]
            p =  K.dot(G[0:3,0:3].dot(p0)+G[0:3,3][:,np.newaxis])
            p /= p[2]
            return p[0],p[1]

        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        ar,ac = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        for a in range(65,ang_scaler.levels+1):
            ac.clear(); ar.clear();ab.clear()
            ar.imshow(Iref); ac.imshow(Icur)
            pr,pc = data[a],data_cur[a]

            if pc:
                pc.sort()
                pc = zip(*pc)
                y, x = zip(*pc[2])
                ab.plot(pc[0],Icur[y,x]*255,'r*-')
                ac.plot(x, y,'r.')

            if pr:
                pr.sort()
                pr = zip(*pr)

                y, x = zip(*pr[2])
                ab.plot(pr[0],Iref[y,x]*255,'b*-')
                ar.plot(x, y,'b.')

                tx,ty = trueProj(np.array(x), np.array(y))
                ac.plot(tx,ty,'g.')

            plt.pause(0.01)
            plt.waitforbuttonpress()
#%% exam the depth calculation

    positive_range = lambda x: x if x>0 else x+2*np.pi
    def calcAngle(x, y, G=None):
        p0 = np.array([(x-cx)/fx, (y-cy)/fy, 1.0])
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

#%%
    from scipy import weave

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


