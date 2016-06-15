#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:26:41 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
pis = plt.imshow
pf = plt.figure


def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage
from vtk_visualizer import *

if __name__ == "__main__":
    imleft = plt.imread('./left0000.jpg')[:,:,0].astype('f').copy()
    imright = plt.imread('./right0000.jpg')[:,:,0].astype('f').copy()
    h,w = imleft.shape[:2]

    normalize = lambda x:x/np.linalg.norm(x)
    plt.ion()
    #%% get good pixels
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)


    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([imleft, imright]):
#        d = calcGradient(im)
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im,1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,80)
        dI.append( d.copy() )
        u, v = np.meshgrid(range(w),range(h))
        pixel_mask = reduce(np.logical_and,[valid_mask, u>1, v>1, u<w-2, v<h-2])
        px.append(u[pixel_mask].copy())
        py.append(v[pixel_mask].copy())
        pvm.append(valid_mask.copy())

    pis(valid_mask)
    cmin = np.minimum(dI[0].min(), dI[1].min())
    dI[1] += -cmin
    dI[0] += -cmin
    scale = int(np.ceil(np.maximum(dI[0].max(), dI[1].max())))

    for vm,d in zip(pvm, dI):
        dt = np.round(d).astype('int')
        pcolor.append(dt[vm])


#%% construct database
    data = [[] for _ in range(h)]
    for x,y in zip(px[1], py[1]):
        """put pixels into bins base on their color"""
        data[y].append(x)

    data_l = [[] for _ in range(h)]
    for x,y in zip(px[0], py[0]):
        """put pixels into bins base on their color"""
        data_l[y].append(x)

#%%
    from scipy import weave

    lim, rim = imleft.astype('u1').copy(),imright.astype('u1').copy()
    def fast_dp(y, l_pts, r_pts):

        res = np.full_like(l_pts, -1)

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
               int x;
               int v;
               std::vector<int> d_list;
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
            const float Penalty1 = 10.0;
            const float Penalty2 = 30.0;
            const float Penalty_Occ = 20;
            const int min_disparity = 10;
            const int max_disparity = 160;

            auto start = std::chrono::system_clock::now();

            size_t M = Nl_pts[0];
            size_t N = Nr_pts[0];
            std::vector< ptrCost > states;
            states.reserve(M);

            // I. forward steps
            // special care for the first point, no regularization term
            size_t target_idx = 0;
            {
                ptrCost p_current_state = std::make_shared<Cost>();
                int x = L_PTS1(target_idx);               // pixel coordinate
                float v = LIM2(y, x);    // pixel intensity
                p_current_state->x = x;
                //std::cout<< "[" << x << "]:-----------------\n";

                // 1. setup occlusion/no-match as the first candidate.
                {
                    p_current_state->d_list.push_back(0);
                    p_current_state->d_costs.push_back(Penalty_Occ);    // constant cost for Occulsion assumption
                    p_current_state->pre_idx.push_back(0);
                    //std::cout<< '\t' << 0 <<'['<< Penalty_Occ <<"] \t" ;
                }

                // 2. calculate the N candiate paths with corresponding optimum cost
                for (size_t candidate_idx=1; candidate_idx<N+1; candidate_idx++)
                {
                    int x_can = R_PTS1(candidate_idx-1);
                    int disparity = x-x_can;

                    // discard points with negative or too small disparity(i.e too far away).
                    // Terminate the loop once the condition breaks, assuming the point-list
                    // are already ordered in x coordinate,
                    if (disparity>max_disparity)
                        continue;
                    else if (disparity<min_disparity)
                        break;

                    // 2a. matching cost for this disparity value
                    int v_can = RIM2(y, x_can);
                    float Edata = std::fabs(v - v_can);
                    //std::cout<< candidate_idx <<'['<< Edata <<"] \t" ;

                    p_current_state->d_list.push_back(disparity);
                    p_current_state->d_costs.push_back(Edata);
                    p_current_state->pre_idx.push_back(candidate_idx);
                }
                //std::cout<< std::endl;

                // 3. done with this target pixel and move on the next
                states.push_back(p_current_state);
            }

            for (target_idx=1; target_idx<M; target_idx++) {
                ptrCost p_last_state = states.back();
                ptrCost p_current_state = std::make_shared<Cost>();

                int x = L_PTS1(target_idx);               // pixel coordinate
                float v = LIM2(y, x);    // pixel intensity
                p_current_state->x = x;
                p_current_state->v = v;

                // 1. setup occlusion/no-match as the first candidate, if there are no
                // valid candiate matching points, then this will be the only option.
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

                // 2. calculate the N-1 candiate paths with corresponding optimum cost
                for (size_t candidate_idx=1; candidate_idx<N+1; candidate_idx++) {
                    int x_can = R_PTS1(candidate_idx-1);
                    int disparity = x-x_can;

                    // discard points with negative or too small disparity(i.e too far away).
                    // Assuming the point-list are ordered in x coordinate, terminate the
                    // loop once the condition breaks
                    if (disparity>max_disparity)
                        continue;
                    else if (disparity<min_disparity)
                        break;


                    // 2a. matching cost for this disparity value
                    int v_can = RIM2(y, x_can);
                    float Edata = std::fabs(v - v_can);
                    //std::cout<< "\t disparity:" << disparity << '-' << "cost:" << Edata << std::endl;
                    //std::cout<< "\t\t";

                    /* 2b. choose a optimum path from state(every possible last disparity) to this disparity */
                    float min_Etotal_last = p_last_state->d_costs[0];   // again, the fisrt one for occulsion/no-match
                    size_t min_E_idx = 0;                  // occulsion = default
                    //std::cout<< 0<< '['<< min_Etotal_last<<"]\t";
                    for (size_t k=1; k<p_last_state->d_list.size(); k++) {
                        int disparity_last = p_last_state->d_list[k];
                        int xdiff = std::abs(x - p_last_state->x);
                        int diff = std::abs(disparity - disparity_last);
                        float Ereg = (diff == 0)? 0 :                  // disparity jump penalty
                                     (diff  < 2)? Penalty1 : Penalty2;
                        float Etotal = p_last_state->d_costs[k] + Ereg;  // cummulated path cost
                        //std::cout<< disparity_last<< '['<< Etotal<<"]\t";
                        if(Etotal < min_Etotal_last) {
                            min_Etotal_last = Etotal;
                            min_E_idx = k;
                        }
                    }
                    //std::cout<< "\t min:No." << min_E_idx << '('<<p_last_state->d_list[min_E_idx] <<','<< min_Etotal_last << ')' << std::endl;

                    // 2c. only keep the optimal path from last state to this disparity value
                    p_current_state->d_list.push_back(disparity);
                    p_current_state->d_costs.push_back(Edata + min_Etotal_last);
                    p_current_state->pre_idx.push_back(min_E_idx);

                }
                //std::cout<< std::endl;
                // 3. done with this target pixel and move on the next
                states.push_back(p_current_state);
            }

            // II. backtrace step
            // 1. best score in the final step
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

            // 2. predecessor tracing
            for (size_t state_idx=M-1; state_idx+1>0; state_idx--)
            {
                RES1(state_idx) = states[state_idx]->d_list[min_path_idx];

                // corresponding path from previous step
                min_path_idx = states[state_idx]->pre_idx[min_path_idx];
            }

            auto duration = std::chrono::duration<double>
                            (std::chrono::system_clock::now() - start);
            //std::cout <<"runtime:" <<duration.count() << "s" <<std::endl;

        """

        test_code = r"""
            size_t M = Nl_pts[0];
            size_t N = Nr_pts[0];
            std::vector< ptrCost > states;
            for (int target_idx=0; target_idx<M; target_idx++)
            {
                int x = L_PTS1(target_idx);               // pixel coordinate
                float v = LIM2(y, x);    // pixel intensity

                RES1(target_idx) = v;
            }
        """
#        import timeit
#        start = timeit.default_timer()
        weave.inline(code,
                   ['lim', 'rim', 'l_pts', 'r_pts', 'y', 'res'],
                    support_code = scode,
                    compiler='gcc',
                    extra_compile_args=['-std=gnu++11 -O3'],
                    verbose=2  )
#        end = timeit.default_timer()
#        print 'fps:',1.0/(end - start)
#        print res
        return res

    debug = True
    d_result = np.full_like(imleft, -1)

    if debug:
        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
    for y in range(h):
        if data_l[y] and data[y]:
            l_pts, r_pts = np.array(data_l[y]), np.array(data[y])
            res = fast_dp(y, l_pts, r_pts)
            d_result[y, l_pts] = res
            if debug:
                al.clear(); ar.clear();ab.clear()
                al.imshow(imleft); ar.imshow(imright)
                ab.plot(l_pts, lim[y, l_pts],'r*-')
                ab.plot(r_pts, rim[y, r_pts],'b*-')
                pts0 = l_pts[res!=0]
                pts1 = pts0 - res[res!=0]
                ab.plot([pts0,         pts1],
                        [lim[y, pts0], rim[y, pts1]],'g-')
                plt.pause(0.01)
                plt.waitforbuttonpress()
        print y

    v,u = np.where(np.logical_and(d_result >10,d_result <150))
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(imleft[v,u],(3,1))]).T)


#%% DP debug class
    if 0:

        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        vec = lambda x:np.reshape(x,(-1,1))

        class Cost:
            def __init__(self, x, v, d_list, pre_idx=None, d_costs=None):
                self.x = x
                self.v = v
                self.d_list = d_list
                self.pre_idx = pre_idx
                self.d_costs = d_costs

        debug = True
        d_result = np.full_like(imleft, -1)

        x_off, y_off = np.meshgrid(range(-1,2),range(-1,2))
        def calcPatchScore(y, x0, x_can):
            dl =  imleft[y+y_off,    x0+x_off]
            dr = imright[y+y_off, x_can+x_off]
    #        ncc = 1-np.dot(normalize(dl.ravel()-dl.mean()),
    #                     normalize(dr.ravel()-dr.mean()))
            ssd = np.linalg.norm((dl-dr).ravel())/9
            return ssd

        for y in range(120,h):
            if debug:
                al.clear(); ar.clear();ab.clear()
                al.imshow(imleft); ar.imshow(imright)
            pl,pr = [],[]
            '''obtain and plot the row data'''
            for p in data_l[y]:
                if debug:
                    al.plot(p, y,'r.',ms=3)
                pl.append((p, imleft[y, p], dI[0][y, p]))
            for p in data[y]:
                if debug:
                    ar.plot(p, y,'b.',ms=3)
                pr.append((p, imright[y,p], dI[1][y, p]))
            if pl and pr:

                pl = zip(*pl)
                pr = zip(*pr)

                M = len(pl[0])

                dis = vec(pl[0])-pr[0]        # corresponding dispairity value for array Edata

                vl = np.array(pl[1],'i')
                vr = np.array(pr[1],'i')

                States = []
                for p_idx in xrange(M):
                    if debug:
                        ab.clear()
                        ab.plot(pr[0],vr,'b*-')
                        ab.plot(pl[0][:p_idx+1], vl[:p_idx+1],'r*-')

                    mask =  dis[p_idx] > 10       # maximum disparity
                    d_idx = np.where(mask)[0]
                    d_list = np.hstack([0, dis[p_idx, d_idx]])        # 0 for occluded

                    x = pl[0][p_idx]
                    n = d_list.size

                    '''matching cost for each possible dispairty value of x2 '''
    #                Edata = np.hstack([10, vl[p_idx] - vr[d_idx]])**2
    #                Edata = np.hstack([30, np.abs(vl[p_idx] - vr[d_idx])])      # 10 for occlusion cost
                    cost = [10]+[calcPatchScore(y,x, x-d_can) for d_can in d_list[1:]]
                    Edata = np.array(cost)

                    if len(States) == 0:
                        cur_state = Cost(x, vl[p_idx], d_list, range(n), Edata)
                        States.append(cur_state)
                    else:
                        '''For each value of x2 determine the cost with each value of x1 '''
                        last_state = States[-1]
                        weight = 1/(x - last_state.x)
                        jumps = np.abs(vec(d_list) - last_state.d_list)
                        Ereg = np.where(jumps<2, 1, 5)
                        Ereg[jumps==0] = 0
                        Ereg[0] = 0        # from non-occluded to occluded
                        Ereg[:,0] = 0      # from occluded to non-occluded
                        Etotal = last_state.d_costs + 100*Ereg   # matching cost + jump cost

                        '''choose the n best path'''
                        best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                        total_cost = Edata + Etotal[range(n), best_idx]

                        cur_state = Cost(x, vl[p_idx], d_list, best_idx, total_cost)
                        States.append(cur_state)

                '''backtrace to readout the optimum'''
                res = np.nanargmin(cur_state.d_costs)                # get the final best
                for j in reversed(xrange(len(States))):
                    '''given the state of parent step, lookup the waypoint to it'''
                    if res != 0:
                        if debug:
                             ab.plot([States[j].x,  pr[0][res-1]],
                                     [States[j].v,     vr[res-1]],'g-')
                        d_result[y,States[j].x] = States[j].x - pr[0][res-1]
                    res = States[j].pre_idx[res]

                if debug:
                    plt.pause(0.01)
                    plt.waitforbuttonpress()
            print y


        v,u = np.where(d_result >10)
        p3d = np.vstack([(u-0.5*w)/435.016,
                         (v-0.5*h)/435.016,
                         np.ones(u.shape[0])
                         ]).astype('f')/d_result[v,u]*0.119554
        plotxyzrgb(np.vstack([p3d,np.tile(imleft[v,u],(3,1))]).T)
        exit(0)



#%% DP debug numpy
        f = plt.figure(num='query')
        gs = plt.GridSpec(2,2)
        al,ar = f.add_subplot(gs[0,0]),f.add_subplot(gs[0,1])
        ab = f.add_subplot(gs[1,:])
        ab.autoscale()
        vec = lambda x:np.reshape(x,(-1,1))
        for y in range(125,h):
            al.clear(); ar.clear();ab.clear()
            al.imshow(imleft); ar.imshow(imright)
            pl,pr = [],[]
            '''obtain and plot the row data'''
            for p in data_l[y]:
                al.plot(p, y,'r.',ms=3)
                pl.append((p, imleft[y, p], dI[0][y, p]))
            for p in data[y]:
                ar.plot(p, y,'b.',ms=3)
                pr.append((p, imright[y,p], dI[1][y, p]))
            if pl and pr:
                pl.sort(key=lambda x:x[0])
                pr.sort(key=lambda x:x[0])
                pl = zip(*pl)
                pr = zip(*pr)
                '''DP 1st step: get all matching error array and corresponding
                   dispairity value, use broacasting to get MxN array,
                   rows for sequential target points(in left image),
                   colums for candidate matching points (in right image)'''
                M,N = (len(pl[0]), len(pr[1]))
                vl = np.array(pl[0])    # x coordinates
                vr = np.array(pr[0])
                dis = vec(vl)-vr        # corresponding dispairity value for array Edata

    #            vl = np.array(pl[2])
    #            vr = np.array(pr[2])
    #            Edata = 0.5*(vec(vl)-vr)**2
                vl = np.array(pl[1])
                vr = np.array(pr[1])
                Edata = (vec(vl)-vr)**2
                Edata[dis<0] = np.inf   # negative disparity should not be considered

                '''DP 2nd step: calculate regularise term'''
                S = np.empty_like(Edata)
                Best_rec = np.empty_like(Edata,'i8')
                S[0] = Edata[0]
                Best_rec[0] = range(N)
                for i in xrange(1, M):
                    ab.clear()
                    ab.plot(pr[0],vr,'b*-')
                    ab.plot(pl[0][:i+1],vl[:i+1],'r*-')

                    ''' non-smooth punishment '''
                    Ereg = (vec(dis[i]) - dis[i-1])**2/(pl[0][i]-pl[0][i-1])   # NxN array, costs for dispairity jumps from last point to this point
                    Etotal = S[i-1] + 10*Ereg           # matching cost + jump cost
                    best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                    S[i] = Edata[i] + Etotal[range(N),best_idx]
                    Best_rec[i] = best_idx

                    '''DP 3rd step: backtrace to readout the optimum'''
                    res = np.nanargmin(S[i]) # get the final best
                    ab.plot([pl[0][i],  pr[0][res]],
                            [vl[i],vr[res]],'g-')
                    for j in xrange(i-1, -1, -1):
                        '''given the state of parent step, lookup the waypoint to it'''
                        res = Best_rec[j+1][res]
                        ab.plot([pl[0][j],  pr[0][res]],
                                [vl[j],vr[res]],'-')

                    plt.pause(0.01)
                    plt.waitforbuttonpress()

#%% fast DP
        d_result = np.full_like(imleft, -1)
        vec = lambda x:np.reshape(x,(-1,1))
        def dpProcess():
            for y in range(h):
                pl,pr = [],[]
                '''obtain and plot the row data'''
                for p in data_l[y]:
                    pl.append((p, imleft[y, p], dI[0][y, p]))

                for p in data[y]:
                    pr.append((p, imright[y,p], dI[1][y, p]))

                if pl and pr:
                    pl.sort(key=lambda x:x[0])
                    pr.sort(key=lambda x:x[0])
                    pl = zip(*pl)
                    pr = zip(*pr)

                    '''DP 1st step: get all matching error array and corresponding
                       dispairity value, use broacasting to get MxN array,
                       rows for sequential target points(in left image),
                       colums for candidate matching points (in right image)'''
                    M,N = (len(pl[0]), len(pr[1]))
                    vl = np.array(pl[0])    # x coordinates
                    vr = np.array(pr[0])
                    dis = vec(vl)-vr        # corresponding dispairity value for array Edata

                    vl = np.array(pl[2])
                    vr = np.array(pr[2])
                    Edata = 0.5*(vec(vl)-vr)**2
                    vl = np.array(pl[1])
                    vr = np.array(pr[1])
                    Edata += 0.5*(vec(vl)-vr)**2
                    Edata[dis<0] = np.inf   # negative disparity should not be considered

                    '''DP 2nd step: calculate regularise term'''
                    S = np.empty_like(Edata)
                    Best_rec = np.empty_like(Edata,'i8')
                    S[0] = Edata[0]
                    Best_rec[0] = range(N)
                    for i in xrange(1, M):
                        ''' non-smooth punishment '''
                        Ereg = (vec(dis[i]) - dis[i-1])**2/(pl[0][i]-pl[0][i-1])   # NxN array, costs for dispairity jumps from last point to this point
                        Etotal = S[i-1] + 10*Ereg           # matching cost + jump cost
                        best_idx = np.nanargmin(Etotal, axis=1)   # Nx1 array, For each value of x2 determine the best value of x1
                        S[i] = Edata[i] + Etotal[range(N),best_idx] #
                        Best_rec[i] = best_idx

                    '''DP 3rd step: backtrace to readout the optimum'''
                    res = np.nanargmin(S[i])        # result of final step
                    d_result[y,pl[0][i]] = pl[0][i]-pr[0][res]
                    for j in xrange(i-1, -1, -1):
                        '''given the state of parent step, lookup the waypoint to it'''
                        res = Best_rec[j+1][res]
                        d_result[y,pl[0][j]] = pl[0][j]-pr[0][res]
                print y
        dpProcess()





 #%% one-by-one matching
        x_off, y_off = np.meshgrid(range(-2,3),range(-2,3))
        def calcPatchScore(y, x0, x_can):
            dl =  imleft[y+y_off,    x0+x_off]
            dr = imright[y+y_off, x_can+x_off]
    #        ncc = 1+np.dot(normalize(dl.ravel()-dl.mean()),
    #                     normalize(dr.ravel()-dr.mean()))
            ssd = np.linalg.norm((dl-dr).ravel())/25.0
            return ssd
        from operator import itemgetter

        def test_calcPatchScore():
            x0, y = (786, 205)
            score = [calcPatchScore(y, x0, x_can) for x_can in range(2,w-2)]
            plt.subplot(2,1,1)
            plt.plot(score)
            plt.subplot(2,1,2)
            plt.plot( imleft[y,:],'r')
            plt.plot(imright[y,:],'b')
            plt.vlines(x0,0,255,'r')
            plt.hlines(imleft[y,x0],2,w-2)
            result, element = min(enumerate(score), key=itemgetter(1))
            plt.vlines(result+2,0,255,'g',linewidths=3)



    #    x,y,c = px[0][1000], py[0][1000], pcolor[0][1000]

        start = 0
        debug = False
        d_result = np.full_like(imleft, 0)
        if debug:
            f,(a0,a1) = plt.subplots(2,1,num='query')
            fi, b0 = plt.subplots(1,1,num='i')
            b0.imshow(imleft)

        for x,y,c in zip(px[0][start:], py[0][start:], pcolor[0][start:]):
            if debug:
                pf(fi.number)
                b0.plot(x,y,'r.')
                pf(f.number)
                a0.clear()
                a1.clear()
                '''show the data slice'''
                a0.plot(dI[0][y,:],'r')
                a0.plot(dI[1][y,:],'b')
                ''' plot target point'''
                a0.vlines(x,0,scale,'r')
                a1.plot( imleft[y,:],'r')
                a1.plot(imright[y,:],'b')
                a1.vlines(x,0,255,'r')
            min_score = np.inf
            '''consider all points in neighbouring gradient-level as candiates'''
            for offset in [0,1,-1,2,-2]:
                plist = data[y][np.clip(c+offset, 0, scale)]
                for cp_x in plist:

                    ''' discard points in negative or out-of-range disparity'''
                    dis = x-cp_x
                    if dis < 15 or dis > 120 :
                        continue

                    ''' discard points different to much in intensity'''
                    if np.abs(imleft[y,x]-imright[y,cp_x]) > 15 :
                        continue

                    if debug:
                        a0.vlines(cp_x,0,scale,'b','dashed')
                        a1.vlines(cp_x,0,255,'b','dashed')

                    ''' discard points with poor score'''
                    score = calcPatchScore(y,x,cp_x)
                    if score > 3:
                        continue
                    ''' only keep the best match'''
                    if score < min_score:
                        min_score = score
                        d_result[y,x] = x-cp_x

            if debug:
                if min_score != np.inf:
                    print 'depth at %d with score:%f' % (cp_x, min_score)
                    a0.vlines(x-d_result[y,x],0,scale,'g',linewidths=3)
                    a1.vlines(x-d_result[y,x],0,255,'g',linewidths=3)

                plt.pause(0.01)
                plt.waitforbuttonpress()
        pis(d_result)

