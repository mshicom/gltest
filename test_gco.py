#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:26:41 2016

@author: nubot
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
pis = plt.imshow
pf = plt.figure

from pygco import *
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage
from vtk_visualizer import *
from scipy import weave
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse
def vec(*arg):
    return np.reshape(arg,(-1,1))

#%%
if __name__ == "__main__":
    lim = plt.imread('./left0000.jpg')[:,:,0].astype('i4').copy()
    rim = plt.imread('./right0000.jpg')[:,:,0].astype('i4').copy()
    h,w = lim.shape[:2]

    normalize = lambda x:x/np.linalg.norm(x)
    plt.ion()
    # get good pixels
    def calcGradient(im):
        dx,dy = np.gradient(im)
        return np.sqrt(dx**2+dy**2)

    dI,px,py,pcolor,pvm = [],[],[],[],[]
    for i,im in enumerate([lim, rim]):
#        d = calcGradient(im)
        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im.astype('f'),1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,80)
        dI.append( d.copy() )
        u, v = np.meshgrid(range(w),range(h))
        pixel_mask = reduce(np.logical_and,[valid_mask, u>1, v>1, u<w-2, v<h-2])
        px.append(u[pixel_mask].copy())
        py.append(v[pixel_mask].copy())
        pvm.append(pixel_mask.copy())

    pis(sim(lim,rim))

    for vm,d in zip(pvm, dI):
        dt = np.round(d).astype('int')
        pcolor.append(dt[vm])

#% construct database
    data = [[] for _ in range(h)]
    for x,y in zip(px[1], py[1]):
        """put pixels into bins base on their color"""
        data[y].append((x, 0, (y,x)))

    data_cur = [[] for _ in range(h)]
    for x,y in zip(px[0], py[0]):
        """put pixels into bins base on their color"""
        data_cur[y].append((x, 0, (y,x)))
    d_result = np.full_like(lim, -1)

#%% edge-only stereo
    from scipy import sparse
    py, px = pvm[0].nonzero()
    node_cnt = px.size
    '''1. setup neighbors'''
    def makeEdges(mask_image):
        py, px = mask_image.nonzero()
        node_cnt = px.size
        id_LUT = np.empty_like(mask_image, 'i4')
        id_LUT[mask_image] = range(node_cnt)      # lookup-table of index number for valid pixels
        edges = []
        for p_id, p_x,p_y in zip(range(node_cnt), px, py):
            degree = 0
            '''diagonal edge'''
            if 0:
                if mask_image[p_y-1, p_x+1]:
                    edges.append([p_id, id_LUT[p_y-1, p_x+1]]); degree += 1
                if mask_image[p_y-1, p_x-1]:
                    edges.append([p_id, id_LUT[p_y-1, p_x-1]]); degree += 1

            if mask_image[p_y-1, p_x]:
                edges.append([p_id, id_LUT[p_y-1,  p_x ]]); degree += 1
            if mask_image[p_y, p_x-1]:
                edges.append([p_id, id_LUT[ p_y,  p_x-1]]); degree += 1

        edge_cnt = len(edges)
        row_ind = np.tile(np.arange(edge_cnt)[:,np.newaxis],2).ravel()
        col_ind = np.array(edges).ravel()
        data = np.tile(np.array([1,-1]), edge_cnt)
        incidence_matrix = sparse.csr_matrix((data,(row_ind,col_ind)), (len(edges),node_cnt),'i4')

        return np.array(edges,'i4'), incidence_matrix      # edges array, each row represent an edge
    edges,incidence_matrix = makeEdges(pvm[0])
#%%
    '''2. setup matching cost'''
    max_disp = 150
    min_disp = 10

    unary_cost = np.empty([node_cnt, max_disp+1],'i4')       # shape= n_vertices x n_disps
    y_ind, x_ind = vec(py) , vec(px)-np.arange(min_disp, max_disp+1)
    unary_cost = np.abs(vec(lim[py, px]) - rim[y_ind, x_ind])  # shape=(n_vertices, n_labels)
    unary_cost[x_ind<0] = 100

    '''3. setup smoothness cost'''
    if 0:
        pairwise_cost = -5*np.eye(max_disp+1, dtype='i4')+5        # shape=(n_labels, n_labels)
    else:
        dx, dy = np.ogrid[:(max_disp-min_disp)+1, :(max_disp-min_disp)+1]
        pairwise_cost = np.abs(dx - dy).astype('i4').copy("C")

    '''4. do the calculation'''
    d_cut = cut_from_graph(edges, unary_cost, 5*pairwise_cost,  n_iter=5)

    '''5. plot '''
    d_result = np.full_like(lim, -1)
    d_result[pvm[0]] = d_cut+10

    v,u = np.where(reduce(np.logical_and, [d_result>10, d_result<150, pvm[0]]))
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(lim[v,u],(3,1))]).T)


#%%
    def warp_d(I, px, py, pv, d):
        """
        linearize the error fuction arround the current depth estimate
        """

        Iw = scipy.ndimage.map_coordinates(I, np.vstack([py,px-d]), order=1, mode='nearest')
        Iwf = scipy.ndimage.map_coordinates(I, np.vstack([py,px-(d+1)]), order=1, mode='nearest')
        Iwb = scipy.ndimage.map_coordinates(I, np.vstack([py,px-(d-1)]), order=1, mode='nearest')
        It = Iw - pv       # 'time' derivative'
        Ig = (Iwf - Iwb)/2
        return  It, Ig, Iw

    def huber_function(data, epsilon):
        data = np.abs(data)
        return np.where(data>=epsilon, data-0.5*epsilon, data**2/(2*epsilon))

    f,a = plt.subplots(1,1,num='tv')
    a.clear()
    e_data = 1
    Lambda = 1e0

    py, px = pvm[0].nonzero()
    pv = lim[py,px]

    p = np.zeros(incidence_matrix.shape[0],'f')
    d = d_cut.astype('f').copy()
    L = incidence_matrix
    Lt = incidence_matrix.transpose()
    tau = 1.0/np.array(np.abs(L).sum(axis=0)).ravel()   # edge number for each node
    sigma = 0.5     # sigma = 1.0/np.array(np.abs(L).sum(axis=1)).ravel()

    d_result = np.full_like(lim, 0)
    p3d = np.vstack([(px-0.5*w)/435.016,
                     (py-0.5*h)/435.016,
                      np.ones(px.size)
                     ]).astype('f')*0.119554
    for ok in range(10):
        d0 = d.copy()
        # data for error terms
        It, Ig, Iw = warp_d(rim, px, py, pv, d0)
        b = It-d0*Ig

        d_ = d.copy()
        for ik in range(10):
            # dual step
            p += sigma*L.dot(d_)    # gradient ascend
            p /= np.maximum(1, np.abs(p)) # reprojection

            # primal step
            d_ = d.copy()           # remember d for later extrapolate
            d -= tau*Lt.dot(p)        # gradient descend

            r = It + Ig*(d-d0)                      # soft-thresholding
            th = tau*Lambda*Ig**2 + e_data
            idx1 = np.where(r > th)
            idx2 = np.where(r < -th)
            idx3 = np.where(np.abs(r) <= th)
            d[idx1] -= tau[idx1]*Lambda*Ig[idx1]
            d[idx2] += tau[idx2]*Lambda*Ig[idx2]
            d[idx3] = (d[idx3] - tau[idx3]*Lambda/e_data*Ig[idx3]*b[idx3] ) / (1+tau[idx3]*Lambda/e_data*Ig[idx3]**2)

            d = np.maximum(10, d)        # make sure depth stays positive
            d = np.minimum(160, d)        # clamp
            d_ = 2*d-d_                       # extrapolate gradient step

            Edata = huber_function(warp_d(rim, px, py, pv, d)[0], e_data).sum()
            Esmooth = np.abs(L.dot(d)).sum()
            print Edata, Esmooth


#        d_result[pvm[0]] = d
#        a.imshow(d_result, cmap='jet')
#        plt.pause(0.01)
    d_result[pvm[0]] = d
    plotxyzrgb(np.vstack([p3d/d,np.tile(pv,(3,1))]).T)

#%%

    def InterplatedMinimum(Cost):
        min_idx = np.argmin(Cost, axis=1)
        x = np.ogrid[:min_idx.size]
        max_id = Cost.shape[1]

        min_value = Cost[x, min_idx]
        min_value_plus1 = Cost[x, np.minimum(min_idx+1, max_id-1)]
        min_value_minus1 = Cost[x, np.maximum(min_idx-1, 0)]
        grad = (min_value_plus1 - min_value_minus1)/2.0
        grad2 = min_value_plus1 + min_value_minus1 - 2*min_value
        nz_mask = grad2!=0

        base = np.zeros_like(min_idx, dtype=np.float)
        base[nz_mask] = min_idx[nz_mask] - grad[nz_mask]/grad2[nz_mask]
        return base/Steps, min_value

    f,a = plt.subplots(1,1,num='tv')
    a.clear()


    Lambda = 1e0
    beta = 1e-3
    theta_init,theta_end = 100,1e-3
    epsilon = 0.01
    py, px = pvm[0].nonzero()

    q = np.zeros(incidence_matrix.shape[0],'f')
    d = d_cut.astype('f').copy()

    L = incidence_matrix
    Lt = incidence_matrix.transpose()
    tau = 1.0/np.array(np.abs(L).sum(axis=0)).ravel()   # edge number for each node
    sigma = 0.5     # sigma = 1.0/np.array(np.abs(L).sum(axis=1)).ravel()  # node number for each edge

    d_result = np.full_like(lim, 0)
    p3d = np.vstack([(px-0.5*w)/435.016,
                     (py-0.5*h)/435.016,
                      np.ones(px.size)
                     ]).astype('f')*0.119554
    Cost = unary_cost
    Eaus = np.empty_like(Cost,'f')
    n = 0
    i_list = np.linspace(0, 1, Cost.shape[1], endpoint=0)
    theta = theta_init
    while theta > theta_end:
        q = (q+sigma*L.dot(d))/(1.0+epsilon*sigma)
        q /= np.maximum(1, np.abs(q)) # reprojection

        # 2. gradient descent on the primal variable
        d = d + tau*(a/theta - Lt.dot(q))
        d = d/(1.0+tau/theta)

        # 3. Fix d, search optimal a
        for i in xrange(Cost.shape[1]):
            Eaus[:,i] = Lambda*Cost[:,i] + 0.5/theta*(d-i_list[i])**2
        a, c = InterplatedMinimum(Eaus)

        theta = theta*(1.0-beta*n)
        n += 1

        Edata = huber_function(warp_d(rim, px, py, pv, d)[0], e_data).sum()
        Esmooth = np.abs(L.dot(d)).sum()
        print Edata, Esmooth
    d_result[pvm[0]] = d
    plotxyzrgb(np.vstack([p3d/d,np.tile(pv,(3,1))]).T)
#%% full-image stereo
    exit()
    max_disp = 150
    def stereo_unaries(img1, img2):
        differences = []
        for disp in np.arange(max_disp+1):
            if disp == 0:
                diff = np.abs(img1 - img2)
            else:
                diff = np.abs(img1[:, disp:] - img2[:, :-disp])
            diff = diff[:, max_disp - disp:]
            differences.append(diff)
        return np.dstack(differences).copy("C")
    unaries = stereo_unaries(lim.astype('i4'), rim.astype('i4')).astype(np.int32)
    n_disps = unaries.shape[2]
    newshape = unaries.shape[:2]

    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.int32).copy("C")
    one_d_cut = cut_simple(unaries, 10 * one_d_topology).reshape(newshape)

    plt.imshow(np.argmin(unaries, axis=2), interpolation='nearest')
    d_result[:,max_disp:] = one_d_cut
    v,u = np.where(reduce(np.logical_and, [d_result>10, d_result<150, pvm[0]]))
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(lim[v,u],(3,1))]).T)

#%%
    def sample(x):
        return scipy.ndimage.map_coordinates(rim, [py,px+x], order=1, mode='constant', cval=1000)

    tau = 0.5

    from pyunlocbox import functions,solvers
    fsmooth = functions.norm_l2(A =lambda x: incidence_matrix.dot(x),
                                At=lambda x: incidence_matrix.transpose().dot(x))
    fdata = functions.norm_l1(y=lim[py,px],
                              A=sample,
                              lambda_= tau )
    solver = solvers.forward_backward(method='FISTA', step=0.5/tau)
    x0 = d_cut
    ret = solvers.solve([fdata, fsmooth], x0, solver, maxit=100)

#%%
    N = 30.0

    coffset = vec(2.0, -3.0, 4.0) #, 3.0
    C = (np.sin(6*np.pi*np.arange(N, dtype='f')/N + coffset)+1)*10
    cdt = [ dt(C[i],Lambda=2)[0] for i in range(3) ]


    I = np.array([[-1,1,0],[0,-1,1]],'f')
#    I = np.array([[-1,1,0,0],[0,-1,1,0],[1,0,-1,0],[0,1,0,-1]],'f'),[1,0,-1]
    L = I.T.dot(I)
    D = L.diagonal()
    A = np.diag(D)-L
    node_edge = np.abs(I).T
    '''setup neighbor LUT'''
    import pandas as pd
    nbrs_list = [np.where(A[v])[0] for v in range(I.shape[1])]
    nbrs = pd.DataFrame(nbrs_list, dtype=int).fillna(-1).values
    nbrs_cnt = D.astype('i')  # np.array([len(nbl) for nbl in nbrs_list],'i')



    def make_func(offset):
        return lambda x: (np.sin(6*np.pi*x/N + offset)+1)*10
    def make_func_reg(old, Lambda, theta, offset):
        return lambda x: Lambda*old(x)+(x-offset)**2/theta
    def find_root(func,x0):
        return scipy.optimize.minimize(func, x0, args=(), method='CG').x
    Cf = [make_func(offset) for offset in coffset]




    def solveLinearl2(x0, y, Lambda): #, L=L
        '''solve matrix equation: (λI+A)*x = y, for argmin[x'Ax +λ(x-y)**2]'''
        if scipy.sparse.isspmatrix(L):
            A = Lambda*scipy.sparse.identity(len(y)) + L
            x = scipy.sparse.linalg.spsolve(A, y)
        else:
            A = Lambda*np.identity(len(y)) + L
            x = np.linalg.solve(A, y)
        return x

    def solveFixIterl2(x, y, Lambda, it_num=10): # A=A, D=D,
        if 1:
            c0 = Lambda/(Lambda+D)
            c1 = 1/(Lambda+D)
            for it in range(it_num):
                x = c0*y + c1*A.dot(x)
            return x
        else:
            '''slower version'''
            nbrs = [np.where(A[v])[0] for v in range(len(x))]
            for it in range(it_num):
                '''for each node v'''
                for v in range(x.shape[0]):
                    '''neighbor u'''
                    c = [ x[u] for u in nbrs[v] ]
                    x[v] = Lambda/(Lambda+D[v])*y[v] + 1/(Lambda+D[v])*np.sum(c)
            return x

    def solveFixIterl1(x, y, Lambda, it_num=10): # A=A, D=D, I=I ,
        edge_node = np.abs(I)                # edges x nodes
        node_edge = edge_node.T
        ''' 1. A[n]*x = (A*x)[n] = [0,.., x, ..,0] vector indicating neighbours, 1xN
               of node n, which will be nonezero  (i.e. N=len(x))
            2. node_edge[m]*x = (node_edge*x)[m] = [0,.., x, ..,0] vector indicating nodes on egde m,
            3. node_edge.dot(e..)[n] = scalar = sum(edges related with node n )
               edge_node.dot(n..)[m] = scalar = sum(nodes related with edge m ) '''

        '''setup neighbor LUT'''
        nbrs = [np.where(A[v])[0] for v in range(len(x))]

        for it in range(it_num):
            eFlow = I.dot(x)        # Flow on edges = ▽x
            nGrad = np.sqrt(node_edge.dot(eFlow**2))     # norm(▽x) of each nodes
            nGrad = 1.0/(nGrad + 1e-4)
            x_ = x.copy()
            for pid in range(len(x)):
                eRuv = nGrad[pid] + nGrad[nbrs[pid]]
                x[pid] = 2*Lambda*y[pid] + np.sum(eRuv*x_[nbrs[pid]])
                x[pid] /= 2*Lambda + np.sum(eRuv)
            print x
        return x


    def solveFixIterl1c(x, y, Lambda, it_num=10): # I=I, nbrs=nbrs, nbrs_cnt=nbrs_cnt,
        node_edge = np.abs(I).T # nodesx edges

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
            print x
        return x

    def solvePDl1(x0, f, Lambda, it_num=10): #A=A, D=D, I=I ,
        node_edge = np.abs(I).T
        enode_out = np.where(I<0)[1]

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
#            print x, np.sqrt(node_edge.dot(I.dot(x)**2)).sum() + Lambda*np.sum((x-f)**2)
        return x

    def evalE_l1(x,y,Lambda):
        return np.sqrt(node_edge.dot(I.dot(x)**2)).sum() + Lambda*np.sum((x-y)**2)

    def evalE_l2(x,y,Lambda):
        return  ((I.dot(x))**2).sum() + Lambda*np.sum((x-y)**2)

    def findOpt(shape, func, *arg):
        n,m = shape  # nodes,datas = n,m
        e = np.empty((m,)*n,'f')
        from itertools import product
        for inds in product(range(m),repeat=n):
            e[inds] = func(inds, *arg)
        x_opt = np.unravel_index(np.argmin(e), e.shape)
        findOpt.e = e
        print Lambda, x_opt, e[x_opt]
        return x_opt

    def test_solveFixIter():
        evalEs = [evalE_l1, evalE_l2][0]

        tLambda = 1
        y = np.array([0,7,6],'f')
        x0 = np.zeros(3,'f') #y.copy()

        x_t = findOpt((len(y),30), evalEs, y, tLambda)
        x_e = [solvePDl1, solveFixIterl1c, solveFixIterl1,solveFixIterl2][2](x0, y, tLambda)
        print 'Lambda & y:',tLambda, y
        print 'reference:', x_t, evalEs(x_t, y, tLambda)
        print 'estimated:', x_e, evalEs(x_e, y, tLambda)

    def evalE_l1c(x, Lambda):
        return np.sqrt(node_edge.dot(I.dot(x)**2)).sum()+Lambda*(np.sum(C[range(len(x)),np.round(x).astype('i')]))

    def evalE_l1f(x, Lambda):
        return np.sqrt(node_edge.dot(I.dot(x)**2)).sum()+Lambda*(np.sum([obj(v) for obj,v in zip(Cf,vec(x)) ]))

    def evalE_l2c(x, Lambda):
        return x.T.dot(L.dot(x))+Lambda*(np.sum(C[range(len(x)),np.round(x).astype('i')]))

    def evalE_l2f(x, Lambda):
        return x.T.dot(L.dot(x))+Lambda*(np.sum([obj(v) for obj,v in zip(Cf,vec(x)) ]))
    evalE = [evalE_l1c,evalE_l1f,evalE_l2c,evalE_l2f][1]

    y = np.array([4,12,1],'f')
    x = y.copy()

    f,a = plt.subplots(1,1,num='cost-volume denoise');a.clear()
    a.set_xlim(-1,21)
    curset0 = [a.plot(data)[0] for data,color in zip(C,'bgrc')]
    curset1 = [a.plot(data,ls='--')[0] for data,color in zip(C,'bgrc')]
    lset1 = [a.axvline(obj,0,2, c=color) for obj,color in zip(vec(x),'bgrc')]
    lset2 = [a.axvline(obj,0,2, c=color, ls='--') for obj,color in zip(vec(y),'bgrc')]

    beta = 1e-3
    theta_init,theta_end = 1e4,1e-3
    theta = theta_init
    n = 0

    Lambda = 1.0

    solver = [solveFixIterl1c, solvePDl1, solveFixIterl2][1]
    while theta > theta_end:

        x = solver(x, y, 1/theta)
        pen = (np.arange(N)-vec(x))**2/theta
        Cnew = Lambda*C+pen
        if 0:
            y = np.argmin(Cnew, axis=-1)
        else:
            Cfnew = [make_func_reg(old, Lambda, theta, offset) for old,offset in zip(Cf,vec(x))]
            y = [find_root(func, x0) for func,x0 in zip(Cfnew, vec(y))]
            y = np.asarray(y).ravel()
            del Cfnew[:]

        [obj.set_xdata(bar) for obj,bar in zip(lset2,vec(y))]
        [obj.set_xdata(bar) for obj,bar in zip(lset1,vec(x))]
        [obj.set_ydata(bar) for obj,bar in zip(curset1,pen)]
        plt.pause(0.1)
        plt.waitforbuttonpress()

        theta = theta*(1.0-beta*n)
        n += 1
        print x, evalE(x, Lambda)

    x_opt = findOpt(C.shape, evalE, Lambda)
    print 'reference:', x_opt, evalE(x_opt, Lambda)
    [obj.set_xdata(bar) for obj,bar in zip(lset2,vec(x_opt))]




