#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tools import *

from scipy import weave
#%% distance transform
@timing
def dt(f, q=None, p=None, Lambda=1.0):
    # f, q,p,Lambda = a,None,None, 1.0
    f = f.astype('f').copy()
    n = f.shape[0]

    q = q.astype('f').copy() if not q is None else np.arange(n, dtype='f')
    p = p.astype('f').copy() if not p is None else np.arange(n, dtype='f')
    m = p.shape[0]
    d = np.empty(m,'f')
    d_arg = np.empty(m,'i')

    if 1:
        v_id = np.zeros(n,'i')
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
        for p_id in np.nditer(np.argsort(p)):
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
#def test_dt():
a = np.array(range(6)+range(5,0,-1))*5
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

