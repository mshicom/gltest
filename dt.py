#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tools import *

from scipy import weave

def gen_dt(f, q=None, Lambda=1.0):
    f = f.astype('f')
    n = f.shape[0]

    q = q.astype('f') if not q is None else np.arange(n, dtype='f')
    v_id = np.empty(n+1,'i4')
    z = np.full(n+1, np.inf, 'f')
    Lambda2 = Lambda*2
    scode = r'''
        template <class T>
            inline T square(const T &x) { return x*x; };

        #define INF std::numeric_limits<float>::infinity()
        void dt_pre(const float *f, const float Lambda2, const float *q, const int n,
                float *z, int *v_id)
        {
            int k = 0;
            v_id[0] = 0;
            z[0] = -INF;
            z[1] = +INF;
            for (int q_id = 1; q_id <= n-1; q_id++) {
                float s = 0.5*((f[q_id]*Lambda2+square(q[q_id]))-(f[v_id[k]]*Lambda2+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
                while (s <= z[k]) {
                    k--;
                    s = 0.5*((f[q_id]*Lambda2+square(q[q_id]))-(f[v_id[k]]*Lambda2+square(q[v_id[k]])))/(q[q_id]-q[v_id[k]]);
                }
                k++;
                v_id[k] = q_id;
                z[k] = s;
                z[k+1] = +INF;
            }
        }'''
    code = r'''
      //std::raise(SIGINT);
     dt_pre(f,Lambda2,q,n,z,v_id);
    '''
    weave.inline(code,['f','n','q','Lambda2','z','v_id'],
                 support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<stdio.h>','<csignal>'],
                 compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])
    debug = 0
    if debug:
        fig,a = plt.subplots(num="dt")
        a.plot(q, f, 'b')
        l = a.axvline(c='b')
        li = a.axvline(c='r')
        c, = a.plot(f,'r')
        a.autoscale(False)

    def dt(p, interplate=True):
        k = np.searchsorted(z, p)-1     # z:=[-inf, ... , +inf]
        q_id = v_id[k]

        if interplate and q_id<n-1 and q_id>0:
            x1,x2,x3 = q[q_id-1],q[q_id],q[q_id+1]
            y1 = (p-x1)**2/Lambda2 + f[q_id-1]
            y2 = (p-x2)**2/Lambda2 + f[q_id]
            y3 = (p-x3)**2/Lambda2 + f[q_id+1]
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
            A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
            B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
            best = -B / (2*A)
        else:
            best = q[q_id]

        if debug:
            li.set_xdata(best)
            l.set_xdata(q[q_id])
            c.set_data(q, (p-q)**2/(Lambda*2)+f)
        return best
    return dt


def test_dt():
    a = np.array(range(5,0,-1)+range(6))*5
    dt = gen_dt(a)
    best_id = np.atleast_1d(0.0)
    best_id = dt(np.atleast_1d(best_id)); print  best_id


def gen_proxg_dt(C, tau=1):
    N = C.shape[0]
    dts = []
    for i in xrange(N):
        dts.append(gen_dt(C[i], Lambda=tau))

    def prox_g(x, interplate=True):
        x_ = x.copy()
        for i in xrange(N):
            x_[i] = dts[i](x[i], interplate)
        return x
    return prox_g

if 0:

    N = 30
    C = (np.sin(2*np.pi*np.arange(N, dtype='f')/N + vec(2,-3,4))+1)*10

    I = np.array([[1,-1,0],
                  [0,1,-1],
                  [-1,0,1]],'f') #
    enode_out = np.asarray([np.nonzero(row==-1)[0] for row in I]).ravel()
    node_edge = np.abs(I).T

    def amp(eFlow):
        return np.sqrt(node_edge.dot(eFlow**2))     # 1xN, norm(▽x) of each nodes
    def F(x, alpha):
        return alpha * np.sum(amp(I.dot(x)))

    def prox_fl1(x, tau):
        """
        F = L1 norm := ||∇u||1 = ∑∑sqrt(∂x**2+∂y**2)
        prox_λf(v) = soft_thresholding
                   = (1 − λ/||v||)*v,  if ||v|| > λ
                   = 0,                if ||v|| < λ
        """
        v = amp(x)
        # return np.where(v<tau, [np.zeros_like(x), (1-tau/v)*x])
        return np.maximum(0, 1 - tau / np.maximum(v, 1E-10)) * x # 1 − λ/||v|| < 0, if ||v|| < λ

    def Projection(eFlow, tau):
        """
        dual(Fl1) = Projection on unit Ball
        """
        node_grad = np.maximum(amp(eFlow), 1)
        eFlow /= node_grad[node_edge]
        return eFlow

    def dual_prox(prox):
        """
        prox_[σ]F'(x) = x - σ*prox_[1/σ]F(x/σ)
        """
        def prox_f_dual(x, sigma):
            return x - sigma * prox(x / sigma, 1 / sigma)
        return prox_fd
    prox_fs = Projection # or dual_prox(prox_fl1)

    prox_g = gen_proxg_dt(C, tau)

    x = np.array([25,5,0],'f')
    xe = x.copy
    p = I.dot(x)

    f,a = plt.subplots(1,1,num='ADMM');a.clear()
    a.plot(C.T)
    a.set_xlim(-1,30)
    l1,l2,l3 = (a.axvline(foo,0,2,c=color) for foo,color in zip(vec(x),['b','g','r']))

    for it in range(30):
        [foo.set_xdata(bar) for foo,bar in zip([l1,l2,l3],vec(x))]
        print np.sqrt(np.sum(I.dot(x)**2))+C[range(3),x.astype('i')].sum()

        plt.pause(0.01)
        plt.waitforbuttonpress()

#        x = prox_g(x-I.T.dot(I.dot(x)-z+u))
#        z = prox_fl2(I.dot(x) + u)
#        u = u+I.dot(x)-z

        x_old = x.copy()

        p += alpha*sigma * I.dot(xe)
        p = prox_fs(p, alpha*sigma) # y = Prox_σF'(y + σ Kx)

        x -= tau * I.T.dot(p)
        x = prox_g(x)               # x = ProxτG(x − τ*K'y)

        xe = 2*x - d_old


#%% distance transform
@timing
def dt_batch(f, q=None, p=None, Lambda=1.0):
    # f, q,p,Lambda = a,None,None, 1.0
    f = f.astype('f').copy()
    n = f.shape[0]

    q = q.astype('f').copy() if not q is None else np.arange(n, dtype='f')
    p = p.astype('f').copy() if not p is None else np.arange(n, dtype='f')
    m = p.shape[0]
    d = np.empty(m,'f')
    d_arg = np.empty(m,'i')

    if 0:
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
         dt(f,Lambda,p,n,q,m,d,d_arg);
        '''
        weave.inline(code,['d','d_arg','f','n','m','p','q','Lambda'],
                     support_code=scode, headers=['<algorithm>','<cmath>','<vector>','<stdio.h>','<csignal>'],
                     compiler='gcc', extra_compile_args=['-std=gnu++11 -msse2 -O3'])
    return d,d_arg

def test_dt_batch():
    a = np.array(range(6)+range(5,0,-1))*5
    cost,best_id = dt_batch(a)
    plt.plot(a)
    plt.plot(cost)
    [plt.plot( a+(np.arange(11)-i)**2, ':') for i in range(11)]



