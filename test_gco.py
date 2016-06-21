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

from pygco import *
def sim(*arg,**kwarg):
    return np.hstack(arg)
import scipy
import scipy.ndimage
from vtk_visualizer import *
from scipy import weave
from mpl_toolkits.mplot3d import Axes3D

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
        d = calcGradient(im)
#        d = scipy.ndimage.filters.gaussian_gradient_magnitude(im.astype('f'),1)
        d_abs = np.abs(d)
        valid_mask = d_abs>np.percentile(d_abs,80)
        dI.append( d.copy() )
        u, v = np.meshgrid(range(w),range(h))
        pixel_mask = reduce(np.logical_and,[valid_mask, u>1, v>1, u<w-2, v<h-2])
        px.append(u[pixel_mask].copy())
        py.append(v[pixel_mask].copy())
        pvm.append(pixel_mask.copy())

    pis(valid_mask)

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

    '''1. setup neighbors'''
    def makeEdges(mask_image):
        py, px = mask_image.nonzero()
        node_cnt = px.size

        id_LUT = np.empty_like(mask_image, 'i4')
        id_LUT[mask_image] = range(node_cnt)      # lookup-table of index number for valid pixels
        edges = []
        for e_id, e_x,e_y in zip(range(node_cnt), px, py):
            if mask_image[e_y-1, e_x+1]:
                edges.append([e_id, id_LUT[e_y-1, e_x+1]])
            if mask_image[e_y-1, e_x]:
                edges.append([e_id, id_LUT[e_y-1,  e_x ]])
            if mask_image[e_y-1, e_x-1]:
                edges.append([e_id, id_LUT[e_y-1, e_x-1]])
            if mask_image[e_y, e_x-1]:
                edges.append([e_id, id_LUT[ e_y,  e_x-1]])
        return np.array(edges,'i4')       # edges array, each row represent an edge
    edges = makeEdges(pvm[0])

    '''2. setup matching cost'''
    max_disp = 150
    vec = lambda x:np.reshape(x,(-1,1))
    py, px = pvm[0].nonzero()

    unary_cost = np.empty([px.size, max_disp+1],'i4')       # shape= n_vertices x n_disps
    y_ind, x_ind = vec(py) , vec(px)-np.arange(max_disp+1)
    unary_cost = np.abs(vec(lim[py, px]) - rim[y_ind, x_ind])  # shape=(n_vertices, n_labels)
    unary_cost[x_ind<0] = 100

    '''3. setup smoothness cost'''
    if 0:
        pairwise_cost = -5*np.eye(max_disp+1, dtype='i4')+5        # shape=(n_labels, n_labels)
    else:
        dx, dy = np.ogrid[:max_disp+1, :max_disp+1]
        pairwise_cost = np.abs(dx - dy).astype('i4').copy("C")

    '''4. do the calculation'''
    d_cut = cut_from_graph(edges, unary_cost, 5*pairwise_cost,  n_iter=5)

    '''5. plot '''
    d_result = np.full_like(lim, -1)
    d_result[pvm[0]] = d_cut

    v,u = np.where(reduce(np.logical_and, [d_result>10, d_result<150, pvm[0]]))
    p3d = np.vstack([(u-0.5*w)/435.016,
                     (v-0.5*h)/435.016,
                     np.ones(u.shape[0])
                     ]).astype('f')/d_result[v,u]*0.119554
    plotxyzrgb(np.vstack([p3d,np.tile(lim[v,u],(3,1))]).T)

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
