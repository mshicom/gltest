#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:15:55 2016

@author: nubot
"""

import roslib; roslib.load_manifest('ORB_SLAM')
import rospy
from TfMessageFilter import TfMessageFilter

import cv2
from sensor_msgs.msg import Image

from ORB_SLAM.msg import KeyFrameInfo
import numpy as np

import message_filters

from test_orb import *
from vtk_visualizer import plotxyzrgb

#from IPython.parallel import Client
#rc = Client()[0]
#
#rc.execute('from vtk_visualizer import plotxyzrgb')
#@rc.remote(block=False)
#def show3D(p3d):
#    plotxyzrgb(p3d)


#%%
# for rectified bumblebee stereo, ~/data/workspace/hkh_slam/orb_pos.bag
K = np.array([[435.016, 0, 511.913],
              [0, 435.016, 418.063],
              [0,       0,      1]])
fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
KF = None
p3d_update = False
frames = []
cGws = []

def callback(image_msg, cGw):
    global KF
    global p3d
    global p3d_update
    print("Got image:%f" % image_msg.header.stamp.secs)
    image = np.frombuffer(image_msg.data, np.uint8)
    image.shape = (image_msg.height, image_msg.width)
    h, w = image.shape
#    f = Frame(image, inv(cGw))

#    if KF is None:
#       KF = f
#       return
#
#    baseline = np.linalg.norm(getG(KF, f)[3,:3])
#
#    d,vm,var = KF.searchEPL(f, K, dmin=iD(2), dmax=iD(0.1), win_width=3)
#    p3d = KF.makePC(1.0/d, vm)
#    p3d_update = True
    frames.append(image)
    cGws.append(cGw)


rospy.init_node('orb_slam_mapping', log_level=rospy.DEBUG)
image_sub = message_filters.Subscriber('/stereo/11170132/left', Image)#'usb_cam/image_mono'
ts = TfMessageFilter(image_sub, 'ORB_SLAM/World', 'ORB_SLAM/Camera', queue_size=1000)

ts.registerCallback(callback)
rospy.spin()
np.savez("/mnt/workbench/orb_pos", frames=frames, cGws=cGws, K=K)

#while not rospy.is_shutdown():
#    if p3d_update:
#        plotxyzrgb(p3d)

# rosbag play orb_pos.bag --pause -r 0.1 -u 1
