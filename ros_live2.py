#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib; roslib.load_manifest('hkh_slam')
import rospy

import numpy as np
import tf
import message_filters
from sensor_msgs.msg import Image

from vtk_visualizer import plotxyzrgb
from test_orb import *


# for rectified bumblebee stereo, ~/data/workspace/hkh_slam/orb_pos.bag
K = np.array([[435.016, 0, 511.913],
              [0, 435.016, 418.063],
              [0,       0,      1]])
fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
KF = None
p3d_update = False
frames=[]
cGws=[]
def stereoImageCallback(left_msg, right_msg):
    global KF
    global p3d
    global p3d_update

    rospy.loginfo("Image received.")

    time = left_msg.header.stamp
    valid = False
    if tfl.frameExists("ORB_SLAM/World") and tfl.frameExists("ORB_SLAM/Camera"):
        try:
            tfl.waitForTransform("ORB_SLAM/World", "ORB_SLAM/Camera", time, rospy.Duration(1))
            (trans, rot) = tfl.lookupTransform("ORB_SLAM/World", "ORB_SLAM/Camera", time)
            cGw = tfl.fromTranslationRotation(trans, rot)
            valid = True
        except tf.Exception as e:
            rospy.loginfo("tf error: {0}".format(e.message))

    if valid:
        # save left & right image data to numpy array
        image = np.frombuffer(left_msg.data, np.uint8)
        image.shape = (left_msg.height, left_msg.width)
        frames.append(image)
        cGws.append(cGw)
#        f = Frame(image, inv(cGw))
#
#        if KF is None:
#           KF = f
#           KF.extractPts(K)
#           return
#
#        baseline = np.linalg.norm(getG(KF, f)[3,:3])
#
#        d,vm,var = KF.searchEPL(f, K, dmin=iD(2), dmax=iD(0.1), win_width=3)
#        p3d = KF.makePC(1.0/d, vm)
#        p3d_update = True


def cleanup():
    rospy.loginfo( "Shutting down vision node.")


#%%
if __name__ == '__main__':

    node_name = "orb_slam_dense"
    rospy.init_node(node_name)
    # What we do during shutdown
    rospy.on_shutdown(cleanup)

    baseline = 0.119554
    focal = 435.016

    left_image_sub = message_filters.Subscriber('/stereo/11170132/left', Image)
    right_image_sub = message_filters.Subscriber('/stereo/11170132/right', Image)
    ts = message_filters.TimeSynchronizer([left_image_sub, right_image_sub], 1)

    tfl = tf.TransformListener()

    # wait until orb_slam is working
    rospy.loginfo("Waiting for orb_slam...")
    tfl.waitForTransform("ORB_SLAM/World", "ORB_SLAM/Camera", rospy.Time(), rospy.Duration(1e9))
    rospy.loginfo("TF from orb_slam received.")

    # start the image callback routine
    ts.registerCallback(stereoImageCallback)
    rospy.spin()
    np.savez("/mnt/workbench/orb_pos", frames=frames, cGws=cGws, K=K)


#    while not rospy.is_shutdown():
#        if p3d_update:
#            plotxyzrgb(p3d)


