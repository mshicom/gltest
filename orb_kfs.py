#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:57:45 2016

@author: nubot
"""
import numpy as np
import matplotlib.pyplot as plt
from EpilineCalculator import EpilineDrawer

import yaml
import re

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

def readOpencvYAMLFile(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

def opencv_matrix_representer(dumper, mat):
    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
yaml.add_representer(np.ndarray, opencv_matrix_representer)

def wirteOpencvYAMLFile(fileName, array_dict):
    # array_dict = {"a matrix": np.zeros((10,10)), "another_one": np.zeros((2,4))}
    with open(fileName, 'w') as f:
        f.write("%YAML:1.0\n")
        yaml.dump(array_dict, f)



def loadImageFromBag():
    import rosbag
    images = {}
    try:
        bag = rosbag.Bag('/home/nubot/data/bumblebee_11170132.bag')
    except:
        bag = rosbag.Bag('/home/kaihong/dataset/bumblebee_11170132.bag')

    for topic, msg, t in bag.read_messages('/stereo/11170132/left'):
        frame = np.fromstring(msg.data, np.uint8)
        frame.shape = (msg.height, msg.width)
        ts = msg.header.stamp.to_time()
        images[ts] = frame.copy()
    K = np.array([[435.016, 0, 511.913],
                  [0, 435.016, 418.063],
                  [0,       0,      1]])
    bag.close()
    return images, K

def dataFromOrb():
    keyframe_info = readOpencvYAMLFile("keyframe_info.yml")['kfs']
    image_set, K = loadImageFromBag()

    def getData(record_id):
        record = keyframe_info[record_id]
        keys = [record['TimeStamp']]
        keys.extend([nbr['TimeStamp']  for nbr in record['Neighbors']])
        frames = [ image_set[key] for key in keys]
        cGw = [record['Tcw']] + [nbr['Tcw'] for nbr in record['Neighbors']]
        wGc = [np.linalg.inv(g) for g in cGw]
        return frames, wGc, K
    return getData
loaddata4 = dataFromOrb()

if __name__ == "__main__":
    frames,wGc,K = loaddata4(30)
    EpilineDrawer(frames, wGc, K)