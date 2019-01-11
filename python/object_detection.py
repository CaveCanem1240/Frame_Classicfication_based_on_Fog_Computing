#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import time
import os
import csv
import math
# the networks compiled for NCS via ncsdk tools
tiny_yolo_graph_file= './yolo_tiny.graph'

# Tiny Yolo assumes input images are these dimensions.
TY_NETWORK_IMAGE_WIDTH = 448
TY_NETWORK_IMAGE_HEIGHT = 448


# googlenet mean values will be read in from .npy file
gn_mean = [0., 0., 0.]

# labels to display along with boxes if googlenet classification is good
# these will be read in from the synset_words.txt file for ilsvrc12
gn_labels = [""]

actual_frame_width = 0
actual_frame_height = 0

############################################################
# Tuning variables

# only keep boxes with probabilities greater than this
# when doing the tiny yolo filtering.
TY_BOX_PROBABILITY_THRESHOLD = 0.10  # 0.07


TY_MAX_IOU = 0.35

# end of tuning variables
#######################################################

def main():
    global gn_mean, gn_labels, actual_frame_height, actual_frame_width, TY_BOX_PROBABILITY_THRESHOLD, TY_MAX_IOU

    # Set logging level and initialize/open the first NCS we find
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
    devices = mvnc.EnumerateDevices()

    ty_device = mvnc.Device(devices[0])
    ty_device.OpenDevice()


    #Load tiny yolo graph from disk and allocate graph via API
    try:
        with open(tiny_yolo_graph_file, mode='rb') as ty_file:
            ty_graph_from_disk = ty_file.read()
        ty_graph = ty_device.AllocateGraph(ty_graph_from_disk)
    except:
        print ('Error - could not load tiny yolo graph file')
        ty_device.CloseDevice()
        return 1


    # GoogLenet initialization
    EXAMPLES_BASE_DIR = '../../'
    gn_mean = np.load(EXAMPLES_BASE_DIR + 'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1)  # loading the mean file

    gn_labels_file = EXAMPLES_BASE_DIR + 'data/ilsvrc12/synset_words.txt'
    gn_labels = np.loadtxt(gn_labels_file, str, delimiter='\t')
    for label_index in range(0, len(gn_labels)):
        temp = gn_labels[label_index].split(',')[0].split(' ', 1)[1]
        gn_labels[label_index] = temp

    exit_app = False


    TY_MAX_IOU = 0.15
    TY_BOX_PROBABILITY_THRESHOLD = 0.13

    while (True):
        write_time = 0
        times = 0
        while True :
            start_time = time.time()
            while True:
                if os.path.exists('/home/pi/Share_pi/mc/cap_ty_go.txt') == 1 and os.path.exists('/home/pi/Share_pi/mc/pic_ty.jpg') == 1 and os.path.exists('/home/pi/Share_pi/mc/ty_graph.txt') != 1:
                    break
            time.sleep(0.01)
            #video_device = cv2.VideoCapture('/home/pi/Share_pi/mc/pic_ty.jpg')
            os.remove('/home/pi/Share_pi/mc/cap_ty_go.txt')
            input_image = cv2.imread('/home/pi/Share_pi/mc/pic_ty.jpg',cv2.IMREAD_COLOR)
            actual_frame_width = 640.0
            actual_frame_height = 480.0
            print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))
            input_image = cv2.resize(input_image, (TY_NETWORK_IMAGE_WIDTH, TY_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
            display_image = input_image.copy()
            input_image = input_image.astype(np.float32)
            input_image = np.divide(input_image, 255.0)
            ty_graph.LoadTensor(input_image.astype(np.float16), 'user object')
            output, userobj = ty_graph.GetResult()

            times += 1
            np.savetxt('/home/pi/Share_pi/mc/ty_graph.txt',output)
            with open('/home/pi/Share_pi/mc/ty_finished.txt', 'w') as f:
                f.write('finished')
            end_time = time.time()
            write_time += (end_time-start_time)
            print(times)
            print(write_time/times)
        video_device.release()

        if (exit_app):
            break
    ty_graph.DeallocateGraph()
    ty_device.CloseDevice()


if __name__ == "__main__":
    print('waiting for gn...')
    while True:
        if os.path.exists('/home/pi/Share_pi/mc/go.txt') == 1:
            break
    time.sleep(0.01)
    os.remove("/home/pi/Share_pi/mc/go.txt")
    sys.exit(main())
