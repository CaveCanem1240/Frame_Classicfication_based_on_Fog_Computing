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
googlenet_graph_file= './googlenet.graph'

input_video_path = '.'

# Tiny Yolo assumes input images are these dimensions.
TY_NETWORK_IMAGE_WIDTH = 448
TY_NETWORK_IMAGE_HEIGHT = 448

# GoogLeNet assumes input images are these dimensions
GN_NETWORK_IMAGE_WIDTH = 224
GN_NETWORK_IMAGE_HEIGHT = 224

# googlenet mean values will be read in from .npy file
gn_mean = [0., 0., 0.]

# labels to display along with boxes if googlenet classification is good
# these will be read in from the synset_words.txt file for ilsvrc12
gn_labels = [""]

# for title bar of GUI window
cv_window_name = 'stream_ty_gn - Q to quit'

actual_frame_width = 0
actual_frame_height = 0

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

############################################################
# Tuning variables

# only keep boxes with probabilities greater than this
# when doing the tiny yolo filtering.
TY_BOX_PROBABILITY_THRESHOLD = 0.10  # 0.07

# if googlenet returns a probablity less than this then
# just use the tiny yolo more general classification ie 'bird'
GN_PROBABILITY_MIN = 0.5

# The intersection-over-union threshold to use when determining duplicates.
# objects/boxes found that are over this threshold will be considered the
# same object when filtering the Tiny Yolo output.
TY_MAX_IOU = 0.35

# end of tuning variables
#######################################################

do_googlenet = False

def filter_objects(inference_result, input_image_width, input_image_height):

    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    # the 20 classes this network was trained on
    network_classifications = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                               "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    # which types of objects do we want to include.
    network_classifications_mask = [0, 1, 1, 1, 0, 1, 1,
                                    1, 0, 1, 0, 1, 1, 1,
                                    1, 0, 1, 0, 1,0]

    num_classifications = len(network_classifications) # should be 20
    grid_size = 7 # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2 # the number of boxes returned for each grid cell

    # grid_size is 7 (grid is 7x7)
    # num classifications is 20
    # boxes per grid cell is 2
    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    # classification_probabilities  contains a probability for each classification for
    # each 64x64 pixel square of the grid.  The source image contains
    # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, input_image_width, input_image_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])


    probability_threshold_mask = np.array(all_probabilities >= TY_BOX_PROBABILITY_THRESHOLD, dtype='bool')
    box_threshold_mask = np.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities,axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then
    # sort the probabilities and classifications to match
    argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]


    # get mask for boxes that seem to be the same object
    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        if (network_classifications_mask[classifications_for_boxes_above[i]] != 0):
            classes_boxes_and_probs.append([network_classifications[classifications_for_boxes_above[i]],boxes_above_threshold[i][0],boxes_above_threshold[i][1],boxes_above_threshold[i][2],boxes_above_threshold[i][3],probabilities_above_threshold[i]])

    return classes_boxes_and_probs

def get_duplicate_box_mask(box_list):

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0: continue
        for j in range(i + 1, len(box_list)):
            if get_intersection_over_union(box_list[i], box_list[j]) > TY_MAX_IOU:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask

def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):

    # number of boxes per grid cell
    boxes_per_cell = 2

    # setup some offset values to map boxes to pixels
    # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    box_offset = np.transpose(np.reshape(np.array([np.arange(grid_size)]*(grid_size*2)),(boxes_per_cell,grid_size, grid_size)),(1,2,0))

    # adjust the box center
    box_list[:,:,:,0] += box_offset
    box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
    box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
    box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

    #scale the boxes to the image size in pixels
    box_list[:,:,:,0] *= image_width
    box_list[:,:,:,1] *= image_height
    box_list[:,:,:,2] *= image_width
    box_list[:,:,:,3] *= image_height

def get_intersection_over_union(box_1, box_2):

    # one diminsion of the intersecting box
    intersection_dim_1 = min(box_1[0]+0.5*box_1[2],box_2[0]+0.5*box_2[2])-\
                         max(box_1[0]-0.5*box_1[2],box_2[0]-0.5*box_2[2])

    # the other dimension of the intersecting box
    intersection_dim_2 = min(box_1[1]+0.5*box_1[3],box_2[1]+0.5*box_2[3])-\
                         max(box_1[1]-0.5*box_1[3],box_2[1]-0.5*box_2[3])

    if intersection_dim_1 < 0 or intersection_dim_2 < 0 :
        # no intersection area
        intersection_area = 0
    else :
        # intersection area is product of intersection dimensions
        intersection_area =  intersection_dim_1*intersection_dim_2

    # calculate the union area which is the area of each box added
    # and then we need to subtract out the intersection area since
    # it is counted twice (by definition it is in each box)
    union_area = box_1[2]*box_1[3] + box_2[2]*box_2[3] - intersection_area;

    # now we can return the intersection over union
    iou = intersection_area / union_area

    return iou


def overlay_on_image(display_image, filtered_objects):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

	# copy image so we can draw on it.
    #display_image = source_image.copy()
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):

        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3])//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4])//2 + DISPLAY_BOX_HEIGHT_PAD

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        #draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text

        if (filtered_objects[obj_index][8] > GN_PROBABILITY_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        if (label_top < 1):
            label_top = 1
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]


        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        # label text above the box
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def get_googlenet_classifications(gn_graph, source_image, filtered_objects):

    # pad the height and width of the image boxes by this amount
    # to make sure we get the whole object in the image that
    # we pass to googlenet
    WIDTH_PAD = 20
    HEIGHT_PAD = 30

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # loop through each box and crop the image in that rectangle
    # from the source image and then use it as input for googlenet
    #print(filtered_objects)
    #print(type(filtered_objects))
    for obj_index in range(len(filtered_objects)):
        if (do_googlenet):
            center_x = int(filtered_objects[obj_index][1])
            center_y = int(filtered_objects[obj_index][2])
            half_width = int(filtered_objects[obj_index][3])//2 + WIDTH_PAD
            half_height = int(filtered_objects[obj_index][4])//2 + HEIGHT_PAD

            # calculate box (left, top) and (right, bottom) coordinates
            box_left = max(center_x - half_width, 0)
            box_top = max(center_y - half_height, 0)
            box_right = min(center_x + half_width, source_image_width)
            box_bottom = min(center_y + half_height, source_image_height)

            # get one image by clipping a box out of source image
            one_image = source_image[box_top:box_bottom, box_left:box_right]

            # Get a googlenet inference on that one image and add the information
            # to the filtered objects list
            filtered_objects[obj_index] += googlenet_inference(gn_graph, one_image)
        else:
            filtered_objects[obj_index] += [0., 0., 0.]

    return

def googlenet_inference(gn_graph, input_image):

    # Resize image to googlenet network width and height
    # then convert to float32, normalize (divide by 255),
    # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
    input_image = cv2.resize(input_image, (GN_NETWORK_IMAGE_WIDTH, GN_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
    input_image = input_image.astype(np.float32)
    input_image[:, :, 0] = (input_image[:, :, 0] - gn_mean[0])
    input_image[:, :, 1] = (input_image[:, :, 1] - gn_mean[1])
    input_image[:, :, 2] = (input_image[:, :, 2] - gn_mean[2])

    # Load tensor and get result.  This executes the inference on the NCS
    gn_graph.LoadTensor(input_image.astype(np.float16), 'googlenet')
    output, userobj = gn_graph.GetResult()

    order = output.argsort()[::-1][:1]


    # index, label, probability
    return order[0], gn_labels[order[0]], output[order[0]]




def main():
    global gn_mean, gn_labels, actual_frame_height, actual_frame_width, TY_BOX_PROBABILITY_THRESHOLD, TY_MAX_IOU

    # Set logging level and initialize/open the first NCS we find
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
    devices = mvnc.EnumerateDevices()

    gn_device = mvnc.Device(devices[0])
    gn_device.OpenDevice()


    try:
        with open(googlenet_graph_file, mode='rb') as gn_file:
            gn_graph_from_disk = gn_file.read()
        gn_graph = gn_device.AllocateGraph(gn_graph_from_disk)
        print(gn_graph)
        print(type(gn_graph))
    except:
        print ('Error - could not load googlenet graph file')
        gn_device.CloseDevice()
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

    print('Starting GUI, press Q to quit')


    TY_MAX_IOU = 0.15
    TY_BOX_PROBABILITY_THRESHOLD = 0.13

    while (True):
        write_time = 0
        times = 0
        while True :
            start_time = time.time()
            while True:
                if os.path.exists('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_go.txt') == 1 and os.path.exists('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_gn_go.txt') == 1:
                    break
            actual_frame_width = 640.0
            actual_frame_height = 480.0
            print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))
            input_image = cv2.imread('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/pic_gn.jpg',cv2.IMREAD_COLOR)
            os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_go.txt')
            os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/cap_gn_go.txt')
            input_image = cv2.resize(input_image, (TY_NETWORK_IMAGE_WIDTH, TY_NETWORK_IMAGE_HEIGHT), cv2.INTER_LINEAR)
            # save a display image as read from video device.
            display_image = input_image.copy()

            # modify input_image for TinyYolo input
            input_image = input_image.astype(np.float32)
            input_image = np.divide(input_image, 255.0)
            output = np.loadtxt('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_graph.txt')
            #s_time = time.time()
            filtered_objs = filter_objects(output.astype(np.float32), input_image.shape[1], input_image.shape[0])
            get_googlenet_classifications(gn_graph, display_image, filtered_objs)
            overlay_on_image(display_image, filtered_objs)
            display_image = cv2.resize(display_image, (int(actual_frame_width), int(actual_frame_height)),cv2.INTER_LINEAR)
            cv2.imwrite('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/display_image.jpg',display_image)
            os.remove('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/ty_graph.txt')
            while True:
                try:
                    with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/gn_finished.txt', 'w') as f:
                        f.write('finished')
                    break
                except FileNotFoundError:
                    pass
            '''
            while True:
                try:
                    with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/display_go.txt', 'w') as f:
                        f.write('go')
                    break
                except FileNotFoundError:
                    pass
            '''
            times+=1
            end_time = time.time()
            write_time += (end_time-start_time)
            print(times)
            print(write_time/times)
        # close video device
        video_device.release()

        if (exit_app):
            break
    # clean up tiny yolo
    ty_graph.DeallocateGraph()
    ty_device.CloseDevice()

    # Clean up googlenet
    gn_graph.DeallocateGraph()
    gn_device.CloseDevice()

    print('Finished')


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    with open('/run/user/1000/gvfs/smb-share:server=192.168.0.100,share=share_pi/mc/go.txt', 'w') as f:
        f.write('go')
    sys.exit(main())
