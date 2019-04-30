#from __future__ import division
# Importing python modules

#The OS module in python provides functions for interacting with the operating system.
#OS, comes under Python’s standard utility modules.
#This module provides a portable way of using operating system dependent functionality.
import os
import cv2
#NumPy is a general-purpose array-processing package.
#It provides a high-performance multidimensional array object,
#and tools for working with these arrays.
import numpy as np
import sys
import math
#The pickle module implements a fundamental,
#but powerful algorithm for serializing and de-serializing a Python object structure
import pickle
#It is a python module which is used to plot graphs.
import matplotlib.pyplot as plt
#It is used to parse command line arguments
#eg. python test_frcnn.py -p <testpath>
from optparse import OptionParser
import time
#Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
#It was developed with a focus on enabling fast experimentation.
import keras.backend as K
#It is a module which contains meta data about the file .
#eg. Image size=300,No of ROI=32 etc.
from keras_frcnn import config
from keras.layers import Input
#The Keras Model API is the way to go for defining complex models,
#such as multi-output models, directed acyclic graphs, or models with shared layers.
from keras.models import Model
#It is a a module for generating regions of interest.
from keras_frcnn import roi_helpers

#It parses command line arguments
#stores them in a destination variable 'dest'
parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='vgg')
parser.add_option("--input_weight",dest="weights",help="Load pre trained weights for testing",default='weight/model_frcnn1.hdf5')

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename
# Loading meta data about the project in variable 'C'
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)
# sImporting base network as VGG(Visual Geometry Group)
if C.network == 'vgg':
    import keras_frcnn.vgg as nn


img_path = options.test_path

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width ,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
  
    """ formats the image channels based on config """

    img = img[:, :, (2, 1, 0)]

    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    #print(img)
    img = np.transpose(img, (2, 0, 1))
    #print(img)
    img = np.expand_dims(img, axis=0)
    return img

# formatting image on the basis of :
# --> fixing the smaller side of the image as 300
# --> adjusting other dimension as well as maintaing the aspect ratio
def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

# This contains mapping of the 20 different classes available in the pascal dataset
class_mapping = C.class_mapping
# Adding an extra class mapping for background.
# 'person': 0,
# 'diningtable': 1,
# 'chair': 2,
# 'pottedplant': 3,
# 'car': 4,
# 'horse': 5,
# 'cat': 6,
# 'sofa': 7,
# 'boat': 8,
# 'aeroplane': 9,
# 'dog': 10,
# 'train': 11,
# 'motorbike': 12,
# 'bicycle': 13,
# 'bus': 14,
# 'bird': 15,
# 'bottle': 16,
# 'sheep': 17,
# 'tvmonitor': 18,
# 'cow': 19,
# 'bg': 20
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

class_to_color = {class_mapping[v]: np.random.randint(0, 255) for v in class_mapping}
#Size of a single batch of ROIS
C.num_rois = int(options.num_rois)
#Setting the number of features =512
if C.network == 'vgg':
    num_features = 512

#input_shape_image : defines the shape of input image --eg. 600 x 540 x 30
#input_shape_features : defines the features of input image --eg. 18 x 28 x512
input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

#creatinf tensor objects of the above defined shapes
img_input = Input(shape=input_shape_img)
#ROI_input defines the shape of a single region of interest
#eg. 32 x 4
roi_input = Input(shape=(C.num_rois, 4))
#print('Number of ROIS;'+str(C.num_rois))
feature_map_input = Input(shape=input_shape_features)

# define the base network which is VGG
shared_layers = nn.nn_base(img_input)

# define the number of anchors ( here num_anchors=9)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# define the RPN, built on the base layers
rpn_layers = nn.rpn(shared_layers, num_anchors)


classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

#creating model of the above defines network
#model_rpn : contains model for RPN(Region Proposal Network)
#model_classifir : contains model for classification

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
model_rpn.summary()

print('Loading weights from {}'.format(options.weights))
model_rpn.load_weights(options.weights, by_name=True)
#print(model_rpn.get_weights())
model_classifier.load_weights(options.weights, by_name=True)

# compiling the above defined models with loss function as mean squared error
model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
model_classifier.summary()
all_imgs = []

classes = {}
#setting a threshold value for picking the desired bounding box
bbox_threshold = 0.7

visualise = True
#model_rpn.summary()

# Testing the images provided by the given path
for idx, img_name in enumerate(sorted(os.listdir 	(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path,img_name)

    img = cv2.imread(filepath)
    #format_image  : It is a function used for formatting the image for prediction
    X, ratio = format_img(img, C)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

# Predicting the output of a given image and thereby passing the ouptut obatined as a feature map(f)
# into the RPN function for further classification
# Y1 : It is used to decide whether we have a backgroungd or an on object
# Y2 : It gives us the co-ordinates of the given object or background
    [Y1, Y2, F] = model_rpn.predict(X)
    print(F)
    print("Feature Map shape ",F.shape)
      
  
#Generates 300 region proposals
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    print(R)
         
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    # the output shape will be 10 batches of 32 region proposals each
    bboxes = {}
    probs = {}
    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
            print(ROIs)

        # passing the feature map and one single batch of ROI into the detector part for further clssification
        # that what is the object which we have figured out
        # P_cls : contains the probabilities for each of the 20 classes
        # P_regr : It contains the relative co-ordinates for each proposed region
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
     
        print(P_cls.shape)
        for ii in range(P_cls.shape[1]):
          
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
               continue
            print(P_cls[0,ii,0])
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
           # print(P_cls[0,0,:])

             
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]
            # Selecting the class with the max probability in P_cls list
            # Scaling the co-ordinates of the region with max probabilty through linear regression
            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    # Creating the boundary boxof an image with the corresponding label and the respective accuracy
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (0,0,255),2)
            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))
            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)


            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 0, 0), 2)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

    print('Elapsed time = {}'.format(time.time() - st))
    cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
    #Storing the result in folder result images