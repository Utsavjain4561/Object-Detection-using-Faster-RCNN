import numpy as np 
import model
import os 
import cv2
import config 
import skimage.io as io
import roi_helpers
from keras.layers import Input
from keras.models import Model
from optparse import OptionParser

def format_img_channels(img):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= 103.939
	img[:, :, 1] -= 116.779
	img[:, :, 2] -= 123.68
	img /= 1.0
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img
def format_img_size(img):
	""" formats the image size based on config """
	img_min_side = float(300)
	(height,width,_) = img.shape
		
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
def format_img(img):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img)
	img = format_img_channels(img)

	return img, ratio

input_shape_features = (None, None, 512)
feature_map_input = Input(shape=input_shape_features)

input_tensor = Input(shape=(None,None,3))
feature_map_input = Input(shape=input_shape_features)
input_roi = Input(shape=(None,4))

base_layers = model.nn_base(input_tensor)
rpn = model.rpn_base(base_layers,9)
classifer = model.classifer(base_layers,input_roi,4,21)


model_rpn = Model(input=input_tensor,output=rpn)

model_classifier = Model(input =(feature_map_input,input_roi),output=classifer)


model_full  = Model([input_tensor,input_roi],rpn[2:3]+classifer)
	
model_rpn.compile(optimizer='sgd',loss='mse')
model_classifier.compile(optimizer='sgd',loss='mse')
print('Model compiled succesfully')

parser = OptionParser()
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
# parser.add_option("--config_filename", dest="config_filename", help=
# 				"Location to read the metadata related to the training (generated when training).",
# 				default="config.pickle")

(options, args) = parser.parse_args()
#config_output_filename = options.config_filename

# with open(config_output_filename, 'rb') as f_in:
# 	C = pickle.load(f_in)
C = config.Config()
#print('Network used is '+C.network)
img_path = options.test_path

for idx,img_name in enumerate(sorted(os.listdir(img_path))):
	filename = img_path+'/'+img_name
	img = io.imread(filename)

	img, ratio = format_img(img)
	
	
	img = np.transpose(img, (0, 2, 3, 1))

	#print(img.shape)
	[features,x_class,x_reg]= model_rpn.predict(img)
	#print(x_class.shape[1:3])

	R = roi_helpers.rpn_to_roi(x_class ,x_reg, C, dim_ordering='tf', overlap_thresh=0.7)
	#print(R.shape)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
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
		[P_cls, P_regr] = model_classifier.predict([features, ROIs])
		print(P_cls)cd k
	



	

	



