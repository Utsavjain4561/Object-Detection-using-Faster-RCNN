import numpy as np 
from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,Input,Flatten,Dense,Dropout
from RoiPoolingConv import RoiPoolingConv

def classifer(base_layers,input_roi,num_rois,num_classes=21):
	pooling_regions = 7
	x = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_roi])
	x = Flatten()(x)
	x = Dense(4096,activation='relu')(x)
	x = Dropout(0.5)(x)

	x = Dense(4096,activation='relu')(x)
	x = Dropout(0.5)(x)

	x_class = Dense(num_classes,activation='softmax')(x)
	x_reg = Dense((num_classes-1)*4,activation='linear')(x)

	return [x_class,x_reg]


def rpn_base(base_layers,num_anchors):
	x = Conv2D(512,(3,3),activation='relu',padding='same')(base_layers)

	x_class = Conv2D(num_anchors,(1,1),activation='sigmoid',padding='same')(x)
	x_reg = Conv2D(num_anchors*4,(1,1),activation='linear',padding='same')(x)

	return [x,x_class,x_reg]

def nn_base(input_tensor):

	x = Conv2D(64,(3,3),activation='relu',padding='same')(input_tensor)
	x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
	x = MaxPooling2D((2,2),strides=(2,2),padding='same')(x)

	x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
	x = MaxPooling2D((2,2),strides=(2,2),padding='same')(x)

	x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
	x = MaxPooling2D((2,2),strides=(2,2),padding='same')(x)

	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)
	x = MaxPooling2D((2,2),strides=(2,2),padding='same')(x)

	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)
	x = Conv2D(512,(3,3),activation='relu',padding='same')(x)

	return x







