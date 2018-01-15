import cv2
import os
import re
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator
from keras.models import load_model
from keras import backend
from keras import backend as K
from keras.layers.convolutional import Convolution1D
from keras.layers import Lambda, Dense
from keras.models import Sequential, model_from_json


test_output = pd.read_csv('./data/sample-submission.csv')
for i in range(10000):
    filename = './data/test/'+str(i+1)+'.jpg'; #print(filename)
    X = cv2.imread(filename)
    X = np.expand_dims(X, 0)
    y_pred = model.predict(X)
    y_pred_decoded = decode_y2(y_pred, confidence_thresh=0.4, iou_threshold=0.4, top_k='all', input_coords='centroids', normalize_coords=False, img_height=None, img_width=None)
    
    if len(y_pred_decoded[0])==0: 
    	label = 'unknown'
    	print(label)
    else:
	    box = y_pred_decoded[0][0] 
	    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	label = label.split(':')[0]
	print(label)=label
	test_output._set_value(i, 'Number', label)
test_output.to_csv("./data/test_output.csv", index=False)