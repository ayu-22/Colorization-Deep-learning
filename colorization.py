import keras
import shutil
import keras
from keras.models import Model,Sequential
from keras.layers import *
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import os
import random



samples = 9000
train = np.empty((samples,256,256,3), 'float32')
l_train = np.empty((samples,256,256,1),'float32')
ab_train = np.empty((samples,256,256,2), 'float32')

for i in range(samples):
  image = cv2.imread('images/Train/'+files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  train[i] = image
  lab = rgb2lab((1.0/255)*image)
  l_train[i] = (lab[:,:,0]).reshape(256,256,1)
  ab_train[i] = (lab[:,:,1:]).reshape(256,256,2)
  
  
ab_train = ab_train/128

inception_res = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet')



x_res = np.empty((samples,299,299,3), 'float32')
img_res = np.empty((299,299,3), 'float32')
for i in range(samples):
  img = (l_train[i]).reshape(256,256)
  img = cv2.resize(img,(299,299))
  img = img.reshape(299,299)
  img_res[:,:,0] = img
  img_res[:,:,1] = img
  img_res[:,:,2] = img
  x_res[i] = img_res

res_out = inception_res.predict(x_res)

lr_red = ReduceLROnPlateau(monitor = 'loss',
                           patience = 5,
                           verbose = 1,
                           factor = 0.5,
                           min_lr = 0.0000000001
                           )

feature = Input(shape=(1000,))
encoder_i = Input(shape = (256,256,1))
encoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_i)
encoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=1)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=1)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same', strides=1)(encoder)
encoder = BatchNormalization()(encoder)
encoder = Conv2D(512, (3,3), activation='relu', padding='same', strides=1)(encoder)
encoder = Conv2D(256, (3,3), activation='relu', padding='same', strides=1)(encoder)

fusion = RepeatVector(32*32)(feature)
fusion = Reshape(([32,32,1000]))(fusion)
combine = concatenate([encoder, fusion], axis=3)
combine = Conv2D(256, (1,1), activation='relu', padding='same', strides=1)(combine)


decoder = Conv2D(128, (3,3), activation='relu', padding='same', strides=1)(combine)
decoder = BatchNormalization()(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=1)(decoder)
decoder = Conv2D(64, (3,3), activation='relu', padding='same', strides=1)(decoder)
decoder = BatchNormalization()(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(32, (3,3), activation='relu', padding='same', strides=1)(decoder)
decoder = Conv2D(2, (3,3), activation='relu', padding='same', strides=1)(decoder)
decoder = BatchNormalization()(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = Conv2D(2, (3,3), activation='tanh', padding='same', strides=1)(decoder)

model = Model(inputs = [encoder_i,feature], outputs = decoder)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='mse')

model.summary()

model.fit([l_train,res_out],
         ab_train,
         epochs = 1000,
         callbacks = [lr_red])