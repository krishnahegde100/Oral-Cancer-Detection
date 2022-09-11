
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np

# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data

from sklearn.cross_validation import train_test_split


# input image dimensions


# In[8]:

path=('/home/krishna/ML/ocd/resiz')  #path of resized images
p1=('/home/krishna/ML/ocd/train')
p2=('/home/krishna/ML/ocd/validation')
imlist = os.listdir(path)
im1 = array(Image.open(path+'//'+ imlist[0]))
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
print(m,n)


# In[13]:

train_data_dir = p1
validation_data_dir = p2
datagen = ImageDataGenerator(rescale=1./255)
img_width=400
img_height=400
# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')



# In[11]:

m=150
n=150
classes=os.listdir(p1)
x=[]
y=[]
for fol in classes:
    print(fol)
    imgfiles=os.listdir(p1+'//'+fol);
    for img in imgfiles:
        im=Image.open(p1+'//'+fol+'//'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(fol)
        
x=np.array(x);
y=np.array(y);


# In[12]:

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=None)
print(x_train.shape[1:])
#print(y_test)


# In[29]:

batch_size=16
nb_classes=len(classes)
nb_epoch=10
nb_pool=2
nb_conv=5

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=None)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

model= Sequential()
nb_filters=32
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))

model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
nb_filters=64
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
nb_filters=128
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
'''
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(186));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
'''


model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))




# In[46]:

#saving model in json
from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 


# In[3]:

from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[4]:

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[15]:

import matplotlib.pyplot as plt
m=150
n=150
classes=os.listdir(p1)
im = Image.open('/home/krishna/ML/ocd/test/test.jpeg');
imrs = im.resize((m,n))
imrs=img_to_array(imrs)/255;
imrs=imrs.transpose(2,0,1);
imrs=imrs.reshape(3,m,n);
#print(classes)
x=[]
x.append(imrs)
x=np.array(x);
predictions = loaded_model.predict(x)
out={}
for i in range(2):
    print("%s --> %s"%(classes[i],predictions[0][i]))
    out[classes[i]]=(predictions[0][i])

import operator
sorted_x =out
z=sorted(out.items(), key=operator.itemgetter(1),reverse=True)

plt.bar(range(len(sorted_x)), list(sorted_x.values()), align='center')
plt.xticks(range(len(sorted_x)), list(sorted_x.keys()))
plt.show()


# In[11]:

im=Image.open('/home/krishna/ML/ocd/test/jpeg');
im=im.convert(mode='RGB')
imrs=im.resize((m,n))
imrs=img_to_array(imrs)/255;
#print(imrs)
imrs=imrs.transpose(2,0,1);
imrs=imrs.reshape(3,m,n);
print(imrs)


