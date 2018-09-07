# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:10:32 2018

@author: SCatheline
"""

PATH = "C:\\Users\\SCatheline\\OneDrive - Schlumberger\\Testing Project\\Machine Learning\\Fast.ai course\\courses\\dl1\\data\\dogscats\\"
sz=224
batch_size=64
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50, Xception, InceptionV3
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras import optimizers

from keras.utils import plot_model



train_data_dir = f'{PATH}train'
validation_data_dir = f'{PATH}valid'


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,rescale=1./255,rotation_range=40,width_shift_range=0.2,
    height_shift_range=0.2,fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
    shuffle=False,
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')

base_model = ResNet50(weights='imagenet', include_top=False)
#base_model = Xception(weights='imagenet', include_top=False)
#base_model = InceptionV3(weights='imagenet', include_top=False)
#plot_model(base_model, to_file='model_ini.png',show_shapes = True)

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)



model = Model(inputs=base_model.input, outputs=predictions)
#plot_model(model, to_file='model_f.png',show_shapes = True)

lr = 0.002

for layer in base_model.layers: layer.trainable = False
#opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  #SC
opt = optimizers.RMSprop(lr=lr, rho = 0.9, epsilon=None, decay=0.0)  #SC
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

#SC: added x2 to the train_generator to get twice as much data (with augementation) from the generator
model.fit_generator(train_generator, 2*train_generator.n // batch_size, epochs=30, workers=4,
        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)

#split_at = 140
#for layer in model.layers[:split_at]: layer.trainable = False
#for layer in model.layers[split_at:]: layer.trainable = True
#
#opt = optimizers.RMSprop(lr=lr/10, rho=0.9, epsilon=None, decay=0.0)  #SC
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])    
#
#model.fit_generator(train_generator, train_generator.n // batch_size, epochs=1, workers=3,
#        validation_data=validation_generator, validation_steps=validation_generator.n // batch_size)

