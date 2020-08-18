import sys
import os
import PIL.Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import  Convolution2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

K.clear_session()

# Preparación del entorno para trabajar sobre GPU
K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto( device_count = {'GPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

data_entrenamiento = 'entrenamiento'
data_validacion = 'validacion'

epocas = 5
longitud, altura = 150, 150
batch_size = 32
pasos = 100
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 2
lr = 0.0004

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Convolution2D(
    filtrosConv1, 
    tamano_filtro1, 
    padding ="same", 
    input_shape =(longitud, altura, 3), 
    activation='relu')
)

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(
    loss='categorical_crossentropy',
    optimizer = optimizers.Adam(lr=lr),
    metrics=['accuracy']
)

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch = pasos,
    epochs = epocas,
    validation_data = validacion_generador,
    validation_steps = validation_steps
)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')