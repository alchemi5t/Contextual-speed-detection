from __future__ import print_function
import cv2.cv2 as cv2
import tensorflow as tf
import tensorflow.keras as keras
import glob
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM,Dense, Dropout, Input, Conv2D
from tensorflow.keras.layers import SpatialDropout1D, concatenate
from tensorflow.keras.layers import Flatten, MaxPooling2D
import sklearn
from tensorflow.keras.layers import Embedding
import datetime
import sys
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


PATH_TO_MODEL = "speed_recognition.PB"
PATH_TO_TRAIN_DIR = "data/images/train/"
PATH_TO_TRAIN_OUTPUT_DATASET = "data/train.txt"

def dataGenerator(filenames, groundTruth, shuffle=True, batchSize=10, targetSize=(250,250), dataset="train"):
	while(True):	
		noOfSample = len(filenames)
		noOfSteps = noOfSample // batchSize
		for i in range (0,noOfSteps):
			images=[]
			outputs=[]
			# print(i*batchSize)
			for j in range (0,batchSize):
				# print(i*batchSize + j)
				image = cv2.imread(filenames[i*batchSize + j]) 
				image = cv2.resize(image, targetSize)
				output = groundTruth[i*batchSize + j]

				images.append(image)
				outputs.append(output)

				lr_image = np.fliplr(image)
				images.append(lr_image)
				outputs.append(output)
			images = np.array(images)
			outputs = np.array(outputs)
			if(shuffle):
				images, outputs = sklearn.utils.shuffle(images, outputs)
			yield images, outputs

class CustomCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(epoch==0):
			self.val_loss=logs.get('val_loss')
		elif(logs.get('val_loss')<self.val_loss):
			print("\nSaving Model for minimum validation loss\n")
			self.val_loss=logs.get('val_loss')
			model.save('data/trained_model')
def setupCoNvidiaModel(inputShape):

    input_img1 = Input((inputShape[1], inputShape[0], 3))
    # Convolution Layer 1
    convLayer1 = Conv2D(filters=4,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(input_img1)
    convLayer2 = Conv2D(filters=4,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 2, kernel_initializer = 'he_normal')(input_img1)
    convLayer3 = Conv2D(filters=4,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 3, kernel_initializer = 'he_normal')(input_img1)
    convLayer4 = Conv2D(filters=14,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 4, kernel_initializer = 'he_normal')(input_img1)
    merged = concatenate([convLayer1, convLayer2,convLayer3,convLayer4])
    merged = Dropout(0.05)(merged)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(merged)

    # Convolution Layer 2

    convLayer1 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer2 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 2, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer3 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 3, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer4 = Conv2D(filters=8,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 4, kernel_initializer = 'he_normal')(poolingLayer)
    merged = concatenate([convLayer1, convLayer2,convLayer3,convLayer4])
    merged = Dropout(0.05)(merged)

    # convLayer = Conv2D(filters=24,
    #                    kernel_size=(5, 5),
    #                    activation='elu')(poolingLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(merged)

    # Convolution Layer 3

    convLayer1 = Conv2D(filters=12,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer2 = Conv2D(filters=12,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer3 = Conv2D(filters=12,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer4 = Conv2D(filters=12,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    merged = concatenate([convLayer1, convLayer2,convLayer3,convLayer4])



    # convLayer = Conv2D(filters=36,
    #                    kernel_size=(3, 3),
    #                    activation='elu')(poolingLayer)
    # Pooling Layer 3
    merged = Dropout(0.05)(merged)
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(merged)

    # Convolution Layer 4

    convLayer1 = Conv2D(filters=16,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer2 = Conv2D(filters=16,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer3 = Conv2D(filters=16,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    convLayer4 = Conv2D(filters=16,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape,
                       padding = "same",
                       dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    merged = concatenate([convLayer1, convLayer2,convLayer3,convLayer4])
    # merged = Dropout(0.05)(merged)
    #from here
    # poolingLayer = MaxPooling2D(pool_size=(2, 2),
    #                             strides=(2, 2))(merged)
    # convLayer1 = Conv2D(filters=16,
    #                    kernel_size=(3, 3),
    #                    strides=(1, 1),
    #                    activation='elu',
    #                    input_shape=inputShape,
    #                    padding = "same",
    #                    dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    # convLayer2 = Conv2D(filters=16,
    #                    kernel_size=(3, 3),
    #                    strides=(1, 1),
    #                    activation='elu',
    #                    input_shape=inputShape,
    #                    padding = "same",
    #                    dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    # convLayer3 = Conv2D(filters=16,
    #                    kernel_size=(3, 3),
    #                    strides=(1, 1),
    #                    activation='elu',
    #                    input_shape=inputShape,
    #                    padding = "same",
    #                    dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    # convLayer4 = Conv2D(filters=16,
    #                    kernel_size=(3, 3),
    #                    strides=(1, 1),
    #                    activation='elu',
    #                    input_shape=inputShape,
    #                    padding = "same",
    #                    dilation_rate = 1, kernel_initializer = 'he_normal')(poolingLayer)
    # merged = concatenate([convLayer1, convLayer2,convLayer3,convLayer4])
    #to here
    # # Pooling Layer 4
    poolingLayer =Flatten()(MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(merged))

    denseLayer = Dense(1164,
                       activation='elu', kernel_initializer = 'he_normal')(poolingLayer)
    # Dense layer 2
    denseLayer = Dense(100,
                       activation='elu', kernel_initializer = 'he_normal')(denseLayer)

    # Dense layer 3
    denseLayer = Dense(50,
                       activation='elu', kernel_initializer = 'he_normal')(denseLayer)

    # Dense layer 4
    denseLayer = Dense(10,
                       activation='elu', kernel_initializer = 'he_normal')(denseLayer)

    # Dense layer 5
    denseLayer = Dense(1, kernel_initializer = 'he_normal')(denseLayer)
    model = Model(inputs=[input_img1], outputs=denseLayer)

    # Compilation
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="mse")

    return model

def make_model(inputShape=(250,250)):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(3,3), strides=(1,1), input_shape = (inputShape[1], inputShape[0], 3), activation='elu', kernel_initializer = 'he_normal'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
  model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', activation='elu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
  model.add(tf.keras.layers.Conv2D(filters = 48, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', activation='elu'))
  model.add(tf.keras.layers.Dropout(0.3))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

  model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', activation='elu'))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
  # model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', padding = 'valid', activation='elu'))
  # model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, kernel_initializer = 'he_normal', activation='elu'))
  model.add(tf.keras.layers.Dense(64, kernel_initializer = 'he_normal', activation='elu'))
  model.add(tf.keras.layers.Dense(16, kernel_initializer = 'he_normal', activation='elu'))
  model.add(tf.keras.layers.Dense(1, kernel_initializer = 'he_normal'))

  adam=tf.keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(loss='mse',optimizer='adam')
  return model


if __name__ == '__main__':
  split=0.9
  targetSize=(240, 320)
  EPOCH = 10
  batch_size = 128

  filenames_original = [img for img in glob.glob(PATH_TO_TRAIN_DIR + "*.jpg")]
  filenames_original.sort() # ADD THIS LINE

  groundTruth_original = np.loadtxt(PATH_TO_TRAIN_OUTPUT_DATASET)
  print(len(filenames_original),groundTruth_original[0])
  groundTruth_original = groundTruth_original[1:]
  filenames_original, groundTruth_original = sklearn.utils.shuffle(filenames_original[1:], groundTruth_original)

  length = len(filenames_original)
  train_samples=int(length*split)

  filenames_train=filenames_original[:train_samples]
  groundTruth_train=groundTruth_original[:train_samples]
  filenames_valid=filenames_original[train_samples:]
  groundTruth_valid=groundTruth_original[train_samples:]

  trainDataGen = dataGenerator(filenames_train, groundTruth_train, shuffle=True, batchSize=batch_size, targetSize=targetSize, dataset="train")
  validDataGen = dataGenerator(filenames_valid, groundTruth_valid, shuffle=True, batchSize=32, targetSize=targetSize, dataset="valid")

  log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  custom_callback = CustomCallback()
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  early_stopping_callback =tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
  callbacks=[tensorboard_callback, early_stopping_callback, custom_callback]

  # model = tf.keras.models.load_model(PATH_TO_MODEL)
  if(sys.argv[1]!="co"):
    print("Normal")
    model = make_model(inputShape=targetSize)
  else:
    print("contextual")
    model = setupCoNvidiaModel(inputShape=targetSize)

  history = model.fit(x=trainDataGen, 
    steps_per_epoch=len(filenames_train)//batch_size, 
    validation_data=validDataGen,
    validation_steps=len(filenames_valid)//32,
    shuffle=True ,
    epochs=EPOCH,
    callbacks=callbacks)

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))

  import matplotlib.pyplot as plt

# plt.plot(loss_[2:])
# plt.plot(val_loss_[2:])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.savefig(sys.argv[1]+'.png')


  plt.plot(epochs, loss, 'r', label='Training Loss')
  plt.plot(epochs, val_loss, 'b', label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.legend(loc=0)
  plt.savefig(sys.argv[1]+".png")
  model.save('data/trained_model')

  # TODO: 
  # 1. Save model 
  # 2. Save model after 10 epochs or 10 minutes

  print(model.summary())