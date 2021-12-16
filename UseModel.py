import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

PATH_TO_MODEL = "speed_recognition"
PATH_TO_TEST_IMAGE_NPY = "test.npy"

model = tf.keras.models.load_model(PATH_TO_MODEL)
model.summary()

test_images = np.load(PATH_TO_TEST_IMAGE_NPY)
test_images = np.reshape(test_images, (np.shape(test_images)[0], 224, 224, 3))
prediction = model.predict(test_images)
print(prediction.shape)
print(prediction)