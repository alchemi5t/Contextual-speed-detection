from __future__ import print_function
import cv2.cv2 as cv2
import glob
import numpy as np
import pathlib
import tensorflow as tf 
import datetime

PATH_TO_TRAIN_DIRECTORY = "data/images/train/"
PATH_TO_TEST_DIRECTORY = "data/images/test/"
PATH_TO_GROUND_TRUTH = "data/train.txt"

PATH_TO_DIRECTORY = PATH_TO_TEST_DIRECTORY

PATH_TO_MODEL = "speed_recognition.PB"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

result_arr = []

class Predict:	
    def predict(self, model, target_size):
        if(PATH_TO_DIRECTORY == PATH_TO_TRAIN_DIRECTORY):
            ground_truth = np.loadtxt(PATH_TO_GROUND_TRUTH)

        print(target_size)
        filenames = [img for img in glob.glob(PATH_TO_DIRECTORY + "*.jpg")]
        filenames.sort() # ADD THIS LINE
        no_of_samples = np.shape(filenames)[0]
        for i in range (0, no_of_samples):
            rgb = cv2.imread(filenames[i]) 
            image = rgb.copy()
            image = cv2.resize(image, target_size)
            image = image[np.newaxis, ...]
            prediction = model.predict(image)[0][0]
            result_arr.append(prediction)

            if(PATH_TO_DIRECTORY == PATH_TO_TRAIN_DIRECTORY):
                text = "GT: " + str(ground_truth[i+1]) + ", Predict: " + str(prediction)
            else:
                text = "Predict: " + str(prediction)				
			
            cv2.putText(rgb, text,(50, 50) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 0) , 2, cv2.LINE_AA)

            cv2.imshow('frame',rgb)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

if __name__ == '__main__':
    model = tf.keras.models.load_model(PATH_TO_MODEL)
    target_size=(model.layers[0].input_shape[2],model.layers[0].input_shape[1])
    print(target_size)
    model.summary()
	
    predict = Predict()
    predict.predict(model, target_size)
    if(PATH_TO_DIRECTORY == PATH_TO_TRAIN_DIRECTORY):
        f_name = "train_predict.txt"
    else:
        f_name = "test_predict.txt"
    np.savetxt(f_name, result_arr)
    print("\n\n\n***********\n\nOperation Done\n\n***********")