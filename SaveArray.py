from __future__ import print_function
import cv2.cv2 as cv2
import glob
import numpy as np
import pathlib
import datetime

PATH_TO_TRAIN_DIRECTORY = "data/training/train/"
PATH_TO_TEST_DIRECTORY = "data/images/test/"
PATH_TO_GROUND_TRUTH = "data/train.txt"

CURRENT_DIRECTORY = PATH_TO_TRAIN_DIRECTORY

SAVE_ARRAY_NAME = "train"

data_dir = pathlib.Path(PATH_TO_TRAIN_DIRECTORY)

MAX_SAMPLE_LIMIT = 20400

no_of_samples = 10

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class readData:
    images = []
    output = []
    def __init__(self):
        self.filenames = [img for img in glob.glob(CURRENT_DIRECTORY + "*.jpg")]
        self.filenames.sort() # ADD THIS LINE
        
        global no_of_samples
        no_of_samples = np.shape(self.filenames)[0]

        print("Total images: " + str(no_of_samples))

    def read_data(self):
        self.output = np.loadtxt(PATH_TO_GROUND_TRUTH)
        prvs = None
        hsv = None

        for i in range (0, no_of_samples):
            if i==0 :
                prvs = cv2.imread(self.filenames[i]) 
                prvs_gray = cv2.cvtColor(prvs, cv2.COLOR_RGB2GRAY)
                continue

            curr = cv2.imread(self.filenames[i]) 
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs_gray,curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            hsv = np.zeros_like(prvs)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,1] = 255
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            rgb = rgb.astype(np.float32)
            rgb = cv2.resize(rgb, (224,224))

            self.images.append(rgb)
            prvs = curr

            print("Reading image number: {}".format(i), end='\r')
            # cv2.imshow('frame2',rgb)
            # print(np.shape(self.images))
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break

if __name__ == '__main__':
    trainData = readData()
    trainData.read_data()

    np.save(SAVE_ARRAY_NAME, trainData.images)
    data = np.load(SAVE_ARRAY_NAME+".npy")
    print(np.shape(data))