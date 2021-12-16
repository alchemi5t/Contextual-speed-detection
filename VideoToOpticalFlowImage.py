from __future__ import print_function
import cv2.cv2 as cv2
import glob
import numpy as np
import pathlib
import datetime

PATH_TO_TRAIN_VIDEO = 'data/train.mp4'
PATH_TO_TEST_VIDEO = 'data/test.mp4'
PATH_TO_TRAIN_DIRECTORY = "data/images/train/"
PATH_TO_TEST_DIRECTORY = "data/images/test/"
PATH_TO_GROUND_TRUTH = "data/train.txt"

PATH_TO_VIDEO = PATH_TO_TEST_VIDEO
PATH_TO_SAVE_DIRECTORY = PATH_TO_TEST_DIRECTORY

cap = cv2.VideoCapture(PATH_TO_VIDEO)

class ReadData:
    def read_data(self):
      print("Hello train")
      prvs = None
      hsv = None
      num_img = 0
      while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
          if (num_img==0):
            num_img += 1
            prvs = frame.copy()
            prvs_gray = cv2.cvtColor(prvs, cv2.COLOR_RGB2GRAY)
            continue
          curr = frame.copy()
          curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)  
          flow = cv2.calcOpticalFlowFarneback(prvs_gray,curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)	  			
          hsv = np.zeros_like(prvs)
          mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
          hsv[...,0] = ang*180/np.pi/2
          hsv[...,1] = 255
          hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
          rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

          number_str = str(num_img)				
          print("Saving file number: {}".format(number_str))
          zero_filled_number = number_str.zfill(5)
          cv2.imwrite(PATH_TO_SAVE_DIRECTORY+zero_filled_number+'.jpg', rgb)
          num_img += 1
          
          prvs = curr
          # cv2.imshow('frame2',rgb)
          # k = cv2.waitKey(30) & 0xff
          # if k == 27:
            # break
        else: 
          break

if __name__ == '__main__':
    trainData = ReadData()
    trainData.read_data()
    print("Done ")