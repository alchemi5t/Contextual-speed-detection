from __future__ import print_function
import cv2.cv2
import numpy

PATH_TO_VIDEO = 'data/train.mp4'
PATH_TO_SAVE_DIRECTORY = "data/images/train/"

cap = cv2.VideoCapture(PATH_TO_VIDEO)

num_img = 0

print("Converting the video")
print("\nSaving to {}\n\n".format(PATH_TO_SAVE_DIRECTORY))

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        num_img += 1
        number_str = str(num_img)
        
        print("Saving file number: {}".format(number_str), end='\r')

        zero_filled_number = number_str.zfill(5)
        cv2.imwrite(PATH_TO_SAVE_DIRECTORY+zero_filled_number+'.jpg', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break
    
print("Successfully saved {} images".format(num_img))