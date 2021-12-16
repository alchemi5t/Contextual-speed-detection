# Contextual-speed-detection

In the work, we aim to estimate the speed of a moving vehicle. The input to the model is a dash camera video feed and the output is the predicted speed. We use Gunnar Farneback optical flow as feature extractor and train a baseline Convolutional Neural Network model and a Contextual Convolution model with the same parameters. Our results show that contextual convolutions improve the performance of the speed detection system.

Usage:

Data preprocessing:
`python ./DatasetConverter.py`
Make sure the data path is set correctly and the data folder has the training video. This script will convert the video to images.

`python VideoToOpticalFlowImage.py`
This script will convert the images into Optical flow frames.


`python train.py [norm|co]`
This script will train the model. command line argument norm will train a Vanilla Nvidia CNN. commandline argument co will train the contextual variant of Nvidia's CNN.

`python SaveResults.py`
This script can be used to generate the output frames for the gif used in the presentation. each frame will be the input optical flow, and it will have the ground truth and predicted value on the image.


Acknowledgement:
Basline model credits : https://github.com/MahimanaGIT/SpeedPrediction_Comma.AI
