This project takes facial expression data of faces (FER2013, a greyscale image data set)
and classifies the emotion of the images.

The model reaches the benchmark for FER2013 with 68 % accuracy (human accuracy: 60-65 %) $\rightarrow$ see paper: https://arxiv.org/pdf/1307.0414.pdf.

The code is based on a former Tensorflow tutorial for deep CNNs with CIFAR-10 as dataset (online not available anymore). 

This work was done in 2017 and requires
- Tensorflow 1.3
- Python 3

and using a GPU is highly recommended.  
To test the application, you can take a photo with your webcam to predict your emotion.  

The data can be found here: https://www.kaggle.com/nicolejyt/facialexpressionrecognition
(original data source: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 

