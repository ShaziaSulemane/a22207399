import tensorflow as tf
import cv2 as cv
import numpy as np

def read_data(folder, json):


def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1