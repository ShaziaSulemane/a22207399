import json

import tensorflow
import tensorflow as tf
import cv2 as cv
import numpy as np


def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        vidcap.set(cv.CAP_PROP_POS_MSEC, (count * 250))  # added this line
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if success:
            cv.imwrite(pathOut + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        count = count + 1


def read_json(path):
    f = open(path)
    data = json.load(f)
    dict = {}

    for frame in data:
        points = []
        for shap_att in data[frame]['regions']:
            points.append([shap_att['shape_attributes']['cx'], shap_att['shape_attributes']['cy']])
        dict[data[frame]['filename']] = points
        print("name:" + data[frame]['filename'] + " points: " + str(points))

    return dict


def find_floor(dict):
    for value in dict.values():


def main():
    video_folder = "/home/shazia/PycharmProjects/a22207399/videos/VID_20210625_100510.mp4"
    dataset_folder = "/home/shazia/PycharmProjects/a22207399/dataset"
    json_notations = ""
    d = read_json("/home/shazia/PycharmProjects/a22207399/dataset/via_project_6Nov2022_10h31m_json.json")
    find_floor(d)

if __name__ == "__main__":
    main()
