import json
import math

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
    ymax = 0
    for name in dict.keys():
        value = dict.get(name)
        for coord in value:
            if ymax < coord[1]:
                ymax = coord[1]
        dict[name] = ymax
    return dict


def draw_floor(dict, img_path, floor_thickness=9, floor_color=(0, 255, 0)):
    for name in dict.keys():
        path = img_path + name
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        h, w = img.shape
        init_coord = [0, dict.get(name)]
        end_coord = [w, dict.get(name)]
        cv.line(img, init_coord, end_coord, floor_color, floor_thickness)
        cv.imshow("image", img)
        cv.waitKey(0)


def measure_lenght(dict, points_to_measure, pixel_irl=None):
    for name in dict.keys():
        value = dict.get(name)
        dist = []
        for p in points_to_measure:
            if value:
                try:
                    d = math.dist(value[p[0]], value[p[1]])
                    if pixel_irl:
                        d = d * pixel_irl
                    dist.append(d)
                    dict[name] = dist
                except:
                    print("Exception Occured")
                print("Distance: " + str(dist))




def main():
    video_folder = "/home/shazia/PycharmProjects/a22207399/videos/VID_20210625_100510.mp4"
    dataset_folder = "/home/shazia/PycharmProjects/a22207399/dataset/"
    json_notations = ""
    d = read_json("/home/shazia/PycharmProjects/a22207399/dataset/via_project_6Nov2022_10h31m_json.json")
    # floors = find_floor(d)
    # draw_floor(floors, dataset_folder)
    measure_lenght(d, [[0, 1], [1, 2]])


if __name__ == "__main__":
    main()
