import json
import math
import statistics

import cv2 as cv
import tensorflow as tf

import sys

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

discovered_plugins = entry_points(group="myapp.plugins")


def extract_images(pathIn, pathOut, verbose=0):
    """
    Extracts and saves video frames every 1/4 of a second
    :param pathIn: Path of the video to read frames from
    :param pathOut: Path of the output folder where the resulting frames will be saved
    :param verbose: Verbose is 0 by default, which gives no information
    Verbose is 1 for additional information while function is running
    :return: none
    """
    count = 0
    vidcap = cv.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        vidcap.set(cv.CAP_PROP_POS_MSEC, (count * 250))  # added this line
        success, image = vidcap.read()
        if verbose == 1:
            print("Read a new frame: ", success)
        if success:
            cv.imwrite(
                pathOut + "/frame%d.jpg" % count, image
            )  # save frame as JPEG file
        count = count + 1


def read_json(path, verbose=0):
    """
    Reads and converts the VGG Image Annotator json to a dictionary
    :param path: Path of the json file
    :param verbose: Verbose is 0 by default, gives no extra information
    Verbose is 1 to give the dictionary entries
    :return: Dictionary structure <Name of File, Coordinates>
    """
    f = open(path)
    data = json.load(f)
    d = {}

    for frame in data:
        points = []
        for shap_att in data[frame]["regions"]:
            points.append(
                [shap_att["shape_attributes"]["cx"], shap_att["shape_attributes"]["cy"]]
            )
        d[data[frame]["filename"]] = points
        if verbose == 1:
            print("name:" + data[frame]["filename"] + " points: " + str(points))
    f.close()
    return d


def find_floor(d):
    """
    Finds the lower coordinate point in an image using the dictionary structure from read_json function

    :param d: Dictionary structure <Name of File, Coordinates>
    :return: Dictionary structure <Name of File, Floor Coordinates>
    """
    ymax = 0
    floor_dict = {}
    for name in d.keys():
        value = d.get(name)
        for coord in value:
            if ymax < coord[1]:
                ymax = coord[1]
        floor_dict[name] = ymax
    return floor_dict


def measure_length(d, points_to_measure, pixel_irl=None, verbose=0, mode="None"):
    """
    Measures the length between coordinates according to the points_to_measure list
    If you want to measure the distances between point 1 and point 2, and point 3 and point 4
    the points to measure array should have this value [[0,1],[2,3]]
    :param d: Dictionary structure <Name of File, Coordinates>
    :param points_to_measure: List of pairs to measure inbetween
    :param pixel_irl: The real life measurement of a pixel
    :param verbose: verbose = 0 for zero informational strings during the function's execution
    verbose = 1 for informational output with the final dictionary elements
    verbose = 2 only for when exceptions occur
    :param mode: How to calculate the final distances across all data points,
    mode='avg' for calculating the average across each distance calculated
    mode='median' for calculating the median across each distance calculated
    :return: Dictionary structure <Name of File, Distances> and Average value - average value is none when mode is not set or 'None'
    """
    dist_dict = {}
    for name in d.keys():
        value = d.get(name)
        distances = []
        for p in points_to_measure:

            try:
                dist = math.dist(value[p[0]], value[p[1]])
                if pixel_irl:
                    dist = dist * pixel_irl
                distances.append(dist)
                dist_dict[name] = distances
            except IndexError:
                continue

    if verbose == 1:
        for name in dist_dict.keys():
            print("Key: " + name + " Value: " + str(dist_dict.get(name)))

    if mode == "avg":
        avg = [0] * len(points_to_measure)

        for value in dist_dict.values():
            for i in range(len(value)):
                avg[i] = avg[i] + value[i]

        div = [0] * len(points_to_measure)
        for values in dist_dict.values():
            l = len(values)
            for i in range(l):
                div[i] = div[i] + 1

        for i in range(len(avg)):
            if div[i] != 0:
                avg[i] = avg[i] / div[i]

        if verbose == 1:
            print("Averages: " + str(avg))
        return dist_dict, avg

    if mode == 'median':
        m = [0] * len(points_to_measure)
        m_arrays = [0] * len(d), [0] * len(points_to_measure)
        print(m)
        print(m_arrays)

        for value in dist_dict.values():
            for i in range(len(value)):
                for j in range(len(m_arrays[i])):
                    if m_arrays[i][j] == 0:
                        m_arrays[i][j] = value[i]
                        break

        for i in range(len(m_arrays)):
            m[i] = statistics.median(m_arrays[i])

        return dist_dict, m

    return dist_dict, None


def make_tensors(d):
    """
    Transforms the dictionary into input tensors
    :param d: dictionary of shape <name_of_file, xy_coordinates>
    :return: a dictionary of the same shape but all xy_coordinates list have the same size and shape, any empty spaces
    are filled with [-1, -1]
    All values are tensorflow constants
    """

    values = list(d.values())
    longest = 0
    for value in values:
        longest = max(longest, len(value))

    for key in d.keys():
        if len(d[key]) != longest:
            d[key].append([-1, -1])
        d[key] = tf.constant(d[key])

    return d
