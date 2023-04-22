import json
import math
import statistics
import cv2 as cv
import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image

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


def make_tensors(coordinates_dict):
    """
    Transforms a dictionary of image coordinates into a dictionary of TensorFlow constants.

    :param coordinates_dict: A dictionary containing image coordinates in the form <image_name, coordinates>, where
                       coordinates is a list of xy pairs.
    :return: A dictionary of the same shape as coordinates_dict, where all coordinate lists have the same size and shape.
             Any empty spaces are filled with [-1, -1].

    """

    max_boxes = max([len(v) for v in coordinates_dict.values()])
    bbox_tensors = {}
    for key in coordinates_dict:
        coordinates_list = coordinates_dict[key]
        bbox_array = [[-1, -1] for i in range(max_boxes)]
        for i, bbox in enumerate(coordinates_list):
            bbox_array[i] = bbox
        bbox_tensor = tf.constant(bbox_array, dtype=tf.float32)
        bbox_tensors[key] = bbox_tensor
    return bbox_tensors


def create_dataset(images_folder, y_train_dict, target_size=(256, 256)):
    """
    Transforms a dictionary of images and their corresponding coordinates into input tensors.

    :param images_folder: the folder of the image dataset
    :param y_train_dict: A dictionary where each key is the name of an image file and its corresponding value is a numpy
    array of shape (n,2) containing the coordinates of n points in the image. If there are fewer than n points, the empty
    coordinates are marked as [-1, -1].
    :param target_size: The size to which each image in the dataset should be resized, default is (256, 256).
    :return: A tuple containing two tensorflow tensors. The first tensor is a list of image arrays that have been
    normalized between 0 and 1 and resized to target_size. The second tensor is a list of corresponding y_train tensors
    where each tensor has shape (n, 2) and is normalized between 0 and 1. If there are fewer than n points, the empty
    coordinates are marked as [-1, -1].
    """

    x_train = []
    y_train = []

    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)

        # Load the image using PIL
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            continue

        # Resize the image
        image = image.resize(target_size)

        # Convert the image to a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        image_array = image_array / 255.0

        # Append the image array to the x_train list
        x_train.append(image_array)

        # Get the corresponding y_train tensor
        y_tensor = y_train_dict.get(image_name, None)

        if y_tensor is not None:
            # Normalize the y_train tensor
            y_tensor = y_tensor / np.array(target_size[::-1])
            print("test:" + str(y_tensor))
            # Append the normalized y_train tensor to the y_train list
            y_train.append(y_tensor)

    # Convert the x_train and y_train lists to tensorflow tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    return x_train, y_train