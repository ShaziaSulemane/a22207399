import random

import cv2 as cv
import matplotlib.pyplot as plt


def show_shapes(d, connections, path, color=(0, 255, 0), thickness=5):
    """
    From the connections list of arrays it connects the different coordinates in the dataset dictionary
    Shows ALL images with the connections made, singularly. Press any key to go to the next entry in dict
    :param thickness: thickness of the lines
    :param color: color of the lines
    :param d: dictionary with structure <name_of_file, array_of_coordinates>
    :param connections: list of arrays describing which xy points connect to each other by the order
    they are represented in the values list of dict
    :param path: the path to the folder where the images described in the keys of dict
    :return dict_img: dictionary shape <img_name, img_with_line>
    """
    dict_img = {}
    for key in d.keys():
        full_path = path + key
        img = cv.imread(full_path)
        values = d[key]
        for i in connections:
            try:
                img = cv.line(img, values[i[0] - 1], values[i[1] - 1], color, thickness)
            except IndexError:
                continue
        dict_img[key] = img

    return dict_img


def show_shapes_random(
        d, connections, path, color=(0, 255, 0), thickness=5, how_many=1
):
    """
    From the connections list of arrays it connects the different coordinates in the dataset dictionary
    Shows ALL images with the connections made. Press any key to go to the next entry in dict
    :param how_many: how many images to show simultaneously instead of showing all singularly
    :param thickness: thickness of the lines
    :param color: color of the lines
    :param d: dictionary with structure <name_of_file, array_of_coordinates>
    :param connections: list of arrays describing which xy points connect to each other by the order
    they are represented in the values list of dict
    :param path: the path to the folder where the images described in the keys of dict
    :return: dict_img: dictionary shape <img_name, img_with_line> size = how_may parameter
    """
    dict_img = {}
    for x in range(0, how_many):
        key = random.choice(list(d.keys()))
        full_path = path + key
        img = cv.imread(full_path)
        values = d[key]
        for i in connections:
            try:
                img = cv.line(
                    img, values[i[0] - 1], values[i[1] - 1], color, thickness
                )
            except IndexError:
                continue
        dict_img[key] = img

    return dict_img


def draw_floor(d, img_path, floor_thickness=9, floor_color=(0, 255, 0)):
    """
    Visualise the different floor coordinates pointed by find_floor
    :param d: Dictionary structure <Name of File, Floor Coordinates>
    :param img_path: Folder of the images in the dictionary structure
    :param floor_thickness: Pixel thickness for visualizing floor
    :param floor_color: Color of the floor line
    :return: Image with floor line
    """
    dict_img = {}
    for name in d.keys():
        path = img_path + name
        img = cv.imread(path)
        h, w = img.shape[:2]
        init_coord = [0, d.get(name)]
        end_coord = [w, d.get(name)]
        img = cv.line(img, init_coord, end_coord, floor_color, floor_thickness)
        dict_img[name] = img

    return dict_img


def plot_loss_curves(history):
    """

    :param history:
    :return:
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["acc"]
    val_accuracy = history.history["val_acc"]

    epochs = range(len(history.history["loss"]))

    # plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="validation_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot acc
    plt.figure()
    plt.plot(epochs, accuracy, label="accuracy")
    plt.plot(epochs, val_accuracy, label="validation accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
