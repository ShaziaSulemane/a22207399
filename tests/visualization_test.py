import os
from image_network_library import visualization as vs

pwd_path = os.path.dirname(os.path.abspath(__file__))


def test_show_shapes():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}

    path = pwd_path + "/test_img/"
    dict_images = vs.show_shapes(mydict, [[0, 1], [1, 2], [2, 3]], path)
    assert len(dict_images) == 3


def test_show_shapes_random():
    mydict = {'frame8.jpg': [[1128, 603], [1158, 246], [1380, 262]],
              'frame7.jpg': [[1131, 603], [1253, 258], [1476, 252]],
              'frame6.jpg': [[1131, 603], [1253, 258]]}

    path = pwd_path + "/test_img/"
    dict_images = vs.show_shapes_random(mydict, [[0, 1], [1, 2], [2, 3]], path, how_many=1)
    assert len(dict_images) == 1


def test_draw_floor():
    mydict = {'frame8.jpg': 603,
              'frame7.jpg': 253,
              'frame6.jpg': 500}

    path = pwd_path + "/test_img/"
    dict_images = vs.draw_floor(mydict, path)
    assert len(dict_images) == 3
